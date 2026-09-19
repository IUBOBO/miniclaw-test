import fs from 'node:fs';

import { z } from 'zod';

import {
  fileSha256,
  isWithin,
  normalizeRelativePath,
  PathOutsideWorkspaceError,
  readLimitedWithinRoot,
  RESEARCH_WORKSPACE_FILE,
  resolveWithinRoot,
} from './workspace-loader.js';

const MAX_CONFIG_BYTES = 256 * 1024;
const MAX_MANIFEST_BYTES = 10 * 1024 * 1024;
const MAX_CODE_BYTES = 2 * 1024 * 1024;
const MAX_EXCERPT_CHARS = 1_200;
const SHA256_PATTERN = /^[a-f0-9]{64}$/i;

const modelSchema = z
  .object({
    code: z.string().min(1),
  })
  .passthrough();

const codeWorkspaceSchema = z
  .object({
    schema_version: z.literal(2),
    root: z.literal('.'),
    workspace_id: z.string().min(1),
    snapshot_id: z.string().min(1),
    manifest: z.string().min(1),
    models: z.record(z.string(), modelSchema),
  })
  .passthrough();

const sourceManifestEntrySchema = z
  .object({
    local_path: z.string().min(1),
    model: z.string().min(1),
    kind: z.string().min(1),
    size: z.number().int().nonnegative(),
    sha256: z.string().regex(SHA256_PATTERN),
  })
  .passthrough();

export type ResearchCodeLookupIssueCode =
  | 'WORKSPACE_CONFIG_INVALID'
  | 'MANIFEST_INVALID'
  | 'MODEL_NOT_FOUND'
  | 'PATH_OUTSIDE_WORKSPACE'
  | 'SOURCE_FILE_MISSING'
  | 'SOURCE_SIZE_MISMATCH'
  | 'SOURCE_HASH_MISMATCH'
  | 'SOURCE_FILE_TOO_LARGE'
  | 'SOURCE_FILE_BINARY'
  | 'SOURCE_READ_FAILED';

export interface ResearchCodeLookupIssue {
  code: ResearchCodeLookupIssueCode;
  message: string;
  model?: string;
  path?: string;
}

export interface ResearchCodeLookupMatch {
  model: string;
  matchType: 'definition' | 'reference';
  symbol: string | null;
  symbolLine: number | null;
  matchedTerms: string[];
  matchLine: number;
  lineStart: number;
  lineEnd: number;
  score: number;
  excerpt: string;
  source: {
    path: string;
    sha256: string;
    snapshotId: string;
    location: string;
  };
}

export interface ResearchCodeLookupResult {
  status: 'ok' | 'no_matches' | 'error';
  workspaceId: string | null;
  snapshotId: string | null;
  query: string;
  requestedModel: string | null;
  searchedModels: string[];
  searchedFiles: number;
  matches: ResearchCodeLookupMatch[];
  issues: ResearchCodeLookupIssue[];
}

interface SourceFile {
  model: string;
  relativePath: string;
  realPath: string;
  size: number;
  sha256: string;
}

interface CodeCatalog {
  workspaceId: string;
  snapshotId: string;
  models: string[];
  sourceFiles: SourceFile[];
}

function issue(
  code: ResearchCodeLookupIssueCode,
  message: string,
  details: Pick<ResearchCodeLookupIssue, 'model' | 'path'> = {},
): ResearchCodeLookupIssue {
  return { code, message, ...details };
}

function emptyResult(
  query: string,
  requestedModel?: string,
): ResearchCodeLookupResult {
  return {
    status: 'error',
    workspaceId: null,
    snapshotId: null,
    query,
    requestedModel: requestedModel ?? null,
    searchedModels: [],
    searchedFiles: 0,
    matches: [],
    issues: [],
  };
}

function isPathInsideDirectory(filePath: string, directory: string): boolean {
  return filePath === directory || filePath.startsWith(directory + '/');
}

function loadCodeCatalog(workspaceRoot: string): CodeCatalog {
  const rootReal = fs.realpathSync(workspaceRoot);
  if (!fs.statSync(rootReal).isDirectory())
    throw new Error('workspace_invalid');

  const configFile = resolveWithinRoot(rootReal, RESEARCH_WORKSPACE_FILE);
  if (!configFile) throw new Error('workspace_outside');
  const workspace = codeWorkspaceSchema.parse(
    JSON.parse(
      readLimitedWithinRoot(rootReal, configFile.path, MAX_CONFIG_BYTES),
    ) as unknown,
  );

  const codeRoots = new Map<string, string>();
  for (const [model, config] of Object.entries(workspace.models)) {
    const normalized = normalizeRelativePath(config.code);
    if (!normalized) throw new Error('workspace_outside');
    const resolved = resolveWithinRoot(rootReal, normalized);
    if (!resolved) throw new Error('workspace_outside');
    codeRoots.set(model, normalized);
  }

  const manifestFile = resolveWithinRoot(rootReal, workspace.manifest);
  if (!manifestFile) throw new Error('manifest_outside');
  const manifestLines = readLimitedWithinRoot(
    rootReal,
    manifestFile.path,
    MAX_MANIFEST_BYTES,
  )
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  const seenPaths = new Set<string>();
  const sourceFiles: SourceFile[] = [];
  for (const line of manifestLines) {
    const entry = sourceManifestEntrySchema.parse(JSON.parse(line) as unknown);
    const normalized = normalizeRelativePath(entry.local_path);
    if (!normalized) throw new Error('manifest_outside');
    if (seenPaths.has(normalized)) throw new Error('manifest_invalid');
    seenPaths.add(normalized);
    if (entry.kind !== 'source_code') continue;

    const codeRoot = codeRoots.get(entry.model);
    if (!codeRoot || !isPathInsideDirectory(normalized, codeRoot)) continue;
    const resolved = resolveWithinRoot(rootReal, normalized);
    if (!resolved) throw new Error('manifest_outside');

    sourceFiles.push({
      model: entry.model,
      relativePath: normalized,
      realPath: resolved.path,
      size: entry.size,
      sha256: entry.sha256.toLowerCase(),
    });
  }

  return {
    workspaceId: workspace.workspace_id,
    snapshotId: workspace.snapshot_id,
    models: [...codeRoots.keys()].sort(),
    sourceFiles: sourceFiles.sort((a, b) =>
      a.relativePath.localeCompare(b.relativePath),
    ),
  };
}

function queryTerms(query: string): string[] {
  const normalized = query.normalize('NFKC').toLocaleLowerCase();
  const tokens = normalized
    .split(/[^\p{L}\p{N}_-]+/u)
    .map((term) => term.trim())
    .filter(Boolean);
  return [...new Set(tokens.length > 0 ? tokens : [normalized])];
}

function symbolAt(
  lines: string[],
  zeroBasedLine: number,
): { name: string; line: number; kind: 'class' | 'function' } | null {
  let nearest: {
    name: string;
    line: number;
    kind: 'class' | 'function';
  } | null = null;
  const pattern =
    /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(class|def|function)\s+([A-Za-z_$][\w$]*)/;
  for (let index = 0; index <= zeroBasedLine; index += 1) {
    const match = lines[index].match(pattern);
    if (!match) continue;
    nearest = {
      name: match[2],
      line: index + 1,
      kind: match[1] === 'class' ? 'class' : 'function',
    };
  }
  return nearest;
}

function boundedExcerpt(
  lines: string[],
  matchIndex: number,
): { lineStart: number; lineEnd: number; excerpt: string } {
  let start = Math.max(0, matchIndex - 2);
  let end = Math.min(lines.length - 1, matchIndex + 3);
  let excerpt = lines
    .slice(start, end + 1)
    .map((line, offset) => `${start + offset + 1}: ${line}`)
    .join('\n');
  while (excerpt.length > MAX_EXCERPT_CHARS && end > start) {
    end -= 1;
    excerpt = lines
      .slice(start, end + 1)
      .map((line, offset) => `${start + offset + 1}: ${line}`)
      .join('\n');
  }
  if (excerpt.length > MAX_EXCERPT_CHARS) {
    excerpt = excerpt.slice(0, MAX_EXCERPT_CHARS - 1).trimEnd() + '…';
  }
  return { lineStart: start + 1, lineEnd: end + 1, excerpt };
}

function matchesForSource(
  source: SourceFile,
  content: string,
  query: string,
  snapshotId: string,
): ResearchCodeLookupMatch[] {
  const terms = queryTerms(query);
  const exactQuery = query.normalize('NFKC').toLocaleLowerCase();
  const lines = content.split(/\r?\n/);
  const matches: ResearchCodeLookupMatch[] = [];
  const pathDepth = source.relativePath.split('/').length;
  const fileName =
    source.relativePath.split('/').at(-1)?.toLocaleLowerCase() ?? '';
  const canonicalFileBonus = fileName === 'model.py' ? 250 : 0;
  const likelyAuxiliaryPenalty = /(?:test|copy|backup|old)/.test(fileName)
    ? 150
    : 0;

  for (let index = 0; index < lines.length; index += 1) {
    const searchable = lines[index].normalize('NFKC').toLocaleLowerCase();
    const matchedTerms = terms.filter((term) => searchable.includes(term));
    const exact = searchable.includes(exactQuery);
    if (!exact && matchedTerms.length === 0) continue;

    const symbol = symbolAt(lines, index);
    const closeToDefinition =
      symbol !== null &&
      index + 1 >= symbol.line &&
      index + 1 - symbol.line <= 3;
    const definitionLine =
      /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:class|def|function)\s+/.test(
        lines[index],
      );
    let score = exact ? 1_000 : 0;
    score += matchedTerms.length * 120;
    if (matchedTerms.length === terms.length) score += 300;
    if (definitionLine || closeToDefinition) score += 400;
    score += Math.max(0, 160 - pathDepth * 20);
    score += canonicalFileBonus - likelyAuxiliaryPenalty;
    const excerpt = boundedExcerpt(lines, index);

    matches.push({
      model: source.model,
      matchType:
        definitionLine || closeToDefinition ? 'definition' : 'reference',
      symbol: symbol?.name ?? null,
      symbolLine: symbol?.line ?? null,
      matchedTerms,
      matchLine: index + 1,
      lineStart: excerpt.lineStart,
      lineEnd: excerpt.lineEnd,
      score,
      excerpt: excerpt.excerpt,
      source: {
        path: source.relativePath,
        sha256: source.sha256,
        snapshotId,
        location: `lines ${excerpt.lineStart}-${excerpt.lineEnd}`,
      },
    });
  }

  return matches;
}

export function lookupResearchCode(
  workspaceRoot: string,
  input: { query: string; model?: string; maxResults?: number },
): ResearchCodeLookupResult {
  const query = input.query.trim();
  const requestedModel = input.model?.trim() || undefined;
  const result = emptyResult(query, requestedModel);
  const maxResults = Math.min(Math.max(input.maxResults ?? 10, 1), 20);

  let catalog: CodeCatalog;
  try {
    catalog = loadCodeCatalog(workspaceRoot);
  } catch (error) {
    const message = error instanceof Error ? error.message : '';
    const outside =
      error instanceof PathOutsideWorkspaceError || message.includes('outside');
    result.issues.push(
      issue(
        outside
          ? 'PATH_OUTSIDE_WORKSPACE'
          : message.includes('manifest')
            ? 'MANIFEST_INVALID'
            : 'WORKSPACE_CONFIG_INVALID',
        outside
          ? '科研源码配置路径越过工作区边界。'
          : message.includes('manifest')
            ? '科研材料清单格式无效。'
            : '科研工作区或模型源码目录配置无效。',
      ),
    );
    return result;
  }

  result.workspaceId = catalog.workspaceId;
  result.snapshotId = catalog.snapshotId;
  if (requestedModel && !catalog.models.includes(requestedModel)) {
    result.issues.push(
      issue('MODEL_NOT_FOUND', '没有找到指定模型的源码登记。', {
        model: requestedModel,
      }),
    );
    return result;
  }

  const selectedFiles = requestedModel
    ? catalog.sourceFiles.filter((file) => file.model === requestedModel)
    : catalog.sourceFiles;
  result.searchedModels = requestedModel
    ? [requestedModel]
    : [...new Set(selectedFiles.map((file) => file.model))].sort();

  const allMatches: ResearchCodeLookupMatch[] = [];
  for (const source of selectedFiles) {
    let realPath: string;
    let stat: fs.Stats;
    try {
      realPath = fs.realpathSync(source.realPath);
      const rootReal = fs.realpathSync(workspaceRoot);
      if (!isWithin(rootReal, realPath)) {
        result.issues.push(
          issue(
            'PATH_OUTSIDE_WORKSPACE',
            '源码文件通过符号链接越过科研工作区边界。',
            { model: source.model, path: source.relativePath },
          ),
        );
        continue;
      }
      stat = fs.statSync(realPath);
      if (!stat.isFile()) throw new Error('not_file');
    } catch {
      result.issues.push(
        issue('SOURCE_FILE_MISSING', 'manifest 登记的源码文件不存在。', {
          model: source.model,
          path: source.relativePath,
        }),
      );
      continue;
    }

    if (stat.size > MAX_CODE_BYTES) {
      result.issues.push(
        issue('SOURCE_FILE_TOO_LARGE', '源码文件超过单文件读取上限。', {
          model: source.model,
          path: source.relativePath,
        }),
      );
      continue;
    }
    if (stat.size !== source.size) {
      result.issues.push(
        issue('SOURCE_SIZE_MISMATCH', '源码文件大小与 manifest 不一致。', {
          model: source.model,
          path: source.relativePath,
        }),
      );
      continue;
    }
    if (fileSha256(realPath) !== source.sha256) {
      result.issues.push(
        issue('SOURCE_HASH_MISMATCH', '源码文件 SHA-256 与 manifest 不一致。', {
          model: source.model,
          path: source.relativePath,
        }),
      );
      continue;
    }

    try {
      const content = readLimitedWithinRoot(
        fs.realpathSync(workspaceRoot),
        realPath,
        MAX_CODE_BYTES,
      );
      if (content.includes('\0')) {
        result.issues.push(
          issue('SOURCE_FILE_BINARY', 'manifest 登记的源码不是可检索文本。', {
            model: source.model,
            path: source.relativePath,
          }),
        );
        continue;
      }
      result.searchedFiles += 1;
      allMatches.push(
        ...matchesForSource(source, content, query, catalog.snapshotId),
      );
    } catch {
      result.issues.push(
        issue('SOURCE_READ_FAILED', '源码文件读取失败。', {
          model: source.model,
          path: source.relativePath,
        }),
      );
    }
  }

  result.matches = allMatches
    .sort(
      (a, b) =>
        b.score - a.score ||
        a.source.path.localeCompare(b.source.path) ||
        a.matchLine - b.matchLine,
    )
    .slice(0, maxResults);
  result.status =
    result.issues.length > 0
      ? 'error'
      : result.matches.length > 0
        ? 'ok'
        : 'no_matches';
  return result;
}
