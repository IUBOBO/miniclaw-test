import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import { parse as parseYaml } from 'yaml';
import { z } from 'zod';

import {
  fileSha256,
  isWithin,
  PathOutsideWorkspaceError,
  readLimitedWithinRoot,
  RESEARCH_WORKSPACE_FILE,
  resolveWithinRoot,
} from './workspace-loader.js';

const MAX_CONFIG_BYTES = 256 * 1024;
const MAX_METADATA_BYTES = 1024 * 1024;
const MAX_PDF_TEXT_BYTES = 20 * 1024 * 1024;
const PDF_EXTRACT_TIMEOUT_MS = 20_000;
const MAX_EXCERPT_CHARS = 700;

const searchWorkspaceSchema = z
  .object({
    schema_version: z.literal(2),
    root: z.literal('.'),
    workspace_id: z.string().min(1),
    snapshot_id: z.string().min(1),
    metadata_root: z.string().min(1),
    default_paper_id: z.string().min(1).optional(),
  })
  .passthrough();

const searchablePaperSchema = z
  .object({
    paper_id: z.string().min(1),
    title: z.string().min(1),
    local_file: z.string().min(1),
    sha256: z.string().regex(/^[a-f0-9]{64}$/i),
    text_extractable: z.boolean().optional().default(true),
  })
  .passthrough();

export type ResearchPaperSearchIssueCode =
  | 'WORKSPACE_CONFIG_INVALID'
  | 'PATH_OUTSIDE_WORKSPACE'
  | 'PAPER_METADATA_INVALID'
  | 'PAPER_NOT_FOUND'
  | 'PAPER_FILE_MISSING'
  | 'PAPER_HASH_MISMATCH'
  | 'PDF_EXTRACTOR_UNAVAILABLE'
  | 'PDF_EXTRACTION_FAILED';

export interface ResearchPaperSearchIssue {
  code: ResearchPaperSearchIssueCode;
  message: string;
  paperId?: string;
  path?: string;
}

export interface ResearchPaperSearchMatch {
  paperId: string;
  title: string;
  page: number;
  score: number;
  matchedTerms: string[];
  excerpt: string;
  source: {
    path: string;
    sha256: string;
    location: string;
  };
}

export interface ResearchPaperSearchResult {
  status: 'ok' | 'no_matches' | 'error';
  workspaceId: string | null;
  snapshotId: string | null;
  query: string;
  requestedPaperId: string | null;
  searchedPaperIds: string[];
  searchedPages: number;
  extractor: 'pdftotext';
  matches: ResearchPaperSearchMatch[];
  issues: ResearchPaperSearchIssue[];
}

interface SearchablePaper {
  paperId: string;
  title: string;
  relativePath: string;
  realPath: string;
  sha256: string;
}

interface PaperCatalog {
  rootReal: string;
  workspaceId: string;
  snapshotId: string;
  defaultPaperId?: string;
  papers: SearchablePaper[];
  issues: ResearchPaperSearchIssue[];
}

export type PdfPageExtractor = (pdfPath: string) => string[];

function issue(
  code: ResearchPaperSearchIssueCode,
  message: string,
  details: Pick<ResearchPaperSearchIssue, 'paperId' | 'path'> = {},
): ResearchPaperSearchIssue {
  return { code, message, ...details };
}

function emptyResult(
  query: string,
  requestedPaperId?: string,
): ResearchPaperSearchResult {
  return {
    status: 'error',
    workspaceId: null,
    snapshotId: null,
    query,
    requestedPaperId: requestedPaperId ?? null,
    searchedPaperIds: [],
    searchedPages: 0,
    extractor: 'pdftotext',
    matches: [],
    issues: [],
  };
}

function loadPaperCatalog(workspaceRoot: string): PaperCatalog {
  const rootReal = fs.realpathSync(workspaceRoot);
  if (!fs.statSync(rootReal).isDirectory()) throw new Error('not_directory');

  const configFile = resolveWithinRoot(rootReal, RESEARCH_WORKSPACE_FILE);
  if (!configFile) throw new Error('config_outside');
  const parsedConfig = JSON.parse(
    readLimitedWithinRoot(rootReal, configFile.path, MAX_CONFIG_BYTES),
  ) as unknown;
  const workspace = searchWorkspaceSchema.parse(parsedConfig);

  const papersDirectory = resolveWithinRoot(
    rootReal,
    workspace.metadata_root + '/papers',
  );
  if (!papersDirectory) throw new Error('papers_outside');
  const papersReal = fs.realpathSync(papersDirectory.path);
  if (!isWithin(rootReal, papersReal)) throw new Error('papers_outside');

  const papers: SearchablePaper[] = [];
  const issues: ResearchPaperSearchIssue[] = [];
  const metadataFiles = fs
    .readdirSync(papersReal)
    .filter((name) => /\.ya?ml$/i.test(name))
    .sort();

  for (const metadataName of metadataFiles) {
    const metadataRelative =
      workspace.metadata_root + '/papers/' + metadataName;
    const metadataFile = resolveWithinRoot(rootReal, metadataRelative);
    if (!metadataFile) {
      issues.push(
        issue('PATH_OUTSIDE_WORKSPACE', '论文元数据路径越过科研工作区边界。', {
          path: metadataRelative,
        }),
      );
      continue;
    }

    let metadata: z.infer<typeof searchablePaperSchema>;
    try {
      metadata = searchablePaperSchema.parse(
        parseYaml(
          readLimitedWithinRoot(
            rootReal,
            metadataFile.path,
            MAX_METADATA_BYTES,
          ),
        ),
      );
    } catch (error) {
      const outside = error instanceof PathOutsideWorkspaceError;
      issues.push(
        issue(
          outside ? 'PATH_OUTSIDE_WORKSPACE' : 'PAPER_METADATA_INVALID',
          outside
            ? '论文元数据通过符号链接越过科研工作区边界。'
            : '论文元数据格式或必填字段无效。',
          { path: metadataRelative },
        ),
      );
      continue;
    }

    if (!metadata.text_extractable) continue;
    const paperFile = resolveWithinRoot(rootReal, metadata.local_file);
    if (!paperFile) {
      issues.push(
        issue('PATH_OUTSIDE_WORKSPACE', '论文文件路径越过科研工作区边界。', {
          paperId: metadata.paper_id,
          path: metadata.local_file,
        }),
      );
      continue;
    }

    let paperReal: string;
    try {
      paperReal = fs.realpathSync(paperFile.path);
      if (!isWithin(rootReal, paperReal)) throw new Error('outside');
      if (!fs.statSync(paperReal).isFile()) throw new Error('not_file');
    } catch (error) {
      const outside = error instanceof Error && error.message === 'outside';
      issues.push(
        issue(
          outside ? 'PATH_OUTSIDE_WORKSPACE' : 'PAPER_FILE_MISSING',
          outside
            ? '论文文件通过符号链接越过科研工作区边界。'
            : '论文文件不存在或无法读取。',
          { paperId: metadata.paper_id, path: metadata.local_file },
        ),
      );
      continue;
    }

    let actualHash: string;
    try {
      actualHash = fileSha256(paperReal);
    } catch {
      issues.push(
        issue('PAPER_FILE_MISSING', '论文文件不存在或无法读取。', {
          paperId: metadata.paper_id,
          path: metadata.local_file,
        }),
      );
      continue;
    }
    if (actualHash !== metadata.sha256.toLowerCase()) {
      issues.push(
        issue('PAPER_HASH_MISMATCH', '论文文件 SHA-256 与元数据不一致。', {
          paperId: metadata.paper_id,
          path: metadata.local_file,
        }),
      );
      continue;
    }

    papers.push({
      paperId: metadata.paper_id,
      title: metadata.title,
      relativePath: paperFile.relativePath,
      realPath: paperReal,
      sha256: actualHash,
    });
  }

  return {
    rootReal,
    workspaceId: workspace.workspace_id,
    snapshotId: workspace.snapshot_id,
    defaultPaperId: workspace.default_paper_id,
    papers,
    issues,
  };
}

export function extractPdfPagesWithPdftotext(pdfPath: string): string[] {
  const result = spawnSync(
    'pdftotext',
    ['-layout', '-enc', 'UTF-8', pdfPath, '-'],
    {
      encoding: 'utf8',
      timeout: PDF_EXTRACT_TIMEOUT_MS,
      maxBuffer: MAX_PDF_TEXT_BYTES,
      windowsHide: true,
      shell: false,
    },
  );

  if (result.error) {
    const code = (result.error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') throw new Error('extractor_unavailable');
    throw new Error('extract_failed');
  }
  if (result.status !== 0 || typeof result.stdout !== 'string') {
    throw new Error('extract_failed');
  }

  const pages = result.stdout.replaceAll('\r\n', '\n').split('\f');
  if (pages.length > 0 && pages[pages.length - 1].trim() === '') pages.pop();
  if (pages.length === 0) throw new Error('extract_failed');
  return pages;
}

const QUERY_STOPWORDS = new Set([
  'about',
  'and',
  'describe',
  'find',
  'how',
  'paper',
  'the',
  'what',
  'where',
  '什么',
  '介绍',
  '如何',
  '描述',
  '查找',
  '论文',
  '请问',
]);

function queryTerms(query: string): string[] {
  const normalized = query.normalize('NFKC');
  const asciiTerms = normalized.match(/[A-Za-z][A-Za-z0-9_-]{1,}/g) ?? [];
  const chineseTerms = normalized.match(/\p{Script=Han}{2,}/gu) ?? [];
  const terms = [...asciiTerms, ...chineseTerms]
    .map((value) => value.toLocaleLowerCase())
    .filter((value) => !QUERY_STOPWORDS.has(value));
  if (terms.length === 0) terms.push(normalized.toLocaleLowerCase());
  return [...new Set(terms)].slice(0, 12);
}

function countOccurrences(text: string, term: string): number {
  let count = 0;
  let from = 0;
  while (from < text.length) {
    const index = text.indexOf(term, from);
    if (index < 0) break;
    count += 1;
    from = index + Math.max(term.length, 1);
  }
  return count;
}

function occurrenceIndices(text: string, term: string): number[] {
  const indices: number[] = [];
  let from = 0;
  while (from < text.length) {
    const index = text.indexOf(term, from);
    if (index < 0) break;
    indices.push(index);
    from = index + Math.max(term.length, 1);
  }
  return indices;
}

function bestEvidenceAnchor(
  searchable: string,
  terms: string[],
  exactQuery: string,
): number {
  const exactIndex = searchable.indexOf(exactQuery);
  if (exactIndex >= 0) return exactIndex;

  const occurrences = terms.flatMap((term) =>
    occurrenceIndices(searchable, term).map((index) => ({ index, term })),
  );
  if (occurrences.length === 0) return 0;

  let best = occurrences[0].index;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const candidate of occurrences) {
    const start = Math.max(0, candidate.index - 220);
    const end = Math.min(searchable.length, candidate.index + 480);
    const nearby = occurrences.filter(
      (item) => item.index >= start && item.index < end,
    );
    const distinct = new Set(nearby.map((item) => item.term)).size;
    const distance = nearby.reduce(
      (sum, item) => sum + Math.abs(item.index - candidate.index),
      0,
    );
    const score = distinct * 10_000 + nearby.length * 100 - distance;
    if (score > bestScore) {
      bestScore = score;
      best = candidate.index;
    }
  }
  return best;
}

function buildExcerpt(pageText: string, index: number): string {
  const start = Math.max(0, index - 220);
  const end = Math.min(pageText.length, index + 480);
  let excerpt = pageText.slice(start, end).replace(/\s+/g, ' ').trim();
  if (start > 0) excerpt = '…' + excerpt;
  if (end < pageText.length) excerpt += '…';
  if (excerpt.length > MAX_EXCERPT_CHARS) {
    excerpt = excerpt.slice(0, MAX_EXCERPT_CHARS - 1).trimEnd() + '…';
  }
  return excerpt;
}

function matchesForPaper(
  paper: SearchablePaper,
  pages: string[],
  query: string,
): ResearchPaperSearchMatch[] {
  const terms = queryTerms(query);
  const exactQuery = query.normalize('NFKC').toLocaleLowerCase();
  const matches: ResearchPaperSearchMatch[] = [];

  for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
    const pageText = pages[pageIndex];
    const searchable = pageText.normalize('NFKC').toLocaleLowerCase();
    const matchedTerms = terms.filter((term) => searchable.includes(term));
    const exactIndex = searchable.indexOf(exactQuery);
    if (matchedTerms.length === 0 && exactIndex < 0) continue;

    let score = exactIndex >= 0 ? 1000 : 0;
    for (const term of matchedTerms) {
      score += 100 + Math.min(countOccurrences(searchable, term), 20) * 10;
    }
    const anchor = bestEvidenceAnchor(searchable, matchedTerms, exactQuery);

    matches.push({
      paperId: paper.paperId,
      title: paper.title,
      page: pageIndex + 1,
      score,
      matchedTerms,
      excerpt: buildExcerpt(pageText, anchor),
      source: {
        path: paper.relativePath,
        sha256: paper.sha256,
        location: 'page ' + (pageIndex + 1),
      },
    });
  }

  return matches;
}

export function searchResearchPapers(
  workspaceRoot: string,
  input: { query: string; paperId?: string; maxResults?: number },
  extractPages: PdfPageExtractor = extractPdfPagesWithPdftotext,
): ResearchPaperSearchResult {
  const query = input.query.trim();
  const result = emptyResult(query, input.paperId);
  const maxResults = Math.min(Math.max(input.maxResults ?? 5, 1), 10);

  let catalog: PaperCatalog;
  try {
    catalog = loadPaperCatalog(workspaceRoot);
  } catch (error) {
    const outside = error instanceof Error && error.message.includes('outside');
    result.issues.push(
      issue(
        outside ? 'PATH_OUTSIDE_WORKSPACE' : 'WORKSPACE_CONFIG_INVALID',
        outside
          ? '科研工作区配置路径越过工作区边界。'
          : '科研工作区或论文目录配置无效。',
      ),
    );
    return result;
  }

  result.workspaceId = catalog.workspaceId;
  result.snapshotId = catalog.snapshotId;
  result.issues.push(...catalog.issues);

  const targetPaperId =
    input.paperId?.trim() || catalog.defaultPaperId || undefined;
  const selected = targetPaperId
    ? catalog.papers.filter((paper) => paper.paperId === targetPaperId)
    : catalog.papers;

  if (selected.length === 0) {
    result.issues.push(
      issue('PAPER_NOT_FOUND', '没有找到可检索的目标论文。', {
        paperId: targetPaperId,
      }),
    );
    return result;
  }

  const allMatches: ResearchPaperSearchMatch[] = [];
  for (const paper of selected) {
    result.searchedPaperIds.push(paper.paperId);
    try {
      const pages = extractPages(paper.realPath);
      result.searchedPages += pages.length;
      allMatches.push(...matchesForPaper(paper, pages, query));
    } catch (error) {
      const unavailable =
        error instanceof Error && error.message === 'extractor_unavailable';
      result.issues.push(
        issue(
          unavailable ? 'PDF_EXTRACTOR_UNAVAILABLE' : 'PDF_EXTRACTION_FAILED',
          unavailable
            ? '当前 Runner 未安装 pdftotext，无法检索 PDF 正文。'
            : '论文 PDF 文本提取失败。',
          { paperId: paper.paperId, path: paper.relativePath },
        ),
      );
    }
  }

  result.matches = allMatches
    .sort((a, b) => b.score - a.score || a.page - b.page)
    .slice(0, maxResults);

  const extractionFailed = result.issues.some(
    (item) =>
      item.code === 'PDF_EXTRACTOR_UNAVAILABLE' ||
      item.code === 'PDF_EXTRACTION_FAILED',
  );
  result.status = extractionFailed
    ? 'error'
    : result.matches.length > 0
      ? 'ok'
      : 'no_matches';
  return result;
}
