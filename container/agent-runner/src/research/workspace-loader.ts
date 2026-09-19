import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { parse as parseYaml } from 'yaml';
import { z } from 'zod';

import type {
  ResearchIssue,
  ResearchWorkspaceHealth,
  ResearchWorkspaceStatus,
} from './types.js';

export const RESEARCH_WORKSPACE_FILE = 'research-workspace.json';

const MAX_CONFIG_BYTES = 256 * 1024;
const MAX_MANIFEST_BYTES = 10 * 1024 * 1024;
const MAX_METADATA_BYTES = 1024 * 1024;
const MAX_ISSUES = 50;
const SHA256_PATTERN = /^[a-f0-9]{64}$/i;

const workspaceSchema = z
  .object({
    schema_version: z.literal(2),
    root: z.string().min(1),
    workspace_id: z.string().min(1),
    snapshot_id: z.string().min(1),
    manifest: z.string().min(1),
    metadata_root: z.string().min(1),
    models: z.record(z.string(), z.unknown()),
  })
  .passthrough();

const manifestEntrySchema = z
  .object({
    local_path: z.string().min(1),
    size: z.number().int().nonnegative(),
    sha256: z.string().regex(SHA256_PATTERN),
  })
  .passthrough();

const paperSchema = z.object({ paper_id: z.string().min(1) }).passthrough();

const predictionRunSchema = z
  .object({
    run_id: z.string().min(1),
    code_snapshot_id: z.string().min(1).optional(),
  })
  .passthrough();

type ManifestEntry = z.infer<typeof manifestEntrySchema>;

function emptyStatus(): ResearchWorkspaceStatus {
  return {
    status: 'error',
    workspaceId: null,
    snapshotId: null,
    manifest: {
      total: 0,
      present: 0,
      missing: 0,
      controlHashesChecked: 0,
      controlHashMismatches: 0,
    },
    models: [],
    papers: [],
    predictionRuns: [],
    issueCount: 0,
    issues: [],
  };
}

export function normalizeRelativePath(value: string): string | null {
  if (!value || value.includes('\0') || path.isAbsolute(value)) return null;
  const slashPath = value.replaceAll('\\', '/');
  const parts = slashPath.split('/');
  if (
    parts.some((part) => part === '..') ||
    /^[a-zA-Z]:/.test(slashPath) ||
    slashPath.startsWith('/')
  ) {
    return null;
  }
  const normalized = path.posix.normalize(slashPath);
  if (normalized === '..' || normalized.startsWith('../')) return null;
  return normalized.replace(/^\.\//, '');
}

export function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative === '' ||
    (!relative.startsWith('..' + path.sep) &&
      relative !== '..' &&
      !path.isAbsolute(relative))
  );
}

export function resolveWithinRoot(
  root: string,
  relativePath: string,
): { path: string; relativePath: string } | null {
  const normalized = normalizeRelativePath(relativePath);
  if (!normalized) return null;
  const candidate = path.resolve(root, ...normalized.split('/'));
  if (!isWithin(root, candidate)) return null;
  return { path: candidate, relativePath: normalized };
}

export class PathOutsideWorkspaceError extends Error {}

function readLimited(filePath: string, maxBytes: number): string {
  const stat = fs.statSync(filePath);
  if (!stat.isFile()) throw new Error('not_file');
  if (stat.size > maxBytes) throw new Error('too_large');
  return fs.readFileSync(filePath, 'utf8');
}

export function readLimitedWithinRoot(
  root: string,
  filePath: string,
  maxBytes: number,
): string {
  const realFile = fs.realpathSync(filePath);
  if (!isWithin(root, realFile)) throw new PathOutsideWorkspaceError();
  return readLimited(realFile, maxBytes);
}

export function fileSha256(filePath: string): string {
  return crypto
    .createHash('sha256')
    .update(fs.readFileSync(filePath))
    .digest('hex');
}

function shouldVerifyControlHash(localPath: string): boolean {
  return (
    localPath === RESEARCH_WORKSPACE_FILE ||
    localPath.startsWith('metadata/') ||
    localPath.startsWith('notes/methods/')
  );
}

function safeMessage(code: ResearchIssue['code']): string {
  switch (code) {
    case 'WORKSPACE_CONFIG_INVALID':
      return '科研工作区入口不存在、过大或格式无效。';
    case 'MANIFEST_INVALID':
      return '科研材料清单不存在、过大或格式无效。';
    case 'METADATA_INVALID':
      return '科研元数据不存在、过大或格式无效。';
    case 'READ_FAILED':
      return '科研工作区读取失败。';
    default:
      return '科研工作区校验失败。';
  }
}

export function hasResearchWorkspace(workspaceRoot: string): boolean {
  return fs.existsSync(path.join(workspaceRoot, RESEARCH_WORKSPACE_FILE));
}

export function loadResearchWorkspaceStatus(
  workspaceRoot: string,
): ResearchWorkspaceStatus {
  const result = emptyStatus();
  let issueCount = 0;
  let hasErrorIssue = false;

  const addIssue = (issue: ResearchIssue): void => {
    issueCount += 1;
    if (issue.severity === 'error') hasErrorIssue = true;
    if (result.issues.length < MAX_ISSUES) result.issues.push(issue);
  };

  let rootReal: string;
  try {
    rootReal = fs.realpathSync(workspaceRoot);
    if (!fs.statSync(rootReal).isDirectory()) throw new Error('not_directory');
  } catch {
    addIssue({
      code: 'READ_FAILED',
      severity: 'error',
      message: safeMessage('READ_FAILED'),
    });
    result.issueCount = issueCount;
    return result;
  }

  const entry = resolveWithinRoot(rootReal, RESEARCH_WORKSPACE_FILE);
  if (!entry) {
    addIssue({
      code: 'PATH_OUTSIDE_WORKSPACE',
      severity: 'error',
      message: '科研工作区入口路径越界。',
      path: RESEARCH_WORKSPACE_FILE,
    });
    result.issueCount = issueCount;
    return result;
  }

  let workspace: z.infer<typeof workspaceSchema>;
  try {
    const parsed = JSON.parse(
      readLimitedWithinRoot(rootReal, entry.path, MAX_CONFIG_BYTES),
    ) as unknown;
    const validated = workspaceSchema.safeParse(parsed);
    if (!validated.success) throw new Error('invalid_schema');
    workspace = validated.data;
    result.workspaceId = workspace.workspace_id;
    result.snapshotId = workspace.snapshot_id;
    result.models = Object.keys(workspace.models);
  } catch (error) {
    const outside = error instanceof PathOutsideWorkspaceError;
    addIssue({
      code: outside ? 'PATH_OUTSIDE_WORKSPACE' : 'WORKSPACE_CONFIG_INVALID',
      severity: 'error',
      message: outside
        ? '科研工作区入口通过符号链接越过工作区边界。'
        : safeMessage('WORKSPACE_CONFIG_INVALID'),
      path: RESEARCH_WORKSPACE_FILE,
    });
    result.issueCount = issueCount;
    return result;
  }

  if (workspace.root !== '.') {
    addIssue({
      code: 'RESEARCH_ROOT_INVALID',
      severity: 'error',
      message:
        'research-workspace.json 的 root 必须为相对于入口文件的“.”，不能使用宿主机绝对路径。',
      path: RESEARCH_WORKSPACE_FILE,
    });
    result.issueCount = issueCount;
    return result;
  }

  const manifestFile = resolveWithinRoot(rootReal, workspace.manifest);
  if (!manifestFile) {
    addIssue({
      code: 'PATH_OUTSIDE_WORKSPACE',
      severity: 'error',
      message: 'manifest 路径越过科研工作区边界。',
      path: workspace.manifest,
    });
    result.issueCount = issueCount;
    return result;
  }

  const manifestEntries: ManifestEntry[] = [];
  const manifestPaths = new Set<string>();
  try {
    const lines = readLimitedWithinRoot(
      rootReal,
      manifestFile.path,
      MAX_MANIFEST_BYTES,
    )
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0);
    for (let index = 0; index < lines.length; index += 1) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(lines[index]);
      } catch {
        addIssue({
          code: 'MANIFEST_INVALID',
          severity: 'error',
          message: 'manifest 第 ' + (index + 1) + ' 行不是有效 JSON。',
          path: manifestFile.relativePath,
        });
        continue;
      }
      const validated = manifestEntrySchema.safeParse(parsed);
      if (!validated.success) {
        addIssue({
          code: 'MANIFEST_INVALID',
          severity: 'error',
          message: 'manifest 第 ' + (index + 1) + ' 行字段无效。',
          path: manifestFile.relativePath,
        });
        continue;
      }
      const normalized = normalizeRelativePath(validated.data.local_path);
      if (!normalized) {
        addIssue({
          code: 'PATH_OUTSIDE_WORKSPACE',
          severity: 'error',
          message: 'manifest 包含绝对路径、父目录跳转或无效路径。',
          path: validated.data.local_path,
        });
        continue;
      }
      if (manifestPaths.has(normalized)) {
        addIssue({
          code: 'MANIFEST_DUPLICATE_PATH',
          severity: 'error',
          message: 'manifest 包含重复的相对路径。',
          path: normalized,
        });
        continue;
      }
      manifestPaths.add(normalized);
      manifestEntries.push({ ...validated.data, local_path: normalized });
    }
  } catch (error) {
    const outside = error instanceof PathOutsideWorkspaceError;
    addIssue({
      code: outside ? 'PATH_OUTSIDE_WORKSPACE' : 'MANIFEST_INVALID',
      severity: 'error',
      message: outside
        ? 'manifest 通过符号链接越过科研工作区边界。'
        : safeMessage('MANIFEST_INVALID'),
      path: manifestFile.relativePath,
    });
  }

  result.manifest.total = manifestEntries.length;

  for (const manifestEntry of manifestEntries) {
    const resolved = resolveWithinRoot(rootReal, manifestEntry.local_path);
    if (!resolved) {
      addIssue({
        code: 'PATH_OUTSIDE_WORKSPACE',
        severity: 'error',
        message: 'manifest 文件路径越过科研工作区边界。',
        path: manifestEntry.local_path,
      });
      continue;
    }

    let realFile: string;
    let stat: fs.Stats;
    try {
      realFile = fs.realpathSync(resolved.path);
      if (!isWithin(rootReal, realFile)) {
        addIssue({
          code: 'PATH_OUTSIDE_WORKSPACE',
          severity: 'error',
          message: 'manifest 文件通过符号链接越过科研工作区边界。',
          path: manifestEntry.local_path,
        });
        continue;
      }
      stat = fs.statSync(realFile);
      if (!stat.isFile()) throw new Error('not_file');
    } catch {
      result.manifest.missing += 1;
      addIssue({
        code: 'MANIFEST_FILE_MISSING',
        severity: 'warning',
        message: 'manifest 记录的文件在当前工作区中不存在。',
        path: manifestEntry.local_path,
      });
      continue;
    }

    result.manifest.present += 1;
    if (stat.size !== manifestEntry.size) {
      addIssue({
        code: 'MANIFEST_SIZE_MISMATCH',
        severity: 'warning',
        message: '当前文件大小与 manifest 记录不一致。',
        path: manifestEntry.local_path,
      });
    }

    if (shouldVerifyControlHash(manifestEntry.local_path)) {
      result.manifest.controlHashesChecked += 1;
      try {
        if (fileSha256(realFile) !== manifestEntry.sha256.toLowerCase()) {
          result.manifest.controlHashMismatches += 1;
          addIssue({
            code: 'CONTROL_HASH_MISMATCH',
            severity: 'warning',
            message: '科研控制文件的 SHA-256 与 manifest 不一致。',
            path: manifestEntry.local_path,
          });
        }
      } catch {
        addIssue({
          code: 'READ_FAILED',
          severity: 'warning',
          message: '科研控制文件无法读取。',
          path: manifestEntry.local_path,
        });
      }
    }
  }

  const readMetadataDirectory = (category: 'papers' | 'predictions'): void => {
    const relativeDirectory = workspace.metadata_root + '/' + category;
    const directory = resolveWithinRoot(rootReal, relativeDirectory);
    if (!directory) {
      addIssue({
        code: 'PATH_OUTSIDE_WORKSPACE',
        severity: 'error',
        message: '元数据目录越过科研工作区边界。',
        path: relativeDirectory,
      });
      return;
    }

    let files: string[];
    try {
      const realDirectory = fs.realpathSync(directory.path);
      if (!isWithin(rootReal, realDirectory)) {
        addIssue({
          code: 'PATH_OUTSIDE_WORKSPACE',
          severity: 'error',
          message: '元数据目录通过符号链接越过科研工作区边界。',
          path: relativeDirectory,
        });
        return;
      }
      files = fs
        .readdirSync(realDirectory)
        .filter((name) => /\.ya?ml$/i.test(name))
        .sort();
    } catch {
      addIssue({
        code: 'METADATA_INVALID',
        severity: 'warning',
        message: '科研元数据目录不存在或无法读取。',
        path: relativeDirectory,
      });
      return;
    }

    for (const name of files) {
      const relativeFile = relativeDirectory + '/' + name;
      const metadataFile = resolveWithinRoot(rootReal, relativeFile);
      if (!metadataFile) {
        addIssue({
          code: 'PATH_OUTSIDE_WORKSPACE',
          severity: 'error',
          message: '元数据文件路径越过科研工作区边界。',
          path: relativeFile,
        });
        continue;
      }
      try {
        const parsed = parseYaml(
          readLimitedWithinRoot(
            rootReal,
            metadataFile.path,
            MAX_METADATA_BYTES,
          ),
        ) as unknown;
        if (category === 'papers') {
          const paper = paperSchema.safeParse(parsed);
          if (!paper.success) throw new Error('invalid_paper');
          result.papers.push(paper.data.paper_id);
        } else {
          const run = predictionRunSchema.safeParse(parsed);
          if (!run.success) throw new Error('invalid_prediction');
          result.predictionRuns.push(run.data.run_id);
          if (
            run.data.code_snapshot_id &&
            run.data.code_snapshot_id !== workspace.snapshot_id
          ) {
            addIssue({
              code: 'SNAPSHOT_MISMATCH',
              severity: 'warning',
              message: 'Prediction Run 引用的代码快照与当前工作区不一致。',
              path: relativeFile,
            });
          }
        }
      } catch (error) {
        const outside = error instanceof PathOutsideWorkspaceError;
        addIssue({
          code: outside ? 'PATH_OUTSIDE_WORKSPACE' : 'METADATA_INVALID',
          severity: outside ? 'error' : 'warning',
          message: outside
            ? '元数据文件通过符号链接越过科研工作区边界。'
            : '科研元数据 YAML 格式或必填字段无效。',
          path: relativeFile,
        });
      }
    }
  };

  readMetadataDirectory('papers');
  readMetadataDirectory('predictions');

  result.models.sort();
  result.papers = [...new Set(result.papers)].sort();
  result.predictionRuns = [...new Set(result.predictionRuns)].sort();
  result.issueCount = issueCount;

  const health: ResearchWorkspaceHealth = hasErrorIssue
    ? 'error'
    : issueCount > 0
      ? 'degraded'
      : 'ready';
  result.status = health;
  return result;
}
