import fs from 'node:fs';

import { parse as parseYaml } from 'yaml';
import { z } from 'zod';

import {
  fileSha256,
  isWithin,
  normalizeRelativePath,
  readLimitedWithinRoot,
  RESEARCH_WORKSPACE_FILE,
  resolveWithinRoot,
} from './workspace-loader.js';

const MAX_CONFIG_BYTES = 256 * 1024;
const MAX_MANIFEST_BYTES = 10 * 1024 * 1024;
const MAX_METADATA_BYTES = 1024 * 1024;
const SHA256_PATTERN = /^[a-f0-9]{64}$/i;

const workspaceSchema = z
  .object({
    schema_version: z.literal(2),
    root: z.literal('.'),
    workspace_id: z.string().min(1),
    snapshot_id: z.string().min(1),
    manifest: z.string().min(1),
    metadata_root: z.string().min(1),
    default_paper_id: z.string().min(1).optional(),
  })
  .passthrough();

const manifestEntrySchema = z
  .object({
    local_path: z.string().min(1),
    sha256: z.string().regex(SHA256_PATTERN),
  })
  .passthrough();

const evidenceLocationSchema = z
  .object({
    pdf_pages: z.array(z.number().int().positive()).optional(),
    tables: z.array(z.number().int().positive()).optional(),
  })
  .passthrough();

const paperSchema = z
  .object({
    paper_id: z.string().min(1),
    local_file: z.string().min(1),
    sha256: z.string().regex(SHA256_PATTERN),
    evidence_locations: z.array(evidenceLocationSchema).optional().default([]),
  })
  .passthrough();

const comparisonRowSchema = z
  .object({
    scene: z.string().min(1),
    model: z.string().min(1),
    precision: z.number(),
    iou: z.number(),
    goc: z.number(),
    guc: z.number(),
    gtc: z.number(),
  })
  .passthrough();

const complexityRowSchema = z
  .object({
    model: z.string().min(1),
    parameters_m: z.number().nonnegative(),
    flops_g: z.number().nonnegative(),
  })
  .passthrough();

const datasetRowSchema = z
  .object({
    scene: z.string().min(1),
    sensor: z.string().min(1),
    train_original: z.number().int().nonnegative(),
    train_expanded: z.number().int().nonnegative(),
    test_original: z.number().int().nonnegative(),
    test_expanded: z.number().int().nonnegative(),
  })
  .passthrough();

const lossWeightRowSchema = z
  .object({
    mask_weight: z.number(),
    boundary_weight: z.number(),
    precision: z.number(),
    iou: z.number(),
    goc: z.number(),
    guc: z.number(),
    gtc: z.number(),
  })
  .passthrough();

const ablationRowSchema = z
  .object({
    variant: z.string().min(1),
    precision: z.number(),
    iou: z.number(),
    goc: z.number(),
    guc: z.number(),
    gtc: z.number(),
  })
  .passthrough();

const resultSetSchema = z
  .object({
    result_set_id: z.string().min(1),
    paper_id: z.string().min(1),
    authority: z.string().min(1),
    source_file: z.string().min(1),
    source_sha256: z.string().regex(SHA256_PATTERN),
    dataset_counts_table_1: z.array(datasetRowSchema).optional().default([]),
    task_loss_weights_table_2: z
      .array(lossWeightRowSchema)
      .optional()
      .default([]),
    comparison_table_3: z.array(comparisonRowSchema).optional().default([]),
    complexity_table_4: z.array(complexityRowSchema).optional().default([]),
    ablation_table_7: z.array(ablationRowSchema).optional().default([]),
  })
  .passthrough();

export type ResearchResultTable =
  | 'all'
  | 'dataset_counts'
  | 'loss_weights'
  | 'comparison'
  | 'complexity'
  | 'ablation';

export interface ResearchResultFact {
  factType: Exclude<ResearchResultTable, 'all'>;
  rowKey: string;
  factText: string;
  values: Record<string, string | number>;
  citation: {
    paperId: string;
    table: number;
    pdfPage: number | null;
    label: string;
  };
}

export interface ResearchResultLookupResult {
  status: 'ok' | 'no_matches' | 'error';
  paperId: string | null;
  authority: string | null;
  filters: {
    table: ResearchResultTable;
    model: string | null;
    scene: string | null;
    variant: string | null;
  };
  facts: ResearchResultFact[];
  issues: Array<{ code: string; message: string }>;
  presentation: {
    defaultMode: 'natural_language';
    citationStyle: 'paper_table_page';
    hideInternalEvidenceByDefault: true;
  };
  internalEvidence: {
    workspaceId: string | null;
    snapshotId: string | null;
    resultSetId: string | null;
    resultPath: string | null;
    resultSha256: string | null;
    sourcePaperPath: string | null;
    sourcePaperSha256: string | null;
  };
}

interface LoadedResultSet {
  workspaceId: string;
  snapshotId: string;
  resultPath: string;
  resultSha256: string;
  paper: z.infer<typeof paperSchema>;
  resultSet: z.infer<typeof resultSetSchema>;
  tablePages: Map<number, number>;
}

function emptyResult(input: {
  paperId?: string;
  table?: ResearchResultTable;
  model?: string;
  scene?: string;
  variant?: string;
}): ResearchResultLookupResult {
  return {
    status: 'error',
    paperId: input.paperId ?? null,
    authority: null,
    filters: {
      table: input.table ?? 'all',
      model: input.model ?? null,
      scene: input.scene ?? null,
      variant: input.variant ?? null,
    },
    facts: [],
    issues: [],
    presentation: {
      defaultMode: 'natural_language',
      citationStyle: 'paper_table_page',
      hideInternalEvidenceByDefault: true,
    },
    internalEvidence: {
      workspaceId: null,
      snapshotId: null,
      resultSetId: null,
      resultPath: null,
      resultSha256: null,
      sourcePaperPath: null,
      sourcePaperSha256: null,
    },
  };
}

function same(value: string, filter?: string): boolean {
  return (
    !filter ||
    value.localeCompare(filter, undefined, { sensitivity: 'accent' }) === 0
  );
}

function loadResultSet(
  workspaceRoot: string,
  paperId?: string,
): LoadedResultSet {
  const rootReal = fs.realpathSync(workspaceRoot);
  if (!fs.statSync(rootReal).isDirectory())
    throw new Error('workspace_invalid');
  const configFile = resolveWithinRoot(rootReal, RESEARCH_WORKSPACE_FILE);
  if (!configFile) throw new Error('path_outside');
  const workspace = workspaceSchema.parse(
    JSON.parse(
      readLimitedWithinRoot(rootReal, configFile.path, MAX_CONFIG_BYTES),
    ) as unknown,
  );
  const requestedPaperId = paperId || workspace.default_paper_id;

  const manifestFile = resolveWithinRoot(rootReal, workspace.manifest);
  if (!manifestFile) throw new Error('path_outside');
  const manifest = new Map<string, string>();
  for (const line of readLimitedWithinRoot(
    rootReal,
    manifestFile.path,
    MAX_MANIFEST_BYTES,
  )
    .split(/\r?\n/)
    .filter(Boolean)) {
    const entry = manifestEntrySchema.parse(JSON.parse(line) as unknown);
    const normalized = normalizeRelativePath(entry.local_path);
    if (!normalized) throw new Error('path_outside');
    if (manifest.has(normalized)) throw new Error('manifest_invalid');
    manifest.set(normalized, entry.sha256.toLowerCase());
  }

  const paperDir = resolveWithinRoot(
    rootReal,
    `${workspace.metadata_root}/papers`,
  );
  if (!paperDir) throw new Error('path_outside');
  const paperDirReal = fs.realpathSync(paperDir.path);
  if (!isWithin(rootReal, paperDirReal)) throw new Error('path_outside');
  let paper: z.infer<typeof paperSchema> | undefined;
  for (const name of fs
    .readdirSync(paperDirReal)
    .filter((item) => /\.ya?ml$/i.test(item))) {
    const relative = `${workspace.metadata_root}/papers/${name}`;
    const file = resolveWithinRoot(rootReal, relative);
    if (!file) throw new Error('path_outside');
    const candidate = paperSchema.parse(
      parseYaml(
        readLimitedWithinRoot(rootReal, file.path, MAX_METADATA_BYTES),
      ) as unknown,
    );
    if (requestedPaperId && candidate.paper_id === requestedPaperId)
      paper = candidate;
    if (!requestedPaperId) {
      if (paper) throw new Error('paper_ambiguous');
      paper = candidate;
    }
  }
  if (!paper) throw new Error('paper_not_found');

  const resultDir = resolveWithinRoot(
    rootReal,
    `${workspace.metadata_root}/results`,
  );
  if (!resultDir) throw new Error('path_outside');
  const resultDirReal = fs.realpathSync(resultDir.path);
  if (!isWithin(rootReal, resultDirReal)) throw new Error('path_outside');
  let resultPath: string | undefined;
  let resultSet: z.infer<typeof resultSetSchema> | undefined;
  for (const name of fs
    .readdirSync(resultDirReal)
    .filter((item) => /\.ya?ml$/i.test(item))) {
    const relative = `${workspace.metadata_root}/results/${name}`;
    const file = resolveWithinRoot(rootReal, relative);
    if (!file) throw new Error('path_outside');
    const candidate = resultSetSchema.parse(
      parseYaml(
        readLimitedWithinRoot(rootReal, file.path, MAX_METADATA_BYTES),
      ) as unknown,
    );
    if (candidate.paper_id === paper.paper_id) {
      resultPath = relative;
      resultSet = candidate;
      break;
    }
  }
  if (!resultPath || !resultSet) throw new Error('result_not_found');

  const expectedResultHash = manifest.get(resultPath);
  const resultFile = resolveWithinRoot(rootReal, resultPath);
  if (
    !expectedResultHash ||
    !resultFile ||
    fileSha256(resultFile.path) !== expectedResultHash
  ) {
    throw new Error('result_hash_mismatch');
  }
  const sourcePath = normalizeRelativePath(resultSet.source_file);
  if (!sourcePath || sourcePath !== normalizeRelativePath(paper.local_file)) {
    throw new Error('source_mismatch');
  }
  const sourceFile = resolveWithinRoot(rootReal, sourcePath);
  const sourceManifestHash = manifest.get(sourcePath);
  if (
    !sourceFile ||
    !sourceManifestHash ||
    sourceManifestHash !== resultSet.source_sha256.toLowerCase() ||
    sourceManifestHash !== paper.sha256.toLowerCase() ||
    fileSha256(sourceFile.path) !== sourceManifestHash
  ) {
    throw new Error('source_hash_mismatch');
  }

  const tablePages = new Map<number, number>();
  for (const location of paper.evidence_locations) {
    const page = location.pdf_pages?.[0];
    if (!page) continue;
    for (const table of location.tables ?? []) tablePages.set(table, page);
  }
  return {
    workspaceId: workspace.workspace_id,
    snapshotId: workspace.snapshot_id,
    resultPath,
    resultSha256: expectedResultHash,
    paper,
    resultSet,
    tablePages,
  };
}

function citation(paperId: string, table: number, page: number | undefined) {
  return {
    paperId,
    table,
    pdfPage: page ?? null,
    label: page ? `论文表 ${table}（PDF 第 ${page} 页）` : `论文表 ${table}`,
  };
}

export function lookupResearchResults(
  workspaceRoot: string,
  input: {
    paperId?: string;
    table?: ResearchResultTable;
    model?: string;
    scene?: string;
    variant?: string;
    maxResults?: number;
  },
): ResearchResultLookupResult {
  const normalized = {
    paperId: input.paperId?.trim(),
    table: input.table ?? 'all',
    model: input.model?.trim(),
    scene: input.scene?.trim(),
    variant: input.variant?.trim(),
  };
  const result = emptyResult(normalized);
  const maxResults = Math.min(Math.max(input.maxResults ?? 20, 1), 50);
  let loaded: LoadedResultSet;
  try {
    loaded = loadResultSet(workspaceRoot, normalized.paperId);
  } catch (error) {
    const code = error instanceof Error ? error.message : 'read_failed';
    result.issues.push({
      code: code.toUpperCase(),
      message: '结构化论文结果不存在、格式无效、来源变化或路径越界。',
    });
    return result;
  }

  const { resultSet, paper, tablePages } = loaded;
  result.paperId = paper.paper_id;
  result.authority = resultSet.authority;
  result.internalEvidence = {
    workspaceId: loaded.workspaceId,
    snapshotId: loaded.snapshotId,
    resultSetId: resultSet.result_set_id,
    resultPath: loaded.resultPath,
    resultSha256: loaded.resultSha256,
    sourcePaperPath: resultSet.source_file,
    sourcePaperSha256: resultSet.source_sha256.toLowerCase(),
  };
  const include = (table: ResearchResultTable) =>
    normalized.table === 'all' || normalized.table === table;
  const facts: ResearchResultFact[] = [];

  if (include('dataset_counts')) {
    for (const row of resultSet.dataset_counts_table_1.filter((item) =>
      same(item.scene, normalized.scene),
    )) {
      facts.push({
        factType: 'dataset_counts',
        rowKey: `${paper.paper_id}:T1:${row.scene}`,
        factText: `${row.scene} 数据集使用 ${row.sensor}；原始训练/测试样本分别为 ${row.train_original}/${row.test_original}，扩增后分别为 ${row.train_expanded}/${row.test_expanded}。`,
        values: {
          scene: row.scene,
          sensor: row.sensor,
          train_original: row.train_original,
          train_expanded: row.train_expanded,
          test_original: row.test_original,
          test_expanded: row.test_expanded,
        },
        citation: citation(paper.paper_id, 1, tablePages.get(1)),
      });
    }
  }
  if (include('loss_weights') && !normalized.model && !normalized.variant) {
    for (const row of resultSet.task_loss_weights_table_2) {
      facts.push({
        factType: 'loss_weights',
        rowKey: `${paper.paper_id}:T2:${row.mask_weight}:${row.boundary_weight}`,
        factText: `掩码/边界损失权重为 ${row.mask_weight}/${row.boundary_weight} 时，Precision=${row.precision}、IoU=${row.iou}、GOC=${row.goc}、GUC=${row.guc}、GTC=${row.gtc}。`,
        values: {
          mask_weight: row.mask_weight,
          boundary_weight: row.boundary_weight,
          precision: row.precision,
          iou: row.iou,
          goc: row.goc,
          guc: row.guc,
          gtc: row.gtc,
        },
        citation: citation(paper.paper_id, 2, tablePages.get(2)),
      });
    }
  }
  if (include('comparison')) {
    for (const row of resultSet.comparison_table_3.filter(
      (item) =>
        same(item.model, normalized.model) &&
        same(item.scene, normalized.scene),
    )) {
      facts.push({
        factType: 'comparison',
        rowKey: `${paper.paper_id}:T3:${row.scene}:${row.model}`,
        factText: `${row.model} 在 ${row.scene} 场景的 Precision=${row.precision}、IoU=${row.iou}、GOC=${row.goc}、GUC=${row.guc}、GTC=${row.gtc}。`,
        values: {
          scene: row.scene,
          model: row.model,
          precision: row.precision,
          iou: row.iou,
          goc: row.goc,
          guc: row.guc,
          gtc: row.gtc,
        },
        citation: citation(paper.paper_id, 3, tablePages.get(3)),
      });
    }
  }
  if (include('complexity')) {
    for (const row of resultSet.complexity_table_4.filter((item) =>
      same(item.model, normalized.model),
    )) {
      facts.push({
        factType: 'complexity',
        rowKey: `${paper.paper_id}:T4:${row.model}`,
        factText: `${row.model} 的参数量为 ${row.parameters_m}M，FLOPs 为 ${row.flops_g}G。`,
        values: {
          model: row.model,
          parameters_m: row.parameters_m,
          flops_g: row.flops_g,
        },
        citation: citation(paper.paper_id, 4, tablePages.get(4)),
      });
    }
  }
  if (include('ablation')) {
    for (const row of resultSet.ablation_table_7.filter((item) =>
      same(item.variant, normalized.variant || normalized.model),
    )) {
      facts.push({
        factType: 'ablation',
        rowKey: `${paper.paper_id}:T7:${row.variant}`,
        factText: `${row.variant} 消融结果为 Precision=${row.precision}、IoU=${row.iou}、GOC=${row.goc}、GUC=${row.guc}、GTC=${row.gtc}。`,
        values: {
          variant: row.variant,
          precision: row.precision,
          iou: row.iou,
          goc: row.goc,
          guc: row.guc,
          gtc: row.gtc,
        },
        citation: citation(paper.paper_id, 7, tablePages.get(7)),
      });
    }
  }

  result.facts = facts.slice(0, maxResults);
  result.status = result.facts.length > 0 ? 'ok' : 'no_matches';
  return result;
}
