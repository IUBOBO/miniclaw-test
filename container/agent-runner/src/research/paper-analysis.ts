import fs from 'node:fs';
import path from 'node:path';

import { parse as parseYaml } from 'yaml';
import { z } from 'zod';

import {
  extractPdfPagesWithPdftotext,
  type PdfPageExtractor,
} from './paper-search.js';
import {
  fileSha256,
  isWithin,
  normalizeRelativePath,
  readLimitedWithinRoot,
  RESEARCH_WORKSPACE_FILE,
  resolveWithinRoot,
} from './workspace-loader.js';

const MAX_CONFIG_BYTES = 256 * 1024;
const MAX_METADATA_BYTES = 1024 * 1024;
const MAX_PDF_BYTES = 100 * 1024 * 1024;
const MAX_SECTION_EXCERPT_CHARS = 1000;

const workspaceSchema = z
  .object({
    schema_version: z.literal(2),
    root: z.literal('.'),
    workspace_id: z.string().min(1),
    snapshot_id: z.string().min(1),
    metadata_root: z.string().min(1),
    default_paper_id: z.string().min(1).optional(),
  })
  .passthrough();

const paperMetadataSchema = z
  .object({
    paper_id: z.string().min(1),
    title: z.string().min(1),
    local_file: z.string().min(1),
    sha256: z.string().regex(/^[a-f0-9]{64}$/i),
    text_extractable: z.boolean().optional().default(true),
    linked_source_files: z.array(z.string()).optional().default([]),
    linked_config_files: z.array(z.string()).optional().default([]),
    linked_log_files: z.array(z.string()).optional().default([]),
    linked_data_files: z.array(z.string()).optional().default([]),
    linked_weight_files: z.array(z.string()).optional().default([]),
  })
  .passthrough();

const resultMetadataSchema = z
  .object({ paper_id: z.string().min(1) })
  .passthrough();

export type ResearchPaperSectionKind =
  | 'abstract'
  | 'introduction'
  | 'related_work'
  | 'method'
  | 'experiments'
  | 'results'
  | 'limitations'
  | 'conclusion'
  | 'document_opening';

export type ResearchPaperAnalysisIssueCode =
  | 'WORKSPACE_CONFIG_INVALID'
  | 'INPUT_CONFLICT'
  | 'PAPER_NOT_FOUND'
  | 'PAPER_METADATA_INVALID'
  | 'PATH_OUTSIDE_WORKSPACE'
  | 'PAPER_FILE_MISSING'
  | 'PAPER_FILE_TOO_LARGE'
  | 'PAPER_HASH_MISMATCH'
  | 'PDF_EXTRACTOR_UNAVAILABLE'
  | 'PDF_EXTRACTION_FAILED'
  | 'OCR_REQUIRED'
  | 'SECTION_HEADINGS_NOT_FOUND';

export interface ResearchPaperAnalysisIssue {
  code: ResearchPaperAnalysisIssueCode;
  severity: 'warning' | 'error';
  message: string;
}

export interface ResearchPaperSectionEvidence {
  kind: ResearchPaperSectionKind;
  heading: string;
  page: number;
  excerpt: string;
  citation: string;
}

export interface ResearchPaperAnalysisResult {
  status: 'ok' | 'partial' | 'error';
  paper: {
    paperId: string | null;
    title: string;
    path: string;
    registration: 'registered' | 'provisional';
    pageCount: number;
  } | null;
  evidenceMode: 'paper_only' | 'paper_with_linked_materials' | null;
  sections: ResearchPaperSectionEvidence[];
  coverage: Record<
    Exclude<ResearchPaperSectionKind, 'document_opening'>,
    boolean
  >;
  linkedMaterials: {
    sourceCode: boolean;
    structuredResults: boolean;
    configurations: boolean;
    logs: boolean;
    data: boolean;
    weights: boolean;
  };
  capabilities: {
    summarizePaper: boolean;
    explainReportedMethod: boolean;
    analyzeReportedExperiments: boolean;
    verifySourceCode: boolean;
    verifyExecution: boolean;
    reproduceExperiment: boolean;
  };
  missingMaterial: Array<{ code: string; impact: string }>;
  analysisBoundary: string[];
  issues: ResearchPaperAnalysisIssue[];
  internalEvidence?: {
    workspaceId: string;
    snapshotId: string;
    sha256: string;
    metadataPath: string | null;
  };
}

interface LoadedPaperMetadata {
  metadataPath: string;
  value: z.infer<typeof paperMetadataSchema>;
}

interface ResolvedPaper {
  rootReal: string;
  workspaceId: string;
  snapshotId: string;
  metadataRoot: string;
  metadata: LoadedPaperMetadata | null;
  paperId: string | null;
  title: string;
  relativePath: string;
  realPath: string;
  sha256: string;
  registration: 'registered' | 'provisional';
}

class PaperAnalysisFailure extends Error {
  constructor(
    readonly code: ResearchPaperAnalysisIssueCode,
    message: string,
  ) {
    super(message);
  }
}

const sectionPatterns: Array<{
  kind: Exclude<ResearchPaperSectionKind, 'document_opening'>;
  pattern: RegExp;
}> = [
  {
    kind: 'abstract',
    pattern: /^(abstract|摘要)(?:\s*[.:：—-]\s*.*)?$/i,
  },
  {
    kind: 'introduction',
    pattern: /^(introduction|引言|绪论)\s*[:：]?\s*$/i,
  },
  {
    kind: 'related_work',
    pattern:
      /^(related work|background|literature review|相关工作|研究背景)\s*[:：]?\s*$/i,
  },
  {
    kind: 'method',
    pattern:
      /^(materials and methods|methods?|methodology|proposed method|approach|framework|network architecture|model architecture|proposed architecture|architecture|方法|研究方法|模型结构|网络结构)\s*[:：]?\s*$/i,
  },
  {
    kind: 'experiments',
    pattern:
      /^(experiments?|experimental setup|experimental results|evaluation|实验|实验设置|实验结果)\s*[:：]?\s*$/i,
  },
  {
    kind: 'results',
    pattern:
      /^(results?( and discussion)?|discussion|结果|结果与讨论|讨论)\s*[:：]?\s*$/i,
  },
  {
    kind: 'limitations',
    pattern:
      /^(limitations?|threats to validity|局限性|研究局限|不足与展望)\s*[:：]?\s*$/i,
  },
  {
    kind: 'conclusion',
    pattern:
      /^(conclusions?( and future work)?|summary|结论|总结|总结与展望)\s*[:：]?\s*$/i,
  },
];

function emptyCoverage(): ResearchPaperAnalysisResult['coverage'] {
  return {
    abstract: false,
    introduction: false,
    related_work: false,
    method: false,
    experiments: false,
    results: false,
    limitations: false,
    conclusion: false,
  };
}

function emptyLinkedMaterials(): ResearchPaperAnalysisResult['linkedMaterials'] {
  return {
    sourceCode: false,
    structuredResults: false,
    configurations: false,
    logs: false,
    data: false,
    weights: false,
  };
}

function emptyResult(): ResearchPaperAnalysisResult {
  return {
    status: 'error',
    paper: null,
    evidenceMode: null,
    sections: [],
    coverage: emptyCoverage(),
    linkedMaterials: emptyLinkedMaterials(),
    capabilities: {
      summarizePaper: false,
      explainReportedMethod: false,
      analyzeReportedExperiments: false,
      verifySourceCode: false,
      verifyExecution: false,
      reproduceExperiment: false,
    },
    missingMaterial: [],
    analysisBoundary: [],
    issues: [],
  };
}

function stripHeadingNumber(line: string): string {
  return line
    .trim()
    .replace(
      /^(?:(?:section|chapter)\s+)?(?:\d+(?:\.\d+)*|[IVXLC]+)[.):-]?\s+/i,
      '',
    )
    .trim();
}

function headingKind(
  line: string,
): Exclude<ResearchPaperSectionKind, 'document_opening'> | null {
  if (line.trim().length === 0 || line.trim().length > 140) return null;
  const normalized = stripHeadingNumber(line);
  return (
    sectionPatterns.find(({ pattern }) => pattern.test(normalized))?.kind ??
    null
  );
}

function buildSectionExcerpt(lines: string[], start: number): string {
  const selected: string[] = [];
  for (let index = start; index < lines.length; index += 1) {
    const line = lines[index].trim();
    if (index > start && headingKind(line)) break;
    if (line) selected.push(line);
    if (selected.join(' ').length >= MAX_SECTION_EXCERPT_CHARS) break;
  }
  const text = selected.join(' ').replace(/\s+/g, ' ').trim();
  return text.length <= MAX_SECTION_EXCERPT_CHARS
    ? text
    : text.slice(0, MAX_SECTION_EXCERPT_CHARS - 1).trimEnd() + '…';
}

function detectSections(
  pages: string[],
  maxSections: number,
): ResearchPaperSectionEvidence[] {
  const found = new Set<ResearchPaperSectionKind>();
  const sections: ResearchPaperSectionEvidence[] = [];
  for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
    const lines = pages[pageIndex].replaceAll('\r\n', '\n').split('\n');
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const kind = headingKind(lines[lineIndex]);
      if (!kind || found.has(kind)) continue;
      const excerpt = buildSectionExcerpt(lines, lineIndex);
      if (!excerpt) continue;
      found.add(kind);
      sections.push({
        kind,
        heading: lines[lineIndex].trim(),
        page: pageIndex + 1,
        excerpt,
        citation: `论文第 ${pageIndex + 1} 页`,
      });
      if (sections.length >= maxSections) return sections;
    }
  }

  if (sections.length === 0) {
    const opening = pages.slice(0, 2).join(' ').replace(/\s+/g, ' ').trim();
    if (opening) {
      sections.push({
        kind: 'document_opening',
        heading: '文档开头（未识别出标准章节标题）',
        page: 1,
        excerpt:
          opening.length <= MAX_SECTION_EXCERPT_CHARS
            ? opening
            : opening.slice(0, MAX_SECTION_EXCERPT_CHARS - 1).trimEnd() + '…',
        citation: '论文第 1 页',
      });
    }
  }
  return sections;
}

function loadPaperMetadata(
  rootReal: string,
  metadataRoot: string,
): LoadedPaperMetadata[] {
  const directory = resolveWithinRoot(rootReal, `${metadataRoot}/papers`);
  if (!directory || !fs.existsSync(directory.path)) return [];
  const directoryReal = fs.realpathSync(directory.path);
  if (!isWithin(rootReal, directoryReal)) {
    throw new PaperAnalysisFailure(
      'PATH_OUTSIDE_WORKSPACE',
      '论文元数据目录越过科研工作区边界。',
    );
  }

  const papers: LoadedPaperMetadata[] = [];
  for (const name of fs
    .readdirSync(directoryReal)
    .filter((item) => /\.ya?ml$/i.test(item))
    .sort()) {
    const metadataPath = `${metadataRoot}/papers/${name}`;
    const file = resolveWithinRoot(rootReal, metadataPath);
    if (!file) {
      throw new PaperAnalysisFailure(
        'PATH_OUTSIDE_WORKSPACE',
        '论文元数据文件越过科研工作区边界。',
      );
    }
    try {
      papers.push({
        metadataPath,
        value: paperMetadataSchema.parse(
          parseYaml(
            readLimitedWithinRoot(rootReal, file.path, MAX_METADATA_BYTES),
          ) as unknown,
        ),
      });
    } catch {
      throw new PaperAnalysisFailure(
        'PAPER_METADATA_INVALID',
        `论文元数据格式无效：${metadataPath}`,
      );
    }
  }
  return papers;
}

function resolvePaper(
  workspaceRoot: string,
  input: { paperId?: string; paperPath?: string },
): ResolvedPaper {
  if (input.paperId?.trim() && input.paperPath?.trim()) {
    throw new PaperAnalysisFailure(
      'INPUT_CONFLICT',
      'paper_id 与 paper_path 只能选择一个。',
    );
  }

  let rootReal: string;
  try {
    rootReal = fs.realpathSync(workspaceRoot);
    if (!fs.statSync(rootReal).isDirectory()) throw new Error('not_directory');
  } catch {
    throw new PaperAnalysisFailure(
      'WORKSPACE_CONFIG_INVALID',
      '科研工作区目录不存在或无法读取。',
    );
  }

  const configFile = resolveWithinRoot(rootReal, RESEARCH_WORKSPACE_FILE);
  if (!configFile) {
    throw new PaperAnalysisFailure(
      'PATH_OUTSIDE_WORKSPACE',
      '科研工作区入口越过工作区边界。',
    );
  }

  let workspace: z.infer<typeof workspaceSchema>;
  try {
    workspace = workspaceSchema.parse(
      JSON.parse(
        readLimitedWithinRoot(rootReal, configFile.path, MAX_CONFIG_BYTES),
      ) as unknown,
    );
  } catch {
    throw new PaperAnalysisFailure(
      'WORKSPACE_CONFIG_INVALID',
      '科研工作区入口格式无效。',
    );
  }

  const metadataEntries = loadPaperMetadata(rootReal, workspace.metadata_root);
  const requestedPaperId = input.paperId?.trim() || workspace.default_paper_id;
  let metadata: LoadedPaperMetadata | null = null;
  let requestedPath = input.paperPath?.trim();

  if (!requestedPath && requestedPaperId) {
    metadata =
      metadataEntries.find(
        (entry) => entry.value.paper_id === requestedPaperId,
      ) ?? null;
    if (!metadata) {
      throw new PaperAnalysisFailure(
        'PAPER_NOT_FOUND',
        `没有找到论文元数据：${requestedPaperId}`,
      );
    }
    requestedPath = metadata.value.local_file;
  } else if (!requestedPath && metadataEntries.length === 1) {
    metadata = metadataEntries[0];
    requestedPath = metadata.value.local_file;
  }

  if (!requestedPath) {
    throw new PaperAnalysisFailure(
      'PAPER_NOT_FOUND',
      '没有指定论文，且工作区无法确定唯一默认论文。',
    );
  }

  const normalizedPath = normalizeRelativePath(requestedPath);
  if (
    !normalizedPath ||
    path.extname(normalizedPath).toLowerCase() !== '.pdf'
  ) {
    throw new PaperAnalysisFailure(
      'PATH_OUTSIDE_WORKSPACE',
      '论文必须是科研工作区内的相对 PDF 路径。',
    );
  }
  const resolved = resolveWithinRoot(rootReal, normalizedPath);
  if (!resolved) {
    throw new PaperAnalysisFailure(
      'PATH_OUTSIDE_WORKSPACE',
      '论文路径越过科研工作区边界。',
    );
  }

  let realPath: string;
  let stat: fs.Stats;
  try {
    realPath = fs.realpathSync(resolved.path);
    if (!isWithin(rootReal, realPath)) throw new Error('outside');
    stat = fs.statSync(realPath);
    if (!stat.isFile()) throw new Error('not_file');
  } catch (error) {
    if (error instanceof Error && error.message === 'outside') {
      throw new PaperAnalysisFailure(
        'PATH_OUTSIDE_WORKSPACE',
        '论文通过符号链接越过科研工作区边界。',
      );
    }
    throw new PaperAnalysisFailure(
      'PAPER_FILE_MISSING',
      '论文文件不存在或无法读取。',
    );
  }
  if (stat.size > MAX_PDF_BYTES) {
    throw new PaperAnalysisFailure(
      'PAPER_FILE_TOO_LARGE',
      '论文文件超过 100 MB 的读取上限。',
    );
  }

  if (!metadata) {
    metadata =
      metadataEntries.find(
        (entry) =>
          normalizeRelativePath(entry.value.local_file) === normalizedPath,
      ) ?? null;
  }
  const sha256 = fileSha256(realPath);
  if (metadata && metadata.value.sha256.toLowerCase() !== sha256) {
    throw new PaperAnalysisFailure(
      'PAPER_HASH_MISMATCH',
      '论文文件与登记元数据的 SHA-256 不一致。',
    );
  }

  return {
    rootReal,
    workspaceId: workspace.workspace_id,
    snapshotId: workspace.snapshot_id,
    metadataRoot: workspace.metadata_root,
    metadata,
    paperId: metadata?.value.paper_id ?? null,
    title: metadata?.value.title ?? path.basename(normalizedPath, '.pdf'),
    relativePath: normalizedPath,
    realPath,
    sha256,
    registration: metadata ? 'registered' : 'provisional',
  };
}

function linkedFileExists(rootReal: string, candidates: string[]): boolean {
  return candidates.some((candidate) => {
    const normalized = normalizeRelativePath(candidate);
    if (!normalized) return false;
    const resolved = resolveWithinRoot(rootReal, normalized);
    if (!resolved || !fs.existsSync(resolved.path)) return false;
    try {
      const real = fs.realpathSync(resolved.path);
      return isWithin(rootReal, real) && fs.statSync(real).isFile();
    } catch {
      return false;
    }
  });
}

function hasStructuredResults(paper: ResolvedPaper): boolean {
  if (!paper.paperId) return false;
  const directory = resolveWithinRoot(
    paper.rootReal,
    `${paper.metadataRoot}/results`,
  );
  if (!directory || !fs.existsSync(directory.path)) return false;
  let real: string;
  try {
    real = fs.realpathSync(directory.path);
    if (!isWithin(paper.rootReal, real)) return false;
  } catch {
    return false;
  }
  return fs
    .readdirSync(real)
    .filter((name) => /\.ya?ml$/i.test(name))
    .some((name) => {
      const file = resolveWithinRoot(
        paper.rootReal,
        `${paper.metadataRoot}/results/${name}`,
      );
      if (!file) return false;
      try {
        const parsed = resultMetadataSchema.parse(
          parseYaml(
            readLimitedWithinRoot(
              paper.rootReal,
              file.path,
              MAX_METADATA_BYTES,
            ),
          ) as unknown,
        );
        return parsed.paper_id === paper.paperId;
      } catch {
        return false;
      }
    });
}

function buildMissingMaterial(
  linked: ResearchPaperAnalysisResult['linkedMaterials'],
): ResearchPaperAnalysisResult['missingMaterial'] {
  const items: ResearchPaperAnalysisResult['missingMaterial'] = [];
  if (!linked.sourceCode) {
    items.push({
      code: 'SOURCE_CODE_NOT_LINKED',
      impact: '可以解释论文方法，但无法核验源码是否按论文实现。',
    });
  }
  if (!linked.structuredResults) {
    items.push({
      code: 'STRUCTURED_RESULTS_NOT_LINKED',
      impact: '可以分析论文正文，精确表格数值需要后续结构化并人工核对。',
    });
  }
  if (!linked.configurations) {
    items.push({
      code: 'CONFIG_NOT_LINKED',
      impact: '无法核验模型参数、数据路径和训练条件。',
    });
  }
  if (!linked.logs) {
    items.push({
      code: 'LOG_NOT_LINKED',
      impact: '无法证明训练或评测实际执行。',
    });
  }
  if (!linked.data) {
    items.push({
      code: 'DATA_NOT_LINKED',
      impact: '无法核验数据内容和实际划分。',
    });
  }
  if (!linked.weights) {
    items.push({
      code: 'WEIGHTS_NOT_LINKED',
      impact: '无法加载论文模型或重复推理。',
    });
  }
  return items;
}

export function analyzeResearchPaper(
  workspaceRoot: string,
  input: {
    paperId?: string;
    paperPath?: string;
    maxSections?: number;
  },
  extractPages: PdfPageExtractor = extractPdfPagesWithPdftotext,
): ResearchPaperAnalysisResult {
  const result = emptyResult();
  let paper: ResolvedPaper;
  try {
    paper = resolvePaper(workspaceRoot, input);
  } catch (error) {
    if (error instanceof PaperAnalysisFailure) {
      result.issues.push({
        code: error.code,
        severity: 'error',
        message: error.message,
      });
    } else {
      result.issues.push({
        code: 'WORKSPACE_CONFIG_INVALID',
        severity: 'error',
        message: '科研工作区或论文信息无法读取。',
      });
    }
    return result;
  }

  result.paper = {
    paperId: paper.paperId,
    title: paper.title,
    path: paper.relativePath,
    registration: paper.registration,
    pageCount: 0,
  };
  result.internalEvidence = {
    workspaceId: paper.workspaceId,
    snapshotId: paper.snapshotId,
    sha256: paper.sha256,
    metadataPath: paper.metadata?.metadataPath ?? null,
  };

  const metadata = paper.metadata?.value;
  result.linkedMaterials = {
    sourceCode: linkedFileExists(
      paper.rootReal,
      metadata?.linked_source_files ?? [],
    ),
    structuredResults: hasStructuredResults(paper),
    configurations: linkedFileExists(
      paper.rootReal,
      metadata?.linked_config_files ?? [],
    ),
    logs: linkedFileExists(paper.rootReal, metadata?.linked_log_files ?? []),
    data: linkedFileExists(paper.rootReal, metadata?.linked_data_files ?? []),
    weights: linkedFileExists(
      paper.rootReal,
      metadata?.linked_weight_files ?? [],
    ),
  };
  result.evidenceMode = Object.values(result.linkedMaterials).some(Boolean)
    ? 'paper_with_linked_materials'
    : 'paper_only';

  let pages: string[];
  try {
    pages = extractPages(paper.realPath);
  } catch (error) {
    const unavailable =
      error instanceof Error && error.message === 'extractor_unavailable';
    result.issues.push({
      code: unavailable ? 'PDF_EXTRACTOR_UNAVAILABLE' : 'PDF_EXTRACTION_FAILED',
      severity: 'error',
      message: unavailable
        ? '当前 Runner 未安装 pdftotext，无法分析 PDF 正文。'
        : '论文 PDF 文本提取失败。',
    });
    result.missingMaterial = buildMissingMaterial(result.linkedMaterials);
    return result;
  }

  result.paper.pageCount = pages.length;
  const textLength = pages.reduce(
    (sum, pageText) => sum + pageText.trim().length,
    0,
  );
  if (textLength < 200 || metadata?.text_extractable === false) {
    result.status = 'partial';
    result.issues.push({
      code: 'OCR_REQUIRED',
      severity: 'warning',
      message: 'PDF 没有足够的可提取文字，需要 OCR 后才能可靠分析。',
    });
    result.missingMaterial = buildMissingMaterial(result.linkedMaterials);
    result.analysisBoundary = [
      '当前只能确认文件存在，不能从空白或扫描文字层推断论文内容。',
    ];
    return result;
  }

  const maxSections = Math.min(Math.max(input.maxSections ?? 8, 1), 12);
  result.sections = detectSections(pages, maxSections);
  for (const section of result.sections) {
    if (section.kind !== 'document_opening') {
      result.coverage[section.kind] = true;
    }
  }
  if (
    result.sections.length === 1 &&
    result.sections[0].kind === 'document_opening'
  ) {
    result.issues.push({
      code: 'SECTION_HEADINGS_NOT_FOUND',
      severity: 'warning',
      message: '已提取正文，但没有识别出标准章节标题。',
    });
  }

  result.capabilities = {
    summarizePaper: result.sections.length > 0,
    explainReportedMethod: result.coverage.method,
    analyzeReportedExperiments:
      result.coverage.experiments || result.coverage.results,
    verifySourceCode: result.linkedMaterials.sourceCode,
    verifyExecution: result.linkedMaterials.logs,
    reproduceExperiment:
      result.linkedMaterials.sourceCode &&
      result.linkedMaterials.configurations &&
      result.linkedMaterials.logs &&
      result.linkedMaterials.data &&
      result.linkedMaterials.weights,
  };
  result.missingMaterial = buildMissingMaterial(result.linkedMaterials);
  result.analysisBoundary = [
    '章节摘录只能证明论文写了什么，不代表论文结论已被独立复现。',
    result.linkedMaterials.structuredResults
      ? '精确指标应继续通过结构化结果工具读取。'
      : '当前没有经过核对的结构化结果，避免从复杂 PDF 表格直接拼接精确数值。',
    result.linkedMaterials.sourceCode
      ? '已发现论文关联源码，可继续使用源码定位工具核验实现。'
      : '当前没有论文关联源码，不能判断方法是否真实实现。',
  ];
  result.status = result.sections.some(
    (section) => section.kind !== 'document_opening',
  )
    ? 'ok'
    : 'partial';
  return result;
}

export function audienceResearchPaperAnalysis(
  payload: ResearchPaperAnalysisResult,
) {
  const { internalEvidence: _internalEvidence, ...audience } = payload;
  if (!audience.paper) return audience;

  const { registration: _registration, ...paper } = audience.paper;
  return { ...audience, paper };
}
