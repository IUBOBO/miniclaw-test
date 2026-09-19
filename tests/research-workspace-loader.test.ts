import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, test } from 'vitest';

import { createMcpTools } from '../container/agent-runner/src/mcp-tools.js';
import { lookupResearchCode } from '../container/agent-runner/src/research/code-lookup.js';
import { searchResearchPapers } from '../container/agent-runner/src/research/paper-search.js';
import { lookupResearchResults } from '../container/agent-runner/src/research/result-lookup.js';
import { createResearchWorkspaceTools } from '../container/agent-runner/src/research/research-tools.js';
import { loadResearchWorkspaceStatus } from '../container/agent-runner/src/research/workspace-loader.js';

const roots: string[] = [];

function sha256(filePath: string): string {
  return crypto
    .createHash('sha256')
    .update(fs.readFileSync(filePath))
    .digest('hex');
}

function writeFile(root: string, relativePath: string, content: string): void {
  const filePath = path.join(root, ...relativePath.split('/'));
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content, 'utf8');
}

function manifestRow(
  root: string,
  relativePath: string,
): Record<string, unknown> {
  const filePath = path.join(root, ...relativePath.split('/'));
  const stat = fs.statSync(filePath);
  return {
    local_path: relativePath,
    model: 'workspace',
    kind: 'test',
    origin: 'test',
    remote_path: null,
    derived_from: [],
    size: stat.size,
    modified_at: stat.mtime.toISOString(),
    sha256: sha256(filePath),
  };
}

function appendManifestRow(root: string, row: Record<string, unknown>): void {
  fs.appendFileSync(
    path.join(root, 'snapshot-manifest.jsonl'),
    JSON.stringify(row) + '\n',
    'utf8',
  );
}

function createFixture(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'miniclaw-research-'));
  roots.push(root);

  writeFile(
    root,
    'research-workspace.json',
    JSON.stringify(
      {
        schema_version: 2,
        root: '.',
        workspace_id: 'bdgi-research',
        snapshot_id: 'SNAP-TEST',
        manifest: 'snapshot-manifest.jsonl',
        metadata_root: 'metadata',
        models: {
          'BDGI-Net': {
            role: 'proposed_model',
            directory: 'mymodel',
            code: 'mymodel/code',
          },
        },
      },
      null,
      2,
    ),
  );
  writeFile(
    root,
    'metadata/papers/PAPER-BDGI.yaml',
    'schema_version: 1\npaper_id: PAPER-BDGI\nstatus: under_review\n',
  );
  writeFile(
    root,
    'metadata/predictions/RUN-BDGI-SD-SAVED.yaml',
    'schema_version: 1\nrun_id: RUN-BDGI-SD-SAVED\ncode_snapshot_id: SNAP-TEST\n',
  );
  writeFile(root, 'notes/methods/BDGI-Net.md', '# BDGI-Net\n');
  writeFile(root, 'samples/example.txt', 'sample');

  const paths = [
    'research-workspace.json',
    'metadata/papers/PAPER-BDGI.yaml',
    'metadata/predictions/RUN-BDGI-SD-SAVED.yaml',
    'notes/methods/BDGI-Net.md',
    'samples/example.txt',
  ];
  fs.writeFileSync(
    path.join(root, 'snapshot-manifest.jsonl'),
    paths.map((item) => JSON.stringify(manifestRow(root, item))).join('\n') +
      '\n',
    'utf8',
  );
  return root;
}

function createPaperSearchFixture(): string {
  const root = createFixture();
  writeFile(root, 'PDFpaper/PAPER-BDGI.pdf', 'fixture-pdf');
  writeFile(
    root,
    'metadata/papers/PAPER-BDGI.yaml',
    [
      'schema_version: 1',
      'paper_id: PAPER-BDGI',
      'title: Fixture BDGI Paper',
      'local_file: PDFpaper/PAPER-BDGI.pdf',
      'sha256: ' + sha256(path.join(root, 'PDFpaper', 'PAPER-BDGI.pdf')),
      'text_extractable: true',
      '',
    ].join('\n'),
  );

  const paths = [
    'research-workspace.json',
    'metadata/papers/PAPER-BDGI.yaml',
    'metadata/predictions/RUN-BDGI-SD-SAVED.yaml',
    'notes/methods/BDGI-Net.md',
    'samples/example.txt',
    'PDFpaper/PAPER-BDGI.pdf',
  ];
  fs.writeFileSync(
    path.join(root, 'snapshot-manifest.jsonl'),
    paths.map((item) => JSON.stringify(manifestRow(root, item))).join('\n') +
      '\n',
    'utf8',
  );
  return root;
}

function createCodeSearchFixture(): string {
  const root = createFixture();
  writeFile(
    root,
    'mymodel/code/model.py',
    [
      'import torch',
      '',
      'class MaskGuidedByBoundaryModule:',
      '    """Guided Mask Refinement Module (GMRM)."""',
      '    def forward(self, mask, boundary):',
      '        return mask * boundary',
      '',
      'class Tripmodel:',
      '    def __init__(self):',
      '        self.mask_fuse = MaskGuidedByBoundaryModule()',
      '',
    ].join('\n'),
  );

  const paths = [
    'research-workspace.json',
    'metadata/papers/PAPER-BDGI.yaml',
    'metadata/predictions/RUN-BDGI-SD-SAVED.yaml',
    'notes/methods/BDGI-Net.md',
    'samples/example.txt',
  ];
  const sourceRow = {
    ...manifestRow(root, 'mymodel/code/model.py'),
    model: 'BDGI-Net',
    kind: 'source_code',
  };
  fs.writeFileSync(
    path.join(root, 'snapshot-manifest.jsonl'),
    paths.map((item) => JSON.stringify(manifestRow(root, item))).join('\n') +
      '\n' +
      JSON.stringify(sourceRow) +
      '\n',
    'utf8',
  );
  return root;
}

function createResultLookupFixture(): string {
  const root = createFixture();
  writeFile(root, 'PDFpaper/BDGI-Net.pdf', 'paper-pdf');
  writeFile(
    root,
    'metadata/papers/PAPER-BDGI.yaml',
    [
      'paper_id: PAPER-BDGI',
      'local_file: PDFpaper/BDGI-Net.pdf',
      'sha256: ' + sha256(path.join(root, 'PDFpaper', 'BDGI-Net.pdf')),
      'evidence_locations:',
      '  - topic: comparison',
      '    pdf_pages: [8]',
      '    tables: [3]',
      '  - topic: complexity',
      '    pdf_pages: [9]',
      '    tables: [4]',
      '',
    ].join('\n'),
  );
  writeFile(
    root,
    'metadata/results/PAPER-BDGI-RESULTS.yaml',
    [
      'result_set_id: PAPER-BDGI-RESULTS',
      'paper_id: PAPER-BDGI',
      'authority: paper_final',
      'source_file: PDFpaper/BDGI-Net.pdf',
      'source_sha256: ' + sha256(path.join(root, 'PDFpaper', 'BDGI-Net.pdf')),
      'comparison_table_3:',
      '  - scene: Shandong',
      '    model: HBGNet',
      '    precision: 0.9220',
      '    iou: 0.8288',
      '    goc: 0.1295',
      '    guc: 0.2692',
      '    gtc: 0.2723',
      'complexity_table_4:',
      '  - model: HBGNet',
      '    parameters_m: 30.45',
      '    flops_g: 463.81',
      '  - model: BDGI-Net',
      '    parameters_m: 20.09',
      '    flops_g: 321.98',
      '  - model: REAUNet',
      '    parameters_m: 142.40',
      '    flops_g: 2035.05',
      '',
    ].join('\n'),
  );

  const paths = [
    'research-workspace.json',
    'metadata/papers/PAPER-BDGI.yaml',
    'metadata/predictions/RUN-BDGI-SD-SAVED.yaml',
    'metadata/results/PAPER-BDGI-RESULTS.yaml',
    'notes/methods/BDGI-Net.md',
    'samples/example.txt',
    'PDFpaper/BDGI-Net.pdf',
  ];
  fs.writeFileSync(
    path.join(root, 'snapshot-manifest.jsonl'),
    paths.map((item) => JSON.stringify(manifestRow(root, item))).join('\n') +
      '\n',
    'utf8',
  );
  return root;
}

afterEach(() => {
  while (roots.length > 0) {
    fs.rmSync(roots.pop() as string, { recursive: true, force: true });
  }
});

describe('research workspace loader', () => {
  test('loads a complete workspace as ready', () => {
    const status = loadResearchWorkspaceStatus(createFixture());

    expect(status).toMatchObject({
      status: 'ready',
      workspaceId: 'bdgi-research',
      snapshotId: 'SNAP-TEST',
      manifest: {
        total: 5,
        present: 5,
        missing: 0,
        controlHashesChecked: 4,
        controlHashMismatches: 0,
      },
      models: ['BDGI-Net'],
      papers: ['PAPER-BDGI'],
      predictionRuns: ['RUN-BDGI-SD-SAVED'],
      issueCount: 0,
      issues: [],
    });
  });

  test('rejects a stale absolute root', () => {
    const root = createFixture();
    const configPath = path.join(root, 'research-workspace.json');
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<
      string,
      unknown
    >;
    config.root = 'E:\\Agent\\miniclaw-data';
    fs.writeFileSync(configPath, JSON.stringify(config), 'utf8');

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('error');
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'RESEARCH_ROOT_INVALID' }),
      ]),
    );
  });

  test('reports duplicate manifest paths as an error', () => {
    const root = createFixture();
    appendManifestRow(root, manifestRow(root, 'samples/example.txt'));

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('error');
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'MANIFEST_DUPLICATE_PATH' }),
      ]),
    );
  });

  test('reports missing files as degraded', () => {
    const root = createFixture();
    fs.unlinkSync(path.join(root, 'samples', 'example.txt'));

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('degraded');
    expect(status.manifest.missing).toBe(1);
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: 'MANIFEST_FILE_MISSING',
          path: 'samples/example.txt',
        }),
      ]),
    );
  });

  test('reports changed control files without reading image payloads', () => {
    const root = createFixture();
    fs.appendFileSync(
      path.join(root, 'metadata', 'papers', 'PAPER-BDGI.yaml'),
      'title: changed\n',
      'utf8',
    );

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('degraded');
    expect(status.manifest.controlHashMismatches).toBe(1);
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: 'CONTROL_HASH_MISMATCH',
          path: 'metadata/papers/PAPER-BDGI.yaml',
        }),
      ]),
    );
  });

  test('rejects traversal paths and malformed JSONL', () => {
    const traversalRoot = createFixture();
    appendManifestRow(traversalRoot, {
      local_path: '../secret.txt',
      size: 0,
      sha256: '0'.repeat(64),
    });
    expect(loadResearchWorkspaceStatus(traversalRoot)).toMatchObject({
      status: 'error',
      issues: expect.arrayContaining([
        expect.objectContaining({ code: 'PATH_OUTSIDE_WORKSPACE' }),
      ]),
    });

    const malformedRoot = createFixture();
    fs.appendFileSync(
      path.join(malformedRoot, 'snapshot-manifest.jsonl'),
      'not-json\n',
      'utf8',
    );
    expect(loadResearchWorkspaceStatus(malformedRoot)).toMatchObject({
      status: 'error',
      issues: expect.arrayContaining([
        expect.objectContaining({ code: 'MANIFEST_INVALID' }),
      ]),
    });
  });

  test('rejects a symlink that leaves the workspace', () => {
    const root = createFixture();
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'miniclaw-outside-'));
    roots.push(outside);
    writeFile(outside, 'secret.txt', 'outside');
    const linkPath = path.join(root, 'linked');
    fs.symlinkSync(
      outside,
      linkPath,
      process.platform === 'win32' ? 'junction' : 'dir',
    );
    appendManifestRow(root, {
      local_path: 'linked/secret.txt',
      size: fs.statSync(path.join(outside, 'secret.txt')).size,
      sha256: sha256(path.join(outside, 'secret.txt')),
    });

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('error');
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: 'PATH_OUTSIDE_WORKSPACE',
          path: 'linked/secret.txt',
        }),
      ]),
    );
  });

  test('rejects a manifest reached through an escaping symlink', () => {
    const root = createFixture();
    const outside = fs.mkdtempSync(
      path.join(os.tmpdir(), 'miniclaw-manifest-'),
    );
    roots.push(outside);
    fs.copyFileSync(
      path.join(root, 'snapshot-manifest.jsonl'),
      path.join(outside, 'snapshot-manifest.jsonl'),
    );
    fs.symlinkSync(
      outside,
      path.join(root, 'manifest-link'),
      process.platform === 'win32' ? 'junction' : 'dir',
    );

    const configPath = path.join(root, 'research-workspace.json');
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<
      string,
      unknown
    >;
    config.manifest = 'manifest-link/snapshot-manifest.jsonl';
    fs.writeFileSync(configPath, JSON.stringify(config), 'utf8');

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('error');
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: 'PATH_OUTSIDE_WORKSPACE',
          path: 'manifest-link/snapshot-manifest.jsonl',
        }),
      ]),
    );
  });

  test('reports malformed YAML without crashing the runner', () => {
    const root = createFixture();
    fs.writeFileSync(
      path.join(root, 'metadata', 'papers', 'PAPER-BDGI.yaml'),
      ': : invalid yaml',
      'utf8',
    );

    const status = loadResearchWorkspaceStatus(root);
    expect(status.status).toBe('degraded');
    expect(status.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'METADATA_INVALID' }),
      ]),
    );
  });
});

describe('research paper search', () => {
  test('returns bounded page-level evidence with relative sources', () => {
    const root = createPaperSearchFixture();
    const result = searchResearchPapers(
      root,
      { query: 'GMRM guided refinement', maxResults: 3 },
      () => [
        'Introduction without the requested module.',
        'The Guided Mask Refinement Module (GMRM) transfers boundary guidance into semantic refinement.',
        'Experiments and conclusions.',
      ],
    );

    expect(result).toMatchObject({
      status: 'ok',
      workspaceId: 'bdgi-research',
      snapshotId: 'SNAP-TEST',
      searchedPaperIds: ['PAPER-BDGI'],
      searchedPages: 3,
      issues: [],
    });
    expect(result.matches).toHaveLength(1);
    expect(result.matches[0]).toMatchObject({
      paperId: 'PAPER-BDGI',
      page: 2,
      matchedTerms: expect.arrayContaining(['gmrm', 'guided', 'refinement']),
      source: {
        path: 'PDFpaper/PAPER-BDGI.pdf',
        location: 'page 2',
      },
    });
    expect(result.matches[0].source).not.toHaveProperty('absolutePath');
    expect(result.matches[0].excerpt.length).toBeLessThanOrEqual(700);
  });

  test('returns no_matches instead of inventing paper evidence', () => {
    const result = searchResearchPapers(
      createPaperSearchFixture(),
      { query: 'quantum-entanglement' },
      () => ['Remote sensing parcel segmentation.', 'Boundary guidance.'],
    );

    expect(result.status).toBe('no_matches');
    expect(result.matches).toEqual([]);
  });

  test('reports an unknown paper id as a structured error', () => {
    const result = searchResearchPapers(
      createPaperSearchFixture(),
      { query: 'GMRM', paperId: 'PAPER-UNKNOWN' },
      () => ['GMRM'],
    );

    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'PAPER_NOT_FOUND' }),
      ]),
    );
  });

  test('rejects a paper whose file hash no longer matches metadata', () => {
    const root = createPaperSearchFixture();
    fs.appendFileSync(
      path.join(root, 'PDFpaper', 'PAPER-BDGI.pdf'),
      '-changed',
      'utf8',
    );

    const result = searchResearchPapers(root, { query: 'GMRM' }, () => [
      'GMRM',
    ]);
    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'PAPER_HASH_MISMATCH' }),
      ]),
    );
  });

  test('returns a structured error when pdftotext is unavailable', () => {
    const result = searchResearchPapers(
      createPaperSearchFixture(),
      { query: 'GMRM' },
      () => {
        throw new Error('extractor_unavailable');
      },
    );

    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'PDF_EXTRACTOR_UNAVAILABLE' }),
      ]),
    );
  });

  test('rejects a paper path that leaves the workspace', () => {
    const root = createPaperSearchFixture();
    writeFile(
      root,
      'metadata/papers/PAPER-BDGI.yaml',
      [
        'paper_id: PAPER-BDGI',
        'title: Fixture BDGI Paper',
        'local_file: ../outside.pdf',
        "sha256: '" + '0'.repeat(64) + "'",
        '',
      ].join('\n'),
    );

    const result = searchResearchPapers(root, { query: 'GMRM' }, () => [
      'GMRM',
    ]);
    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'PATH_OUTSIDE_WORKSPACE' }),
      ]),
    );
  });
});

describe('research structured result lookup', () => {
  test('keeps model complexity rows atomic and does not mix adjacent models', () => {
    const root = createResultLookupFixture();
    const hbg = lookupResearchResults(root, {
      model: 'HBGNet',
      table: 'complexity',
    });
    expect(hbg).toMatchObject({
      status: 'ok',
      paperId: 'PAPER-BDGI',
      authority: 'paper_final',
      facts: [
        {
          rowKey: 'PAPER-BDGI:T4:HBGNet',
          values: { model: 'HBGNet', parameters_m: 30.45, flops_g: 463.81 },
          citation: { table: 4, pdfPage: 9 },
        },
      ],
      issues: [],
    });
    expect(JSON.stringify(hbg.facts)).not.toContain('2035.05');

    const bdgi = lookupResearchResults(root, {
      model: 'BDGI-Net',
      table: 'complexity',
    });
    expect(bdgi.facts[0].values).toMatchObject({
      model: 'BDGI-Net',
      parameters_m: 20.09,
      flops_g: 321.98,
    });
  });

  test('filters comparison facts by both model and scene', () => {
    const result = lookupResearchResults(createResultLookupFixture(), {
      model: 'HBGNet',
      scene: 'Shandong',
      table: 'comparison',
    });
    expect(result.facts).toHaveLength(1);
    expect(result.facts[0]).toMatchObject({
      rowKey: 'PAPER-BDGI:T3:Shandong:HBGNet',
      values: { model: 'HBGNet', scene: 'Shandong', iou: 0.8288 },
      citation: { table: 3, pdfPage: 8 },
    });
  });

  test('rejects changed structured result metadata', () => {
    const root = createResultLookupFixture();
    fs.appendFileSync(
      path.join(root, 'metadata', 'results', 'PAPER-BDGI-RESULTS.yaml'),
      '# changed\n',
      'utf8',
    );
    const result = lookupResearchResults(root, {
      model: 'HBGNet',
      table: 'complexity',
    });
    expect(result.status).toBe('error');
    expect(result.issues[0].code).toBe('RESULT_HASH_MISMATCH');
  });
});

describe('research code lookup', () => {
  test('returns a registered source definition with lines and snapshot evidence', () => {
    const root = createCodeSearchFixture();
    const result = lookupResearchCode(root, {
      query: 'GMRM',
      model: 'BDGI-Net',
      maxResults: 3,
    });

    expect(result).toMatchObject({
      status: 'ok',
      workspaceId: 'bdgi-research',
      snapshotId: 'SNAP-TEST',
      requestedModel: 'BDGI-Net',
      searchedModels: ['BDGI-Net'],
      searchedFiles: 1,
      issues: [],
    });
    expect(result.matches[0]).toMatchObject({
      model: 'BDGI-Net',
      matchType: 'definition',
      symbol: 'MaskGuidedByBoundaryModule',
      symbolLine: 3,
      matchLine: 4,
      source: {
        path: 'mymodel/code/model.py',
        snapshotId: 'SNAP-TEST',
      },
    });
    expect(result.matches[0].source.sha256).toBe(
      sha256(path.join(root, 'mymodel', 'code', 'model.py')),
    );
    expect(result.matches[0].source).not.toHaveProperty('absolutePath');
  });

  test('returns no_matches instead of inventing code evidence', () => {
    const result = lookupResearchCode(createCodeSearchFixture(), {
      query: 'quantum_entanglement',
      model: 'BDGI-Net',
    });
    expect(result.status).toBe('no_matches');
    expect(result.matches).toEqual([]);
  });

  test('reports an unknown model as a structured error', () => {
    const result = lookupResearchCode(createCodeSearchFixture(), {
      query: 'GMRM',
      model: 'UNKNOWN',
    });
    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'MODEL_NOT_FOUND', model: 'UNKNOWN' }),
      ]),
    );
  });

  test('rejects source content whose hash no longer matches the manifest', () => {
    const root = createCodeSearchFixture();
    const sourcePath = path.join(root, 'mymodel', 'code', 'model.py');
    const changed = fs.readFileSync(sourcePath, 'utf8').replace('GMRM', 'XMRM');
    fs.writeFileSync(sourcePath, changed, 'utf8');

    const result = lookupResearchCode(root, {
      query: 'GMRM',
      model: 'BDGI-Net',
    });
    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'SOURCE_HASH_MISMATCH' }),
      ]),
    );
  });

  test('rejects a configured code root that leaves the workspace', () => {
    const root = createCodeSearchFixture();
    const configPath = path.join(root, 'research-workspace.json');
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<
      string,
      any
    >;
    config.models['BDGI-Net'].code = '../outside';
    fs.writeFileSync(configPath, JSON.stringify(config), 'utf8');

    const result = lookupResearchCode(root, {
      query: 'GMRM',
      model: 'BDGI-Net',
    });
    expect(result.status).toBe('error');
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'PATH_OUTSIDE_WORKSPACE' }),
      ]),
    );
  });
});

describe('research tool audience output', () => {
  test('hides result snapshot and hashes by default but exposes them for audit', async () => {
    const tools = createResearchWorkspaceTools(createResultLookupFixture());
    const tool = tools.find((item) => item.name === 'research_result_lookup');
    expect(tool).toBeDefined();

    const publicResult = await tool!.handler(
      { model: 'HBGNet', table: 'complexity', max_results: 5 },
      {},
    );
    const publicPayload = JSON.parse(
      (publicResult.content[0] as { type: 'text'; text: string }).text,
    ) as Record<string, unknown>;
    expect(publicPayload).not.toHaveProperty('internalEvidence');
    expect(JSON.stringify(publicPayload)).not.toContain('SNAP-TEST');
    expect(JSON.stringify(publicPayload)).not.toMatch(/[a-f0-9]{64}/i);

    const auditResult = await tool!.handler(
      {
        model: 'HBGNet',
        table: 'complexity',
        max_results: 5,
        include_internal_evidence: true,
      },
      {},
    );
    const auditPayload = JSON.parse(
      (auditResult.content[0] as { type: 'text'; text: string }).text,
    ) as { internalEvidence: { snapshotId: string; resultSha256: string } };
    expect(auditPayload.internalEvidence.snapshotId).toBe('SNAP-TEST');
    expect(auditPayload.internalEvidence.resultSha256).toMatch(
      /^[a-f0-9]{64}$/,
    );
  });

  test('hides code snapshot and hash by default but exposes them for audit', async () => {
    const tools = createResearchWorkspaceTools(createCodeSearchFixture());
    const tool = tools.find((item) => item.name === 'research_code_lookup');
    expect(tool).toBeDefined();

    const publicResult = await tool!.handler(
      { query: 'GMRM', model: 'BDGI-Net', max_results: 3 },
      {},
    );
    const publicPayload = JSON.parse(
      (publicResult.content[0] as { type: 'text'; text: string }).text,
    ) as Record<string, unknown>;
    expect(publicPayload).not.toHaveProperty('snapshotId');
    expect(JSON.stringify(publicPayload)).not.toContain('SNAP-TEST');
    expect(JSON.stringify(publicPayload)).not.toMatch(/[a-f0-9]{64}/i);

    const auditResult = await tool!.handler(
      {
        query: 'GMRM',
        model: 'BDGI-Net',
        max_results: 3,
        include_internal_evidence: true,
      },
      {},
    );
    const auditPayload = JSON.parse(
      (auditResult.content[0] as { type: 'text'; text: string }).text,
    ) as { snapshotId: string; matches: Array<{ source: { sha256: string } }> };
    expect(auditPayload.snapshotId).toBe('SNAP-TEST');
    expect(auditPayload.matches[0].source.sha256).toMatch(/^[a-f0-9]{64}$/);
  });
});

describe('research workspace tool registration', () => {
  test('does not expose the tool in a normal workspace', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'miniclaw-normal-'));
    roots.push(root);
    expect(createResearchWorkspaceTools(root)).toEqual([]);
  });

  test('registers the tool through the existing Runner tool chain', async () => {
    const root = createFixture();
    const directTools = createResearchWorkspaceTools(root);
    expect(directTools.map((item) => item.name)).toEqual([
      'research_workspace_status',
      'research_paper_analyze',
      'research_paper_search',
      'research_result_lookup',
      'research_code_lookup',
    ]);

    const statusTool = directTools.find(
      (item) => item.name === 'research_workspace_status',
    );
    expect(statusTool).toBeDefined();
    const result = await statusTool!.handler({}, {});
    const payload = JSON.parse(
      (result.content[0] as { type: 'text'; text: string }).text,
    ) as { status: string };
    expect(payload.status).toBe('ready');

    const names = createMcpTools({
      chatJid: 'web:research',
      groupFolder: 'research',
      isHome: false,
      isAdminHome: true,
      agentBuilderEnabled: false,
      workspaceIpc: path.join(root, '.ipc'),
      workspaceGroup: root,
    }).map((item) => item.name);
    expect(names).toContain('research_workspace_status');
    expect(names).toContain('research_paper_analyze');
    expect(names).toContain('research_paper_search');
    expect(names).toContain('research_result_lookup');
    expect(names).toContain('research_code_lookup');
  });
});
