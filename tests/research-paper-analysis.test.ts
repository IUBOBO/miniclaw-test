import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, test } from 'vitest';

import {
  analyzeResearchPaper,
  audienceResearchPaperAnalysis,
} from '../container/agent-runner/src/research/paper-analysis.js';

const roots: string[] = [];

function sha256(filePath: string): string {
  return crypto
    .createHash('sha256')
    .update(fs.readFileSync(filePath))
    .digest('hex');
}

function writeFile(
  root: string,
  relativePath: string,
  content: string,
): string {
  const target = path.join(root, ...relativePath.split('/'));
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, content, 'utf8');
  return target;
}

function createFixture(): string {
  const root = fs.mkdtempSync(
    path.join(os.tmpdir(), 'miniclaw-paper-analysis-'),
  );
  roots.push(root);
  fs.mkdirSync(path.join(root, 'metadata', 'papers'), { recursive: true });
  fs.mkdirSync(path.join(root, 'metadata', 'results'), { recursive: true });
  fs.mkdirSync(path.join(root, 'PDFpaper'), { recursive: true });
  writeFile(
    root,
    'research-workspace.json',
    JSON.stringify({
      schema_version: 2,
      root: '.',
      workspace_id: 'paper-analysis-test',
      snapshot_id: 'SNAP-PAPER-TEST',
      manifest: 'snapshot-manifest.jsonl',
      metadata_root: 'metadata',
    }),
  );
  writeFile(root, 'snapshot-manifest.jsonl', '');
  return root;
}

function representativePages(): string[] {
  return [
    [
      'A New Segmentation Method',
      'Abstract',
      'This paper studies parcel segmentation and proposes a boundary-aware framework.',
    ].join('\n'),
    [
      '1. Introduction',
      'Existing methods lose boundary details. We study how to preserve them.',
    ].join('\n'),
    [
      '3 Method',
      'The proposed method combines semantic features with boundary guidance.',
    ].join('\n'),
    [
      '4 Experiments',
      'The method is evaluated on three remote-sensing datasets.',
      '4.1 Experimental setup',
    ].join('\n'),
    [
      '5 Conclusion',
      'The paper reports improved parcel segmentation and discusses future work.',
    ].join('\n'),
  ];
}

afterEach(() => {
  while (roots.length > 0) {
    fs.rmSync(roots.pop() as string, { recursive: true, force: true });
  }
});

describe('research paper analysis', () => {
  test('analyzes an unregistered PDF as paper-only evidence', () => {
    const root = createFixture();
    writeFile(root, 'PDFpaper/unknown.pdf', 'fixture-pdf');

    const result = analyzeResearchPaper(
      root,
      { paperPath: 'PDFpaper/unknown.pdf' },
      representativePages,
    );

    expect(result).toMatchObject({
      status: 'ok',
      paper: {
        paperId: null,
        title: 'unknown',
        path: 'PDFpaper/unknown.pdf',
        registration: 'provisional',
        pageCount: 5,
      },
      evidenceMode: 'paper_only',
      coverage: {
        abstract: true,
        introduction: true,
        method: true,
        experiments: true,
        conclusion: true,
      },
      capabilities: {
        summarizePaper: true,
        explainReportedMethod: true,
        analyzeReportedExperiments: true,
        verifySourceCode: false,
        verifyExecution: false,
        reproduceExperiment: false,
      },
    });
    expect(result.sections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'abstract', page: 1 }),
        expect.objectContaining({ kind: 'method', page: 3 }),
        expect.objectContaining({ kind: 'experiments', page: 4 }),
      ]),
    );
    expect(result.missingMaterial.map((item) => item.code)).toEqual(
      expect.arrayContaining([
        'SOURCE_CODE_NOT_LINKED',
        'CONFIG_NOT_LINKED',
        'LOG_NOT_LINKED',
        'DATA_NOT_LINKED',
      ]),
    );
    expect(result.internalEvidence?.sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  test('detects registered paper links without claiming experiment reproduction', () => {
    const root = createFixture();
    const pdfPath = writeFile(
      root,
      'PDFpaper/registered.pdf',
      'registered-pdf',
    );
    writeFile(root, 'code/model.py', 'class Model: pass');
    writeFile(
      root,
      'metadata/papers/PAPER-REGISTERED.yaml',
      [
        'paper_id: PAPER-REGISTERED',
        'title: Registered Paper',
        'local_file: PDFpaper/registered.pdf',
        `sha256: ${sha256(pdfPath)}`,
        'text_extractable: true',
        'linked_source_files:',
        '  - code/model.py',
        '',
      ].join('\n'),
    );
    writeFile(
      root,
      'metadata/results/PAPER-REGISTERED-RESULTS.yaml',
      'paper_id: PAPER-REGISTERED\n',
    );

    const result = analyzeResearchPaper(
      root,
      { paperId: 'PAPER-REGISTERED' },
      representativePages,
    );

    expect(result.paper).toMatchObject({
      paperId: 'PAPER-REGISTERED',
      registration: 'registered',
    });
    expect(result.evidenceMode).toBe('paper_with_linked_materials');
    expect(result.linkedMaterials).toMatchObject({
      sourceCode: true,
      structuredResults: true,
      configurations: false,
      logs: false,
    });
    expect(result.capabilities.verifySourceCode).toBe(true);
    expect(result.capabilities.verifyExecution).toBe(false);
    expect(result.capabilities.reproduceExperiment).toBe(false);
  });

  test('does not invent experiments when the paper has no experiment section', () => {
    const root = createFixture();
    writeFile(root, 'PDFpaper/concept.pdf', 'concept-paper');
    const result = analyzeResearchPaper(
      root,
      { paperPath: 'PDFpaper/concept.pdf' },
      () => [
        'Abstract\nA conceptual framework is introduced for scientific reasoning.',
        '1 Introduction\nThis work defines the research problem.',
        '2 Method\nWe formulate a theoretical workflow.',
        '3 Conclusion\nThe paper summarizes the proposed framework.',
      ],
    );

    expect(result.status).toBe('ok');
    expect(result.coverage.experiments).toBe(false);
    expect(result.coverage.results).toBe(false);
    expect(result.capabilities.analyzeReportedExperiments).toBe(false);
  });

  test('reports OCR required for a PDF without usable text', () => {
    const root = createFixture();
    writeFile(root, 'PDFpaper/scanned.pdf', 'scanned-paper');
    const result = analyzeResearchPaper(
      root,
      { paperPath: 'PDFpaper/scanned.pdf' },
      () => [' ', ''],
    );

    expect(result.status).toBe('partial');
    expect(result.sections).toEqual([]);
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: 'OCR_REQUIRED' }),
    );
    expect(result.capabilities.summarizePaper).toBe(false);
  });

  test('rejects paths outside the research workspace', () => {
    const result = analyzeResearchPaper(
      createFixture(),
      { paperPath: '../outside.pdf' },
      representativePages,
    );
    expect(result.status).toBe('error');
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: 'PATH_OUTSIDE_WORKSPACE' }),
    );
  });

  test('rejects conflicting paper id and path inputs', () => {
    const root = createFixture();
    writeFile(root, 'PDFpaper/unknown.pdf', 'fixture-pdf');
    const result = analyzeResearchPaper(
      root,
      { paperId: 'PAPER-X', paperPath: 'PDFpaper/unknown.pdf' },
      representativePages,
    );
    expect(result.status).toBe('error');
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: 'INPUT_CONFLICT' }),
    );
  });

  test('hides snapshot and hash from normal audience output', () => {
    const root = createFixture();
    writeFile(root, 'PDFpaper/unknown.pdf', 'fixture-pdf');
    const internal = analyzeResearchPaper(
      root,
      { paperPath: 'PDFpaper/unknown.pdf' },
      representativePages,
    );
    const audience = audienceResearchPaperAnalysis(internal);

    expect(audience).not.toHaveProperty('internalEvidence');
    expect(audience.paper).not.toHaveProperty('registration');
    expect(JSON.stringify(audience)).not.toContain('SNAP-PAPER-TEST');
    expect(internal.internalEvidence?.snapshotId).toBe('SNAP-PAPER-TEST');
  });
});
