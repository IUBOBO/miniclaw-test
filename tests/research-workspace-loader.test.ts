import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, test } from 'vitest';

import { createMcpTools } from '../container/agent-runner/src/mcp-tools.js';
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
          'BDGI-Net': { role: 'proposed_model', directory: 'mymodel' },
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
    ]);

    const result = await directTools[0].handler({}, {});
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
  });
});
