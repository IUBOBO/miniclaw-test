import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'miniclaw-auth-regression-'));
const out = fs.openSync(path.join(tmp, 'server.log'), 'a');
const child = spawn(process.execPath, [path.join(root, 'dist/index.js')], {
  cwd: tmp,
  env: { ...process.env, WEB_PORT: '0', WEB_HOST: '127.0.0.1' },
  stdio: ['ignore', out, out],
  windowsHide: true,
});
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
let base;
try {
  for (let i = 0; i < 100; i++) {
    if (child.exitCode !== null)
      throw Error('Test server exited; inspect ' + tmp);
    const log = fs.readFileSync(path.join(tmp, 'server.log'), 'utf8');
    const cleanLog = log.replace(/\u001b\[[0-9;]*m/g, '');
    const match = [...cleanLog.matchAll(/port:\s*(\d+)/g)].find(
      (m) => Number(m[1]) > 0,
    );
    if (match && Number(match[1]) > 0) {
      base = 'http://127.0.0.1:' + match[1];
      break;
    }
    await pause(200);
  }
  if (!base) throw Error('Could not locate test server port; ' + tmp);
  const creds = {
    username: 'regression_user',
    password: crypto.randomBytes(24).toString('hex'),
  };
  async function request(route, body, expected) {
    const r = await fetch(base + route, {
      method: body ? 'POST' : 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(10000),
    });
    await r.text();
    if (r.status !== expected) throw Error(route + ' status ' + r.status);
    return r;
  }
  await request('/api/auth/setup', creds, 201);
  for (let i = 0; i < 20; i++) await request('/api/auth/login', creds, 200);
  for (let i = 0; i < 90; i++) {
    await request('/api/auth/status', null, 200);
    await pause(1000);
  }
  const result = {
    passed: true,
    setup: 1,
    logins: 20,
    statusChecks: 90,
    minimumObservationSeconds: 90,
    tempDirectory: tmp,
    at: new Date().toISOString(),
  };
  fs.mkdirSync(path.join(root, 'logs'), { recursive: true });
  fs.writeFileSync(
    path.join(root, 'logs', 'auth-regression.json'),
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result));
} finally {
  child.kill();
  fs.closeSync(out);
}
