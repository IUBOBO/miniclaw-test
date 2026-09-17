import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {spawn,spawnSync} from 'node:child_process';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const local=path.join(root,'.local');
const logs=path.join(root,'logs');
fs.mkdirSync(local,{recursive:true});fs.mkdirSync(logs,{recursive:true});
const args=process.argv.slice(2),action=args[0]||'start';
const port=Number(args[args.indexOf('--port')+1]||3310);
if(!Number.isInteger(port)||port<1024||port>65535)throw Error('Invalid port');
const entry=path.join(root,'dist','index.js');
const stateFile=path.join(local,'service.json');
const readJson=p=>{try{return JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''))}catch{return null}};
const psQuote=s=>"'"+s.replaceAll("'","''")+"'";
function powershell(code){
 const r=spawnSync('powershell.exe',['-NoProfile','-EncodedCommand',Buffer.from(code,'utf16le').toString('base64')],{encoding:'utf8',windowsHide:true});
 if(r.error)throw r.error;
 if(r.status!==0)throw Error(r.stderr||'PowerShell failed');
 return r.stdout.trim();
}
function owned(s){
 if(!s||s.root!==root||!Number.isInteger(s.pid))return false;
 const p=powershell(`$p=Get-CimInstance Win32_Process -Filter "ProcessId=${s.pid}"; if($p){$p | Select-Object CommandLine,ExecutablePath | ConvertTo-Json -Compress}`);
 if(!p)return false;
 const info=JSON.parse(p);return String(info.CommandLine).toLowerCase().includes(entry.toLowerCase());
}
async function healthy(p){try{const r=await fetch(`http://127.0.0.1:${p}/api/health`,{signal:AbortSignal.timeout(2000)});return r.ok}catch{return false}}
async function listening(p){return new Promise(resolve=>{const s=net.connect({host:'127.0.0.1',port:p});s.setTimeout(1000);s.once('connect',()=>{s.destroy();resolve(true)});s.once('error',()=>resolve(false));s.once('timeout',()=>{s.destroy();resolve(false)})})}
async function stop(){
 const s=readJson(stateFile);
 if(!owned(s)){console.log('This project is not running.');return}
 const owner=powershell(`Get-NetTCPConnection -State Listen -LocalPort ${s.port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique`);
 if(owner.split(/\s+/).includes(String(s.pid))){
  const r=spawnSync('taskkill.exe',['/PID',String(s.pid),'/T','/F'],{encoding:'utf8',windowsHide:true});
  if(r.status!==0)throw Error(r.stderr||r.stdout);
  for(let i=0;i<30&&await listening(s.port);i++)await new Promise(r=>setTimeout(r,200));
  if(owned(s))throw Error('Stop incomplete; inspect logs before restarting.');
  fs.unlinkSync(stateFile);console.log('Stopped this project service. Runtime data retained.');
 }else throw Error('Recorded process is not the listener; refusing to stop it. Inspect status.');
}
function runPnpm(cwd,commands){
 const pm=process.env.MINICLAW_PNPM;
 if(!pm||!fs.existsSync(pm))throw Error('pnpm is required. Install pnpm or set MINICLAW_PNPM to pnpm.cmd.');
 const env={...process.env,ELECTRON_SKIP_BINARY_DOWNLOAD:'1',PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:'1',CI:'true'};
 const script=`& ${psQuote(pm)} ${commands.map(psQuote).join(' ')}; exit $LASTEXITCODE`;
 const r=spawnSync('powershell.exe',['-NoProfile','-EncodedCommand',Buffer.from(script,'utf16le').toString('base64')],{cwd,env,stdio:'inherit',windowsHide:true});
 if(r.error)throw r.error;
 if(r.status!==0)throw Error(`pnpm ${commands.join(' ')} failed in ${cwd}`);
}
const packages=['','web','container/agent-runner'];
function manifestHash(dir){const h=crypto.createHash('sha256');for(const f of ['package.json','pnpm-lock.yaml']){const p=path.join(dir,f);if(fs.existsSync(p))h.update(fs.readFileSync(p))}return h.digest('hex')}
function ensureDependencies(){
 const old=readJson(path.join(local,'dependencies.json'))||{};
 const now={};
 for(const sub of packages){const dir=path.join(root,sub);let hash=manifestHash(dir);
  if(args.includes('--install')||!fs.existsSync(path.join(dir,'node_modules','.modules.yaml'))||old[sub]!==hash){
   console.log(`Installing dependencies: ${sub||'backend'}`);
   const flags=['install','--config.dangerously-allow-all-builds=true'];
   if(fs.existsSync(path.join(dir,'pnpm-lock.yaml')))flags.push('--frozen-lockfile');
   const modules=readJson(path.join(dir,'node_modules','.modules.yaml'));
   if(modules?.storeDir&&fs.existsSync(modules.storeDir)){
    flags.push('--store-dir',path.dirname(modules.storeDir));
   }else{
    flags.push('--store-dir',path.join(local,'pnpm-store'));
    if(fs.existsSync(path.join(dir,'node_modules','.modules.yaml')))flags.push('--force');
   }
   runPnpm(dir,flags);hash=manifestHash(dir);
  }now[sub]=hash;
 }
 fs.writeFileSync(path.join(local,'dependencies.json'),JSON.stringify(now,null,2));
}
function sourceHash(){
 const h=crypto.createHash('sha256');
 function walk(p){if(!fs.existsSync(p))return;const s=fs.statSync(p);if(s.isDirectory()){for(const n of fs.readdirSync(p).sort()){if(['node_modules','dist','.git'].includes(n))continue;walk(path.join(p,n))}}else{h.update(path.relative(root,p));h.update(fs.readFileSync(p))}}
 for(const p of ['src','shared','web/src','web/public','web/index.html','web/vite.config.ts','web/tsconfig.json','web/package.json','web/pnpm-lock.yaml','container/agent-runner/src','container/agent-runner/prompts','container/agent-runner/tsconfig.json','container/agent-runner/package.json','container/agent-runner/pnpm-lock.yaml','package.json','pnpm-lock.yaml','tsconfig.json'])walk(path.join(root,p));
 return h.digest('hex');
}
function buildIfNeeded(){
 const hash=sourceHash(),old=readJson(path.join(local,'build.json'));
 const complete=['dist/index.js','web/dist/index.html','container/agent-runner/dist/pi-index.js'].every(f=>fs.existsSync(path.join(root,f)));
 if(args.includes('--rebuild')||!complete||old?.hash!==hash){
  for(const sub of packages){console.log(`Building ${sub||'backend'}`);runPnpm(path.join(root,sub),['run','build'])}
  fs.writeFileSync(path.join(local,'build.json'),JSON.stringify({hash,builtAt:new Date().toISOString()}));
 }else console.log('Source unchanged; using verified build.');
 return hash;
}
async function main(){
 if(Number(process.versions.node.split('.')[0])<22)throw Error('Node.js 22+ required');
 if(action==='status'){
  const s=readJson(stateFile);const alive=owned(s);const ok=alive&&await healthy(s.port);
  console.log(JSON.stringify({root,running:alive,healthy:ok,url:alive?`http://127.0.0.1:${s.port}`:null,pid:alive?s.pid:null,logs,runnerBuilt:fs.existsSync(path.join(root,'container/agent-runner/dist/pi-index.js'))},null,2));return;
 }
 if(action==='stop'){await stop();return}
 if(!['start','restart','build'].includes(action))throw Error('Unknown action');
 const s=readJson(stateFile),isOwn=owned(s),currentHash=sourceHash();
 if(action==='start'&&isOwn&&await healthy(s.port)&&s.port===port&&s.hash===currentHash&&!args.includes('--rebuild')&&!args.includes('--install')){console.log(`Already running: http://127.0.0.1:${port}`);return}
 if(await listening(port)&&!(isOwn&&s.port===port))throw Error(`Port ${port} belongs to another service. Stop the old MiniClaw or use -Port 3311. Nothing was stopped.`);
 ensureDependencies();const hash=buildIfNeeded();
 if(action==='build'){console.log('Backend, web and Runner built. No service started.');return}
 if(isOwn)await stop();
 const stamp=new Date().toISOString().replace(/[:.]/g,'-');
 const outPath=path.join(logs,`service-${stamp}.out.log`),errPath=path.join(logs,`service-${stamp}.err.log`);
 const out=fs.openSync(outPath,'a'),err=fs.openSync(errPath,'a');
 const child=spawn(process.execPath,[entry],{cwd:root,detached:true,windowsHide:true,stdio:['ignore',out,err],env:{...process.env,WEB_PORT:String(port),WEB_HOST:'127.0.0.1'}});
 fs.closeSync(out);fs.closeSync(err);
 await new Promise((resolve,reject)=>{child.once('spawn',resolve);child.once('error',reject)});
 child.unref();
 const state={root,pid:child.pid,port,hash,startedAt:new Date().toISOString(),stdout:outPath,stderr:errPath};fs.writeFileSync(stateFile,JSON.stringify(state,null,2));
 for(let i=0;i<45;i++){
  if(await healthy(port)){await new Promise(r=>setTimeout(r,1500));if(owned(state)&&await healthy(port)){console.log(`Started: http://127.0.0.1:${port}\nProject: ${root}\nLogs: ${logs}\nNext: initialize admin and configure model provider.`);return}}
  try{process.kill(child.pid,0)}catch{throw Error(`Service exited. Read ${errPath}`)}
  await new Promise(r=>setTimeout(r,500));
 }
 // Stop only the child we just created if startup never becomes healthy.
 if(owned(state))spawnSync('taskkill.exe',['/PID',String(child.pid),'/T','/F'],{windowsHide:true});
 throw Error(`Startup did not pass health check. Read ${errPath}`);
}
let lock;
try{
 const lockFile=path.join(local,'operation.lock');
 try{lock=fs.openSync(lockFile,'wx')}catch{
  const p=Number(fs.readFileSync(lockFile,'utf8'));if(!Number.isInteger(p)||p<=0)throw Error('Startup lock is incomplete. Wait and retry; inspect .local/operation.lock if it persists.');let alive=false;try{process.kill(p,0);alive=true}catch{}
  if(alive)throw Error('Another start/stop/build command is running. Retry after it finishes.');
  fs.unlinkSync(lockFile);lock=fs.openSync(lockFile,'wx');
 }
 fs.writeFileSync(lock,String(process.pid));await main();
}catch(e){console.error(e.message);process.exitCode=1}
finally{if(lock!==undefined){fs.closeSync(lock);fs.unlinkSync(path.join(local,'operation.lock'))}}
