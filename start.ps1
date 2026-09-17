param(
    [ValidateSet('Start','Stop','Status','Restart','Build')][string]$Action='Start',
    [ValidateRange(1024,65535)][int]$Port=3310,
    [switch]$Rebuild,
    [switch]$Install
)
$ErrorActionPreference='Stop'
$nodeCmd=Get-Command node -ErrorAction SilentlyContinue
$nodeExe=if($nodeCmd){$nodeCmd.Source}else{Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'}
if(-not(Test-Path -LiteralPath $nodeExe)){throw 'Node.js 20+ is required. Install Node.js and retry.'}
$env:PATH=(Split-Path -Parent $nodeExe)+';'+$env:PATH
$pnpmCmd=Get-Command pnpm.cmd -ErrorAction SilentlyContinue
$pnpmExe=if($pnpmCmd){$pnpmCmd.Source}else{Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd'}
if(Test-Path -LiteralPath $pnpmExe){$env:MINICLAW_PNPM=$pnpmExe}
$opts=@((Join-Path $PSScriptRoot 'scripts\local-service.mjs'),$Action.ToLower(),'--port',"$Port")
if($Rebuild){$opts+='--rebuild'}
if($Install){$opts+='--install'}
& $nodeExe @opts
exit $LASTEXITCODE
