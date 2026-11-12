param(
    [string]$Python = 'D:\Anaconda\envs\rag\python.exe',
    [string]$Java = 'D:\Java\jdk-21\bin\java.exe',
    [switch]$OpenBrowser,
    [switch]$Build
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'logs') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'data\tmp') | Out-Null
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:TOKENIZERS_PARALLELISM='false'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PATENT_CONFIG=Join-Path $projectRoot 'config\workbench.json'
$env:PATENT_STATE_DIR=Join-Path $projectRoot 'data'
$jar=Join-Path $projectRoot 'app\target\patent-workbench-2.0.0.jar'
if($Build -or -not(Test-Path -LiteralPath $jar)) {
    $env:JAVA_HOME=Split-Path -Parent (Split-Path -Parent $Java)
    $env:PATH=(Split-Path -Parent $Java)+';'+$env:PATH
    & mvn -f (Join-Path $projectRoot 'app\pom.xml') package
    if($LASTEXITCODE -ne 0){throw 'Java 构建失败'}
}
$processes=@{start_times=@{}}
foreach($port in @(8088,8002)) {
    if(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue){throw "端口 $port 已被占用。未停止现有进程，请先检查。"}
}
try {
    try {Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null}
    catch {
        $env:OLLAMA_HOST='127.0.0.1:11434'
        $env:OLLAMA_NUM_PARALLEL='1'
        $env:OLLAMA_MAX_LOADED_MODELS='1'
        $ollama=(Get-Command ollama -ErrorAction Stop).Source
        $p=Start-Process -FilePath $ollama -ArgumentList 'serve' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'logs\ollama.out.log') -RedirectStandardError (Join-Path $projectRoot 'logs\ollama.err.log')
        $processes.ollama=$p.Id
        $processes.start_times.ollama=$p.StartTime.ToUniversalTime().ToString('o')
    }
    # A short project-local socket path also works when the parent process has a long sandbox TEMP.
    $socketTemp=Join-Path $projectRoot 'data\tmp'
    $app=Start-Process -FilePath $Java -ArgumentList @(('"-Djdk.net.unixdomain.tmpdir='+$socketTemp+'"'),'-jar',('"'+$jar+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'logs\java.out.log') -RedirectStandardError (Join-Path $projectRoot 'logs\java.err.log')
    $processes.java=$app.Id
    $processes.start_times.java=$app.StartTime.ToUniversalTime().ToString('o')
    $pythonProcess=Start-Process -FilePath $Python -ArgumentList @('-m','uvicorn','retrieval.app:app','--host','127.0.0.1','--port','8002','--no-server-header') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'logs\retrieval.out.log') -RedirectStandardError (Join-Path $projectRoot 'logs\retrieval.err.log')
    $processes.python=$pythonProcess.Id
    $processes.start_times.python=$pythonProcess.StartTime.ToUniversalTime().ToString('o')
    $processes.root=$projectRoot
    $processes | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $projectRoot 'data\processes.json')
    $ready=$false
    for($i=0;$i -lt 180;$i++) {
        if($app.HasExited -or $pythonProcess.HasExited){throw '服务启动失败，请查看 logs 目录。'}
        try {
            Invoke-RestMethod 'http://127.0.0.1:8088/api/health' -TimeoutSec 1 | Out-Null
            $internal=(Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'data\internal.token')).Trim()
            $health=Invoke-RestMethod 'http://127.0.0.1:8002/health' -Headers @{'X-Internal-Token'=$internal} -TimeoutSec 1
            if($health.ready){$ready=$true;break}
        }catch{}
        Start-Sleep -Seconds 1
    }
    if(-not $ready){throw '启动等待超时，详情请查看 logs 目录。'}
    Write-Output '工作台已就绪：http://127.0.0.1:8088'
    Write-Output '访问令牌位于项目 data\access.token；原始数据以只读方式访问。'
    if($OpenBrowser){$token=(Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'data\access.token')).Trim();Start-Process "http://127.0.0.1:8088/#token=$token"}
} catch {
    $startupError=$_
    foreach($name in @('java','python','ollama')) {
        if(-not $processes.ContainsKey($name)){continue}
        $owned=Get-Process -Id $processes[$name] -ErrorAction SilentlyContinue
        if($owned -and $processes.start_times[$name]) {
            $expected=[datetime]::Parse($processes.start_times[$name]).ToUniversalTime()
            if([Math]::Abs(($owned.StartTime.ToUniversalTime()-$expected).TotalSeconds) -lt 2){Stop-Process -Id $owned.Id -ErrorAction SilentlyContinue}
        }
    }
    $pidFile=Join-Path $projectRoot 'data\processes.json'
    if(Test-Path -LiteralPath $pidFile){Remove-Item -LiteralPath $pidFile}
    throw $startupError
}
