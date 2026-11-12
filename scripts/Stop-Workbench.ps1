$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$manifest=Join-Path $projectRoot 'data\processes.json'
if(-not(Test-Path -LiteralPath $manifest)){Write-Output '没有本项目的启动记录。';exit}
$state=Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
if([IO.Path]::GetFullPath($state.root) -ne [IO.Path]::GetFullPath($projectRoot)){throw '启动记录不属于此项目，拒绝停止进程。'}
foreach($name in @('java','python','ollama')) {
    $processId=$state.$name
    if(-not $processId){continue}
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
    if(-not $process){continue}
    if(-not $state.start_times.$name){Write-Warning "进程 $processId 缺少启动时间，拒绝仅凭 PID 停止";continue}
    $expected=[datetime]::Parse($state.start_times.$name).ToUniversalTime()
    if([Math]::Abs(($process.CreationDate.ToUniversalTime()-$expected).TotalSeconds) -gt 2){Write-Warning "进程 $processId 已被复用，跳过";continue}
    $matches=($name -eq 'java' -and $process.CommandLine -like '*patent-workbench-2.0.0.jar*') -or
             ($name -eq 'python' -and $process.CommandLine -like '*uvicorn*retrieval.app:app*8002*') -or
             ($name -eq 'ollama' -and $process.CommandLine -like '*ollama*serve*')
    if(-not $matches){Write-Warning "进程 $processId 与启动记录不符，跳过";continue}
    Stop-Process -Id $processId
}
Remove-Item -LiteralPath $manifest
Write-Output '已停止本项目启动的服务。'
