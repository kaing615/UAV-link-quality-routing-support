param(
    [string]$Distribution = 'Ubuntu',
    [string]$Ns3Prefix = '/home/kainguyen/ns-3.45/build'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$translated = & rtk proxy wsl -d $Distribution --exec wslpath -a $repoRoot.Replace('\', '/')
if ($LASTEXITCODE -ne 0) { throw 'Cannot translate project path for WSL' }
$linuxRoot = ($translated | Select-Object -Last 1).Trim()
$buildPath = "$linuxRoot/build/recovery"
& rtk proxy wsl -d $Distribution -- cmake -S "$linuxRoot/simulation/ns3" -B $buildPath "-DNS3_PREFIX=$Ns3Prefix" -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'ns-3 configure failed' }
& rtk proxy wsl -d $Distribution -- cmake --build $buildPath -j 2
if ($LASTEXITCODE -ne 0) { throw 'ns-3 build failed' }
$files = @('simulation/ns3/uav-olsr-dataset.cc', 'simulation/ns3/CMakeLists.txt',
           'simulation/ns3/recovery-control.h', 'simulation/ns3/recovery-protocol.h',
           'simulation/ns3/routing-diagnostics.h',
           'simulation/ns3/service-observation.h',
           'build/recovery/uav-olsr-dataset')
$digests = foreach ($relative in $files) {
    $stream = [IO.File]::OpenRead((Join-Path $repoRoot $relative))
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $digest = ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hasher.Dispose() }
    "$digest  $relative"
}
$digests | Set-Content -Encoding ascii -LiteralPath (Join-Path $repoRoot 'build/recovery/source.sha256')
Write-Output "Recovery simulator: $repoRoot\build\recovery\uav-olsr-dataset"
