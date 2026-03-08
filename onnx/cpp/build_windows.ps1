param(
    [string]$BuildDir = "build",
    [string]$Generator = "NMake Makefiles",
    [string]$OnnxRuntimeRoot = "",
    [string]$OnnxRuntimeDll = "",
    [switch]$SkipBuild,
    [switch]$SkipDevCmd
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "[build_windows] $Message"
}

function Ensure-CMake([string]$RepoRoot) {
    $pythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (!(Test-Path $pythonExe)) {
        throw "Python venv not found: $pythonExe"
    }

    $cmakeExe = Join-Path $RepoRoot ".venv\Scripts\cmake.exe"
    if (!(Test-Path $cmakeExe)) {
        Write-Step "CMake not found in venv. Installing cmake package..."
        & $pythonExe -m pip install --upgrade cmake
    }

    if (!(Test-Path $cmakeExe)) {
        throw "Failed to provision CMake at: $cmakeExe"
    }

    return $cmakeExe
}

function Get-VsWherePath {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        return $vswhere
    }
    return $null
}

function Test-VcToolsInstalled {
    $vswhere = Get-VsWherePath
    if ($null -eq $vswhere) {
        return $false
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return -not [string]::IsNullOrWhiteSpace(($installPath | Out-String))
}

function Import-VsDevEnvironment {
    $vswhere = Get-VsWherePath
    if ($null -eq $vswhere) {
        Write-Step "vswhere not found. Skipping DevCmd import."
        return
    }

    $installationPath = (& $vswhere -latest -products * -property installationPath | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($installationPath)) {
        Write-Step "Visual Studio Build Tools installation not found."
        return
    }

    $devCmd = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
    if (!(Test-Path $devCmd)) {
        Write-Step "VsDevCmd.bat not found at: $devCmd"
        return
    }

    Write-Step "Importing VS developer environment from: $devCmd"

    $tempBat = Join-Path $env:TEMP ("gliner_import_dev_env_" + [Guid]::NewGuid().ToString("N") + ".bat")
    @(
        "@echo off",
        ('call "' + $devCmd + '" -arch=x64 -host_arch=x64 >nul 2>nul'),
        "set"
    ) | Set-Content -Path $tempBat -Encoding ASCII

    $envDump = cmd /c $tempBat
    Remove-Item -Path $tempBat -Force -ErrorAction SilentlyContinue

    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") {
            Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
        }
    }
}

function Assert-CompilerAvailable {
    $cl = Get-Command cl -ErrorAction SilentlyContinue
    $nmake = Get-Command nmake -ErrorAction SilentlyContinue

    if ($null -ne $cl -and $null -ne $nmake) {
        Write-Step "Compiler toolchain detected: cl + nmake"
        return
    }

    $vcInstalled = Test-VcToolsInstalled

    if (-not $vcInstalled) {
        Write-Host ""
        Write-Host "[ERROR] VC++ toolchain is not installed yet (cl/nmake missing)." -ForegroundColor Red
        Write-Host "Run the following command in an elevated (Administrator) terminal:" -ForegroundColor Yellow
        Write-Host '  "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
        Write-Host "Then re-run this script."
        throw "Missing VC++ workload"
    }

    Write-Host ""
    Write-Host "[ERROR] VC++ workload exists but current shell has no cl/nmake in PATH." -ForegroundColor Red
    Write-Host "Open 'x64 Native Tools Command Prompt for VS 2022' or run this script with -SkipDevCmd:$false so it can import DevCmd." -ForegroundColor Yellow
    throw "Compiler tools unavailable in current environment"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..\")).Path

Write-Step "Repository root: $repoRoot"
$cmakeExe = Ensure-CMake -RepoRoot $repoRoot
Write-Step "Using CMake: $cmakeExe"

if (-not $SkipDevCmd) {
    Import-VsDevEnvironment
}

Assert-CompilerAvailable

$configureArgs = @(
    "-S", $scriptDir,
    "-B", (Join-Path $scriptDir $BuildDir),
    "-G", $Generator
)

if (-not [string]::IsNullOrWhiteSpace($OnnxRuntimeRoot)) {
    $configureArgs += "-DONNXRUNTIME_ROOT=$OnnxRuntimeRoot"
}
if (-not [string]::IsNullOrWhiteSpace($OnnxRuntimeDll)) {
    $configureArgs += "-DONNXRUNTIME_DLL=$OnnxRuntimeDll"
}

Write-Step "Configuring project..."
& $cmakeExe @configureArgs

if (-not $SkipBuild) {
    Write-Step "Building project..."
    & $cmakeExe --build (Join-Path $scriptDir $BuildDir) --config Release
}

Write-Step "Done."
