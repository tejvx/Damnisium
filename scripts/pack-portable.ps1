$ErrorActionPreference = "Stop"

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:USERPROFILE\.cargo\bin"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasMsvc = $false
if (Test-Path $vswhere) {
    $vcPath = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vcPath) { $hasMsvc = $true }
}

pnpm run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($hasMsvc) {
    pnpm exec tauri build --no-bundle
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $exeSource = Join-Path $PSScriptRoot "..\src-tauri\target\release\damnisium.exe"
} else {
    $toolsRoot = Join-Path $env:USERPROFILE ".damnisium-build-tools"
    $llvmDir = Get-ChildItem -Path $toolsRoot -Directory -Filter "llvm-mingw*" | Select-Object -First 1
    if (-not $llvmDir) {
        Write-Error "Neither MSVC Build Tools nor portable .damnisium-build-tools found."
        exit 1
    }
    $clangCl = Join-Path $llvmDir.FullName "bin\clang-cl.exe"
    if (-not (Test-Path $clangCl)) {
        Copy-Item -Force (Join-Path $llvmDir.FullName "bin\clang-23.exe") $clangCl
    }
    $rustBin = Join-Path $env:USERPROFILE ".rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\x86_64-pc-windows-msvc\bin"
    $xwin = Join-Path $env:LOCALAPPDATA "cargo-xwin\xwin"
    $env:Path = (Join-Path $llvmDir.FullName "bin") + ";$rustBin;" + $env:Path
    $env:LIB = "$xwin\crt\lib\x86_64;$xwin\sdk\lib\um\x86_64;$xwin\sdk\lib\ucrt\x86_64"
    $env:INCLUDE = "$xwin\crt\include;$xwin\sdk\include\ucrt;$xwin\sdk\include\um;$xwin\sdk\include\shared;$xwin\sdk\include\winrt;$xwin\sdk\include\cppwinrt"
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = "$rustBin\rust-lld.exe"
    $env:RUSTFLAGS = "-C linker-flavor=lld-link"
    $env:CC_x86_64_pc_windows_msvc = $clangCl
    $env:CXX_x86_64_pc_windows_msvc = $clangCl
    $env:AR_x86_64_pc_windows_msvc = Join-Path $llvmDir.FullName "bin\llvm-ar.exe"
    $env:RC = Join-Path $llvmDir.FullName "bin\llvm-rc.exe"
    pnpm exec tauri build --no-bundle --target x86_64-pc-windows-msvc
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $exeSource = Join-Path $PSScriptRoot "..\src-tauri\target\x86_64-pc-windows-msvc\release\damnisium.exe"
}

$exeDest = Join-Path $PSScriptRoot "..\Damnisium.exe"
Get-Process -Name "Damnisium*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 300
Copy-Item -Force $exeSource $exeDest

$sizeMb = [Math]::Round((Get-Item $exeDest).Length / 1MB, 2)
Write-Host "Portable build ready: $exeDest ($sizeMb MB)"
