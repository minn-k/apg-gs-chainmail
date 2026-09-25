[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SibrRoot,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$overlayRoot = Join-Path $repoRoot "overlay"
$targetRoot = (Resolve-Path -LiteralPath $SibrRoot).Path

function Get-OverlayRelativePath {
    param([string]$BasePath, [string]$TargetPath)
    $separator = [IO.Path]::DirectorySeparatorChar
    $baseFull = [IO.Path]::GetFullPath($BasePath).TrimEnd($separator) + $separator
    $targetFull = [IO.Path]::GetFullPath($TargetPath)
    if (-not $targetFull.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the overlay root: $TargetPath"
    }
    return $targetFull.Substring($baseFull.Length)
}

foreach ($required in @(
    "CMakeLists.txt",
    "src\projects\gaussianviewer\renderer\GaussianView.cpp",
    "src\projects\gaussianviewer\apps\gaussianViewer\main.cpp"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $targetRoot $required))) {
        throw "The target does not look like a compatible SIBR Gaussian-viewer checkout: missing $required"
    }
}

$files = Get-ChildItem -LiteralPath $overlayRoot -Recurse -File | Sort-Object FullName
if (-not $Apply) {
    Write-Host "Dry run: $($files.Count) overlay files would be copied to $targetRoot"
    $files | Select-Object -First 20 | ForEach-Object { Write-Host (Get-OverlayRelativePath $overlayRoot $_.FullName) }
    if ($files.Count -gt 20) { Write-Host "... and $($files.Count - 20) more files" }
    Write-Host "Run again with -Apply after reviewing the target checkout."
    return
}

foreach ($file in $files) {
    $relative = Get-OverlayRelativePath $overlayRoot $file.FullName
    if ($relative.StartsWith("..")) { throw "Refusing invalid overlay path: $relative" }
    $destination = Join-Path $targetRoot $relative
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
}

Write-Host "Applied $($files.Count) ChainMail overlay files to $targetRoot"