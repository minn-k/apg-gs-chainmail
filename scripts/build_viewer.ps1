[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SibrRoot,
    [Parameter(Mandatory = $true)]
    [string]$Eigen3Include,
    [ValidateSet("Release", "RelWithDebInfo", "Debug")]
    [string]$Configuration = "Release",
    [string]$Generator = "Visual Studio 17 2022"
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path -LiteralPath $SibrRoot).Path
$eigen = (Resolve-Path -LiteralPath $Eigen3Include).Path
if (-not (Test-Path -LiteralPath (Join-Path $eigen "Eigen\Core"))) {
    throw "Eigen3Include must directly contain Eigen\Core: $eigen"
}
$build = Join-Path $source "build"

& cmake -S $source -B $build -G $Generator -A x64 "-DEIGEN3_INCLUDE_DIR=$eigen"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
& cmake --build $build --config $Configuration --target SIBR_gaussianViewer_app
if ($LASTEXITCODE -ne 0) { throw "Viewer build failed." }
