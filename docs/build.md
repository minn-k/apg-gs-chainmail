# Build guide

## Required environment

- Windows 10 or 11 with an NVIDIA GPU
- A compatible SIBR Core / Gaussian viewer checkout
- CMake 3.22 or newer
- Visual Studio 2022 C++ Build Tools (Desktop development with C++)
- CUDA Toolkit compatible with the local driver and viewer build
- Eigen 3 headers

The CUDA rasterizer CMake file accepts `EIGEN3_INCLUDE_DIR`. Pass the directory that directly contains the `Eigen` folder. For a vcpkg installation that is usually:

~~~text
<vcpkg>\installed\x64-windows\include
~~~

## Apply the source overlay

From this repository:

~~~powershell
.\scripts\apply_overlay.ps1 -SibrRoot "D:\src\SIBR_viewers"
.\scripts\apply_overlay.ps1 -SibrRoot "D:\src\SIBR_viewers" -Apply
~~~

The first command only lists the files that would change. Use a clean Git worktree for the target checkout so the replacement is reviewable.

## Configure and build

~~~powershell
.\scripts\build_viewer.ps1 `
  -SibrRoot "D:\src\SIBR_viewers" `
  -Eigen3Include "D:\vcpkg\installed\x64-windows\include" `
  -Configuration Release
~~~

The helper runs CMake configure followed by a Release build of `SIBR_gaussianViewer_app`. If the host uses a different generator, pass `-Generator` explicitly.

## Run

Use the normal SIBR Gaussian-viewer command with an existing trained 3DGS model:

~~~powershell
.\install\bin\SIBR_gaussianViewer_app.exe --model-path "D:\models\scene" --iteration 30000 --device 0
~~~

Model data is intentionally not supplied by this repository.
