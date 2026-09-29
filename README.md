<h1 align="center">APG-GS ChainMail</h1>

<p align="center"><strong>Interactive, appearance-preserving deformation for 3D Gaussian Splatting</strong></p>

<p align="center">
  <a href="https://minn-k.github.io/3d-representation-portfolio/"><img src="https://img.shields.io/badge/Portfolio-Website-green?logo=googlechrome&logoColor=white" alt="Portfolio"></a>
  <a href="https://github.com/minn-k/3dgs-isaac-sim-xpbd"><img src="https://img.shields.io/badge/Later_work-Isaac_Sim_XPBD-blue?logo=github" alt="Isaac Sim XPBD runtime"></a>
  <img src="https://img.shields.io/badge/Language-C%2B%2F%2FCUDA-00599C?logo=cplusplus&logoColor=white" alt="C++ and CUDA">
  <img src="https://img.shields.io/badge/Viewer-SIBR_Gaussian_Viewer-4B8BBE" alt="SIBR Gaussian Viewer">
</p>

## Overview

APG-GS ChainMail is a source overlay for the SIBR Gaussian Viewer. It gives an unstructured 3D Gaussian Splatting
scene a deformation structure: a Gaussian graph receives a mouse-drag constraint, ChainMail propagates it through the
neighbourhood, and a local deformation estimate updates each affected Gaussian's position, orientation, and scale.

The repository preserves the earlier ChainMail research scope. The later OpenUSD, Isaac Sim, CUDA XPBD, and two-arm
interaction runtime is released separately in [3dgs-isaac-sim-xpbd](https://github.com/minn-k/3dgs-isaac-sim-xpbd).

## Pipeline

~~~text
Trained 3DGS PLY
  → Gaussian graph construction
  → mouse pick / drag constraint
  → CPU or CUDA ChainMail propagation
  → local affine deformation estimate
  → Gaussian covariance update
  → SIBR real-time renderer
~~~

## What is implemented

- **Gaussian-aware graph.** Spatial k-NN candidates are weighted by centre distance, quaternion alignment, scale
  aspect ratio, and luminance-weighted spherical-harmonic appearance similarity. Stronger edges connect more similar
  Gaussians.
- **Interactive propagation.** A picked Gaussian becomes a drag constraint. CPU and CUDA ChainMail implementations,
  active-region updates, and graph diagnostics support responsive editing.
- **Appearance-preserving update.** The CUDA path estimates a local affine map from rest and deformed neighbour
  offsets. A regularized least-squares estimate is decomposed into rotation and scale before the Gaussian covariance is
  updated for rendering.
- **Scene-independent runtime.** The graph is built from every Gaussian in the loaded PLY; no scene-specific crop
  coordinates or point IDs are embedded in the public runtime.

## Install and build

The upstream SIBR viewer is a dependency and is intentionally not re-hosted here. <code>overlay/</code> contains every source file
added or changed by this project, including the CUDA rasterizer extension and its bundled GLM headers.

### Requirements

- A compatible SIBR Core / Gaussian-viewer checkout containing
  <code>src/projects/gaussianviewer/renderer/GaussianView.cpp</code>
- CMake, Visual Studio C++ Build Tools, CUDA Toolkit, and Eigen 3 headers
- On Windows, Eigen can be installed with <code>vcpkg install eigen3:x64-windows</code>

### Apply the overlay deliberately

~~~powershell
# Preview first; this makes no change to the SIBR checkout.
.\scripts\apply_overlay.ps1 -SibrRoot "D:\src\SIBR_viewers"

# Apply only after reviewing the file list.
.\scripts\apply_overlay.ps1 -SibrRoot "D:\src\SIBR_viewers" -Apply

.\scripts\build_viewer.ps1 `
  -SibrRoot "D:\src\SIBR_viewers" `
  -Eigen3Include "D:\vcpkg\installed\x64-windows\include"
~~~

See [docs/build.md](docs/build.md) for the full setup and troubleshooting notes. The usual viewer executable is
<code>SIBR_gaussianViewer_app</code>.

## Interaction

Open a trained 3DGS model in the Gaussian Viewer, enable picking, and drag a Gaussian. The graph overlay exposes
CPU/GPU ChainMail selection, propagation controls, active-map settings, and visual graph diagnostics.

## Repository map

~~~text
overlay/     source files applied to a compatible SIBR checkout
scripts/     explicit overlay and Windows build helpers
docs/        system design and build notes
LICENSES/    upstream license copies and notices
~~~

## Reproducibility and scope

- The public overlay contains source code, not SIBR itself.
- Datasets, trained PLY models, build outputs, binaries, videos, personal data, and presentation files are excluded.
- Source-level checks pass. A complete SIBR build depends on the platform-specific stack described in the build guide.
- This project is an interactive deformation prototype, not a physical material simulator.

## License and attribution

This repository contains derivative SIBR and 3D Gaussian Splatting components. Use and redistribute the code under the
upstream terms in [LICENSE.md](LICENSE.md) and [LICENSES](LICENSES), and preserve included notices when redistributing
modifications.
