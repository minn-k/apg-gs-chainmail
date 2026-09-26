# APG-GS ChainMail

A focused source overlay for interactive deformation of 3D Gaussian Splatting scenes in the SIBR Gaussian viewer. The project builds a Gaussian-aware graph, propagates a user drag through a ChainMail solver, and updates Gaussian orientation and scale for real-time rendering.

This repository preserves the earlier ChainMail research scope. The later OpenUSD, Isaac Sim, and two-arm XPBD runtime is released separately in [3dgs-isaac-sim-xpbd](https://github.com/minn-k/3dgs-isaac-sim-xpbd).

## What the system does

~~~text
Trained 3DGS PLY
  → Gaussian graph construction
  → mouse pick / drag constraint
  → CPU or CUDA ChainMail propagation
  → local deformation estimate
  → Gaussian covariance update
  → SIBR real-time renderer
~~~

- **Gaussian graph.** Candidate neighbors come from a spatial k-NN search. Each edge combines center distance, quaternion orientation alignment, scale aspect-ratio difference, and luminance-weighted spherical-harmonic appearance similarity. Low dissimilarity scores become stronger ChainMail edges. The default graph is built from every Gaussian in the loaded PLY; no scene-specific crop coordinates or point IDs are embedded in the runtime.
- **Interactive deformation.** A picked Gaussian becomes a drag constraint. The solver propagates positional corrections through the graph, with CPU and CUDA ChainMail paths plus active-region updates for responsive interaction.
- **Appearance-preserving update.** For each affected Gaussian, the CUDA path estimates a local affine deformation from rest and deformed neighbor offsets. A regularized least-squares estimate is decomposed into rotation and scale before the Gaussian covariance is updated for rendering.

## Repository form

The SIBR viewer is an upstream dependency and is not re-hosted here. `overlay/` contains every source file added or changed by this work, including the CUDA rasterizer extension and its bundled GLM headers. Apply it to a compatible SIBR Gaussian-viewer checkout with the supplied script.

No datasets, trained PLY models, build outputs, binaries, videos, personal data, or presentation files are tracked.

## Install and build

1. Prepare a clean, compatible SIBR Core / Gaussian-viewer source checkout. Its Gaussian viewer must contain `src/projects/gaussianviewer/renderer/GaussianView.cpp`.
2. Install CMake, Visual Studio C++ Build Tools, CUDA Toolkit, and Eigen 3 headers. On Windows, vcpkg can provide Eigen with `vcpkg install eigen3:x64-windows`.
3. Preview the overlay files, then apply them deliberately:

~~~powershell
.\scripts\apply_overlay.ps1 -SibrRoot "D:\src\SIBR_viewers"
.\scripts\apply_overlay.ps1 -SibrRoot "D:\src\SIBR_viewers" -Apply
~~~

4. Configure and build the Gaussian viewer:

~~~powershell
.\scripts\build_viewer.ps1 `
  -SibrRoot "D:\src\SIBR_viewers" `
  -Eigen3Include "D:\vcpkg\installed\x64-windows\include"
~~~

See [docs/build.md](docs/build.md) for prerequisites and troubleshooting. The usual viewer entry point is `SIBR_gaussianViewer_app`.

## Interaction

Open a trained 3DGS model in the Gaussian viewer, enable picking, then select and drag a Gaussian. The graph overlay controls expose CPU/GPU ChainMail selection, propagation parameters, active-map settings, and visual graph diagnostics.

## Layout

~~~text
overlay/     source files applied to a compatible SIBR checkout
scripts/     explicit overlay and Windows build helpers
docs/        system design and build notes
LICENSES/    upstream license copies and notices
~~~

## Validation status

The public overlay was curated from the ChainMail implementation, stripped of later solver and simulator code, and scanned for local paths, datasets, binaries, and generated artifacts. Source-level checks pass; a full SIBR application build requires the platform-specific dependency stack described in the build guide.

## License and attribution

This repository contains derivative SIBR and 3D Gaussian Splatting components. Use and redistribute the code under the upstream terms in [LICENSE.md](LICENSE.md) and [LICENSES](LICENSES). Preserve the included notices when redistributing modifications.
