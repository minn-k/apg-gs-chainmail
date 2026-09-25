# System design

## 1. Gaussian graph construction

The graph builder first finds spatial k-nearest-neighbor candidates for every Gaussian. It evaluates each candidate pair with a weighted dissimilarity score:

~~~text
spatial separation + orientation mismatch + scale-shape mismatch + SH appearance mismatch
~~~

The score is thresholded by a data-dependent percentile. Accepted edges are deduplicated and assigned a rest length and stiffness. A lower score maps to a stronger edge, so nearby Gaussians with compatible orientation, shape, and appearance transmit deformation more strongly.

## 2. Constraint-driven ChainMail deformation

A screen-space pick selects a Gaussian. During a drag, the viewer writes the target motion as a constraint and asks the solver to update the connected graph. The CUDA path keeps graph buffers and the active region on the GPU, applies propagation and relaxation passes, and returns the changed Gaussian positions to the renderer.

## 3. Covariance-aware rendering update

Moving only the Gaussian centers makes a deformed object look disconnected. For every changed Gaussian, the CUDA kernel uses its rest and deformed neighbors to estimate a local affine transform:

~~~text
F = Q Pᵀ (P Pᵀ + λI)⁻¹
~~~

The small diagonal term stabilizes sparse or nearly planar neighborhoods. SVD/polar decomposition separates rotation from scale, then the renderer receives updated orientation and scale with the position update. This keeps the Gaussian ellipsoids aligned with the local deformation.

## 4. Runtime boundary

~~~text
Viewer input → pick / drag target → ChainMail graph solver → position + covariance update → Gaussian renderer
                     CPU control        CUDA buffers                 CUDA rasterizer
~~~

The source retains both CPU and CUDA ChainMail modes for comparison. The CUDA path is the runtime path used for interactive-scale scenes.
