/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include<cuda_fp16.h>
#include <cuda_runtime.h>
#include "header.h"
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <queue>
#include <set>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
namespace cg = cooperative_groups;
static float t = 0;

// GPU command mode: CPU only submits (idx, delta) commands, GPU applies them.
static bool g_gpuCommandMode = false;
static std::vector<int> g_cmdIdx;
static std::vector<glm::vec3> g_cmdDelta;
static int bfsHops = 2;
// Chainmail active map mode (GPU)
static int g_GpuChainmailMode = 1;
static bool g_useActiveMap = false;
static bool g_activeMapReset = false;
static int g_lastActiveCount = 0;
static float g_lastActiveRatio = 0.0f;
// ChainMail tunables (UI controllable)
static int g_cmPropIters = 20;
static int g_cmRelaxIters = 5;
static float g_cmPropStrength = 0.5f;
static float g_cmStiffness = 0.99f;
static float g_cmDamping = 0.1f;
// ChainMail material controls (defaults preserve legacy behavior).
static float g_cmConstraintGlobalScale = 1.0f;
static float g_cmConstraintAirScale = 0.1f;
static float g_cmConstraintSkinScale = 0.1f;
static float g_cmConstraintBoneScale = 0.1f;
static bool g_cmUseEdgeStiffness = true;
static float g_cmEdgeStiffnessInfluence = 1.0f;
// Optional motion controls for cloth-like flutter effects.
static float g_cmInertiaGain = 0.0f;
static float g_cmVelocityRetention = 0.92f;
static float g_cmVelocityClamp = 0.05f;

static void setGpuCommandMode(bool enabled)
{
	g_gpuCommandMode = enabled;
	if (!enabled) {
		g_cmdIdx.clear();
		g_cmdDelta.clear();
	}
}

static void enqueueGpuCommand(int idx, const glm::vec3& delta)
{
	if (!g_gpuCommandMode) return;
	g_cmdIdx.push_back(idx);
	g_cmdDelta.push_back(delta);
}

static void drainGpuCommands(std::vector<int>& outIdx, std::vector<glm::vec3>& outDelta)
{
	outIdx.swap(g_cmdIdx);
	outDelta.swap(g_cmdDelta);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix,

	float rotX, float rotY, float rotZ


	 )
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);


	//float c = cos(modRot);
	//float s = sin(modRot);



	float rx = glm::radians(rotX);
	float ry = glm::radians(rotY);
	float rz = glm::radians(rotZ);


	//float s = sin(angle_rad / 2.f);
	//float c = cos(angle_rad / 2.f);
	glm::mat3 Rx = glm::mat3(
		1, 0, 0,
		0, cos(rx), -sin(rx),
		0, sin(rx), cos(rx));

	glm::mat3 Ry = glm::mat3(
		cos(ry), 0, sin(ry),
		0, 1, 0,
		-sin(ry), 0, cos(ry));

	glm::mat3 Rz = glm::mat3(
		cos(rz), -sin(rz), 0,
		sin(rz), cos(rz), 0,
		0, 0, 1);

	// 최종 회전 조합 적용
	W = Rz * Ry * Rx * W;

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}
__device__ glm::vec4 createQuatFromAxisAngle(const glm::vec3& axis, float angle)
{
	float half_angle = angle * 0.5f;
	float s = sin(half_angle);
	float r = cos(half_angle);

	return glm::vec4(r, axis.x * s, axis.y * s, axis.z * s);  // (r, x, y, z)
}

__device__ inline void jacobiEigenDecomposition3x3(
	const float A_in[3][3],
	float eigenValues[3],
	float eigenVectors[3][3],
	const int maxIter = 50,
	const float eps = 1e-6f)
{
	// local copy (row-major)
	float A[3][3];
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			A[i][j] = A_in[i][j];

	// initialize eigenVectors to identity
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) eigenVectors[i][j] = (i == j) ? 1.0f : 0.0f;
	}
	float maxOff = 0.0f;
	int actual_iter = 0;
	for (int iter = 0; iter < maxIter; ++iter) {
		actual_iter = iter;
		// find largest off-diagonal |A[p][q]|
		int p = 0, q = 1;
		maxOff = fabsf(A[0][1]);
		if (fabsf(A[0][2]) > maxOff) { p = 0; q = 2; maxOff = fabsf(A[0][2]); }
		if (fabsf(A[1][2]) > maxOff) { p = 1; q = 2; maxOff = fabsf(A[1][2]); }

		if (maxOff < eps) break; // sufficiently diagonal

		// compute Jacobi rotation for indices (p,q) with p < q
		float app = A[p][p];
		float aqq = A[q][q];
		float apq = A[p][q];

		float phi = 0.5f * (aqq - app) / (apq + 1e-20f);
		// tan(2*theta) = 2*apq/(aqq-app)  but we use numerically stable formula
		float t = (phi >= 0.0f) ? (1.0f / (phi + sqrtf(1.0f + phi * phi))) : (1.0f / (phi - sqrtf(1.0f + phi * phi)));
		float c = 1.0f / sqrtf(1.0f + t * t);
		float s = t * c;

		// update A: only rows/cols p,q and off diag other elements
		float app_new = app - t * apq;
		float aqq_new = aqq + t * apq;

		A[p][p] = app_new;
		A[q][q] = aqq_new;
		A[p][q] = A[q][p] = 0.0f;

		// update the other entries
		for (int r = 0; r < 3; ++r) {
			if (r == p || r == q) continue;
			float arp = A[r][p];
			float arq = A[r][q];
			A[r][p] = A[p][r] = c * arp - s * arq;
			A[r][q] = A[q][r] = c * arq + s * arp;
		}

		// update eigenVectors: V = V * J  (apply rotation on columns p,q)
		for (int r = 0; r < 3; ++r) {
			float vrp = eigenVectors[r][p];
			float vrq = eigenVectors[r][q];
			eigenVectors[r][p] = c * vrp - s * vrq;
			eigenVectors[r][q] = s * vrp + c * vrq;
		}
	}
	// after convergence, diagonal entries are eigenvalues
	eigenValues[0] = A[0][0];
	eigenValues[1] = A[1][1];
	eigenValues[2] = A[2][2];

	// sort eigenvalues ascending and reorder eigenvectors accordingly
	// simple bubble sort for 3 elements
	for (int i = 0; i < 2; ++i) {
		for (int j = i + 1; j < 3; ++j) {
			if (eigenValues[i] > eigenValues[j]) {
				float tmp = eigenValues[i];
				eigenValues[i] = eigenValues[j];
				eigenValues[j] = tmp;
				// swap corresponding eigenvector columns
				for (int r = 0; r < 3; ++r) {
					float t = eigenVectors[r][i];
					eigenVectors[r][i] = eigenVectors[r][j];
					eigenVectors[r][j] = t;
				}
			}
		}
	}
}
__device__ inline void eigenDecomposition_glm(
	const glm::mat3& glmA, // 입력(대칭이라고 가정)
	glm::vec3& out_eigenValues,
	glm::mat3& out_eigenVectors) // columns are eigenvectors
{
	float A_row[3][3];
	// convert glm::mat3 (column-major) -> row-major A_row
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			A_row[r][c] = glmA[c][r];
		}
	}

	float ev[3];
	float evecs[3][3];
	jacobiEigenDecomposition3x3(A_row, ev, evecs);

	// write back eigenvalues
	out_eigenValues = glm::vec3(ev[0], ev[1], ev[2]);

	// convert eigenvectors (evecs is row-major where column j is vector)
	// we stored eigenVectors[r][c] as row r, col c -> column c is eigenvector
	for (int c = 0; c < 3; ++c) {
		for (int r = 0; r < 3; ++r) {
			out_eigenVectors[c][r] = evecs[r][c]; // glm::mat3[col][row]
		}
	}
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float scaleModifier, const glm::vec4 rotation, float* covariance)
{
	glm::mat3 scaleMatrix(1.0f);
	scaleMatrix[0][0] = scaleModifier * scale.x;
	scaleMatrix[1][1] = scaleModifier * scale.y;
	scaleMatrix[2][2] = scaleModifier * scale.z;

	const float r = rotation.x;
	const float x = rotation.y;
	const float y = rotation.z;
	const float z = rotation.w;
	const glm::mat3 rotationMatrix(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	const glm::mat3 transform = scaleMatrix * rotationMatrix;
	const glm::mat3 sigma = glm::transpose(transform) * transform;

	covariance[0] = sigma[0][0];
	covariance[1] = sigma[0][1];
	covariance[2] = sigma[0][2];
	covariance[3] = sigma[1][1];
	covariance[4] = sigma[1][2];
	covariance[5] = sigma[2][2];
}

__device__ void computeDeformedCovariance(
	const glm::vec3 scale,
	const glm::mat3& deformationRotation,
	const glm::mat3& deformationStretch,
	float scaleModifier,
	const glm::vec4 rotation,
	float* covariance)
{
	glm::mat3 scaleMatrix(1.0f);
	scaleMatrix[0][0] = scaleModifier * scale.x;
	scaleMatrix[1][1] = scaleModifier * scale.y;
	scaleMatrix[2][2] = scaleModifier * scale.z;

	const float r = rotation.x;
	const float x = rotation.y;
	const float y = rotation.z;
	const float z = rotation.w;
	const glm::mat3 rotationMatrix(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	const glm::mat3 transform = scaleMatrix * rotationMatrix * deformationStretch * glm::transpose(deformationRotation);
	const glm::mat3 sigma = glm::transpose(transform) * transform;

	covariance[0] = sigma[0][0];
	covariance[1] = sigma[0][1];
	covariance[2] = sigma[0][2];
	covariance[3] = sigma[1][1];
	covariance[4] = sigma[1][2];
	covariance[5] = sigma[2][2];
}

// ===== math utils (device/host 공용) =====
__host__ __device__ inline float3 make_f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 load_f3(const float* a, int i) { return make_float3(a[3 * i + 0], a[3 * i + 1], a[3 * i + 2]); }

// ===== 커널 상단 (또는 별도 헤더)에 유틸 =====
struct Mat3 {
	float m[3][3];
	__device__ static Mat3 I() { Mat3 A{}; A.m[0][0] = A.m[1][1] = A.m[2][2] = 1.f; return A; }
};

__device__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float s, const float3& a) { return make_float3(s * a.x, s * a.y, s * a.z); }
__device__ inline float  dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ inline Mat3 outer(const float3& a, const float3& b) {
	Mat3 M{};
	M.m[0][0] = a.x * b.x; M.m[0][1] = a.x * b.y; M.m[0][2] = a.x * b.z;
	M.m[1][0] = a.y * b.x; M.m[1][1] = a.y * b.y; M.m[1][2] = a.y * b.z;
	M.m[2][0] = a.z * b.x; M.m[2][1] = a.z * b.y; M.m[2][2] = a.z * b.z;
	return M;
}
__device__ inline Mat3 add(const Mat3& A, const Mat3& B) {
	Mat3 C{};
#pragma unroll
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) C.m[r][c] = A.m[r][c] + B.m[r][c];
	return C;
}
__device__ inline Mat3 mul(const Mat3& A, const Mat3& B) {
	Mat3 C{};
#pragma unroll
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			C.m[r][c] = A.m[r][0] * B.m[0][c] + A.m[r][1] * B.m[1][c] + A.m[r][2] * B.m[2][c];
		}
	}
	return C;
}
__device__ inline Mat3 trans(const Mat3& A) {
	Mat3 B{};
#pragma unroll
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) B.m[r][c] = A.m[c][r];
	return B;
}
__device__ inline float3 mul(const Mat3& A, const float3& v) {
	return make_float3(
		A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
		A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
		A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z
	);
}

// 3x3 대칭행렬(B=F^T F)의 역제곱근 근사: 뉴턴?슐츠 2회(빠르고 충분히 안정적)
__device__ inline Mat3 inv_sqrt_sym(const Mat3& B) {
	// normalize for stability
	float t = B.m[0][0] + B.m[1][1] + B.m[2][2];
	float s = fmaxf(t / 3.f, 1e-6f);
	Mat3 A{};
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) A.m[r][c] = B.m[r][c] / s;

	Mat3 Y = A;                   // target
	Mat3 X = Mat3::I();           // approx to A^{-1/2}
	const float alpha = 1.5f;
	// two Newton-Schulz iterations
#pragma unroll
	for (int it = 0; it < 2; ++it) {
		Mat3 XYA = mul(X, Y);
		Mat3 AX = mul(Y, X);
		Mat3 M{};
		for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
			M.m[r][c] = 0.5f * (3.f * Mat3::I().m[r][c] - (XYA.m[r][c] + AX.m[r][c]) * 0.5f);
		X = mul(X, M);
		Y = mul(M, Y);
	}
	// scale back
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) X.m[r][c] /= sqrtf(s);
	return X; // ? B^{-1/2}
}

__device__ inline glm::vec4 rotmat_to_quat(const Mat3& R) {
	float trace = R.m[0][0] + R.m[1][1] + R.m[2][2];
	glm::vec4 q;
	if (trace > 0.f) {
		float s = sqrtf(trace + 1.f) * 2.f; // 4w
		q.w = 0.25f * s;
		q.x = (R.m[2][1] - R.m[1][2]) / s;
		q.y = (R.m[0][2] - R.m[2][0]) / s;
		q.z = (R.m[1][0] - R.m[0][1]) / s;
	}
	else {
		int i = (R.m[0][0] < R.m[1][1]) ? ((R.m[1][1] < R.m[2][2]) ? 2 : 1) : ((R.m[0][0] < R.m[2][2]) ? 2 : 0);
		float a = R.m[i][i];
		int j = (i + 1) % 3, k = (i + 2) % 3;
		float s = sqrtf(1.f + a - R.m[j][j] - R.m[k][k]) * 2.f;
		float qarr[4] = { 0,0,0,0 };
		qarr[i] = 0.25f * s;
		q.w = (R.m[k][j] - R.m[j][k]) / s;
		qarr[j] = (R.m[j][i] + R.m[i][j]) / s;
		qarr[k] = (R.m[k][i] + R.m[i][k]) / s;
		q.x = qarr[0]; q.y = qarr[1]; q.z = qarr[2];
	}
	// normalize
	float l = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) + 1e-8f;
	q.x /= l; q.y /= l; q.z /= l; q.w /= l;
	return q;
}
__device__ inline float length3(const float3& v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// ===== 3x3 SVD (Kabsch용) 보조 루틴 =====
__device__ void jacobiEigenSym3(const glm::mat3& A, glm::mat3& V, glm::vec3& eval) {
	glm::mat3 D = A;
	V = glm::mat3(1.0f);

	for (int it = 0; it < 10; ++it) {
		// 최대 절대 비대각 원소 선택
		int p = 0, q = 1;
		float a01 = fabsf(D[1][0]), a02 = fabsf(D[2][0]), a12 = fabsf(D[2][1]);
		if (a02 > a01 && a02 > a12) { p = 0; q = 2; }
		else if (a12 > a01) { p = 1; q = 2; }

		if (fabsf(D[q][p]) < 1e-10f) break;

		float app = D[p][p], aqq = D[q][q], apq = D[q][p];
		float phi = 0.5f * atanf(2.0f * apq / (aqq - app + 1e-20f));
		float c = cosf(phi), s = sinf(phi);

		// 회전 적용
		for (int k = 0; k < 3; ++k) {
			float dkp = D[p][k], dkq = D[q][k];
			D[p][k] = c * dkp - s * dkq;
			D[q][k] = s * dkp + c * dkq;

			float vkp = V[p][k], vkq = V[q][k];
			V[p][k] = c * vkp - s * vkq;
			V[q][k] = s * vkp + c * vkq;
		}
	}

	eval.x = D[0][0]; eval.y = D[1][1]; eval.z = D[2][2];
}

__device__ void svd3x3(const glm::mat3& M, glm::mat3& U, glm::vec3& S, glm::mat3& Vt) {
	glm::mat3 MtM = glm::transpose(M) * M;
	glm::mat3 V; glm::vec3 eval;
	jacobiEigenSym3(MtM, V, eval);

	// 고유값 내림차순 정렬
	int order[3] = { 0,1,2 };
	auto swapi = [&](int a, int b) { int t = order[a]; order[a] = order[b]; order[b] = t; };
	if (eval[order[0]] < eval[order[1]]) swapi(0, 1);
	if (eval[order[0]] < eval[order[2]]) swapi(0, 2);
	if (eval[order[1]] < eval[order[2]]) swapi(1, 2);

	glm::mat3 Vsorted;
	glm::vec3 Sdiag;
	for (int i = 0; i < 3; ++i) {
		int oi = order[i];
		Vsorted[i] = V[oi]; // 열 복사
		Sdiag[i] = sqrtf(fmaxf(eval[oi], 0.0f));
	}

	glm::mat3 VinvSigma(0.0f);
	for (int i = 0; i < 3; ++i) {
		if (Sdiag[i] > 1e-8f) VinvSigma[i][i] = 1.0f / Sdiag[i];
	}

	glm::mat3 Utmp = M * Vsorted * VinvSigma;

	// Gram-Schmidt 정규화
	for (int i = 0; i < 3; ++i) {
		glm::vec3 col(Utmp[i][0], Utmp[i][1], Utmp[i][2]);
		float n = glm::length(col);
		if (n < 1e-8f) { col = glm::vec3(0.0f); col[i] = 1.0f; n = 1.0f; }
		col /= n;
		for (int j = 0; j < 3; ++j) Utmp[i][j] = col[j];
	}

	U = Utmp;
	Vt = glm::transpose(Vsorted);
	S = Sdiag;
}

__device__ inline glm::mat3 quatToMat3(const glm::vec4& rot) {
	float r = rot.x, x = rot.y, y = rot.z, z = rot.w;
	return glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
}

// 3x3 행렬의 역행렬을 계산하는 __device__ 함수
__device__ __inline__ glm::mat3 inverse(const glm::mat3& m) {
	// 행렬식(determinant) 계산
	float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
		m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) +
		m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);

	// 행렬식이 0에 가까우면 (특이 행렬), 단위 행렬 반환
	if (abs(det) < 1e-8f) {
		return glm::mat3(1.0f);
	}

	float inv_det = 1.0f / det;
	glm::mat3 inv;

	// 수반 행렬(Adjugate Matrix)을 사용하여 역행렬 계산
	// GLM은 Column-Major 순서이므로 inv[col][row] 형태로 대입합니다.
	inv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
	inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
	inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
	inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
	inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
	inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
	inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
	inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
	inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

	return inv;
}


// 공분산 갱신 (대칭 3x3 → 6개 요소)

__device__ inline glm::mat3 quat_to_mat3(const glm::vec4& q) {
	float x = q.x, y = q.y, z = q.z, w = q.w;
	float x2 = x * x, y2 = y * y, z2 = z * z;
	float xy = x * y, xz = x * z, yz = y * z;
	float wx = w * x, wy = w * y, wz = w * z;
	return glm::mat3(1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
		2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
		2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2));
}

// 회전 행렬을 쿼터니언으로 변환하는 수치적으로 안정적인 __device__ 함수
__device__ __inline__ glm::vec4 mat3_to_quat(const glm::mat3& m) {
	float t;
	glm::vec4 q;

	// 행렬의 대각합(trace)을 확인하여 가장 안정적인 계산법 선택
	float trace = m[0][0] + m[1][1] + m[2][2];

	if (trace > 0.0f) {
		t = sqrtf(trace + 1.0f) * 2.0f;
		q.w = 0.25f * t;
		q.x = (m[2][1] - m[1][2]) / t;
		q.y = (m[0][2] - m[2][0]) / t;
		q.z = (m[1][0] - m[0][1]) / t;
	}
	else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2])) {
		t = sqrtf(m[0][0] - m[1][1] - m[2][2] + 1.0f) * 2.0f;
		q.x = 0.25f * t;
		q.y = (m[1][0] + m[0][1]) / t;
		q.z = (m[0][2] + m[2][0]) / t;
		q.w = (m[2][1] - m[1][2]) / t;
	}
	else if (m[1][1] > m[2][2]) {
		t = sqrtf(m[1][1] - m[0][0] - m[2][2] + 1.0f) * 2.0f;
		q.y = 0.25f * t;
		q.x = (m[1][0] + m[0][1]) / t;
		q.z = (m[2][1] + m[1][2]) / t;
		q.w = (m[0][2] - m[2][0]) / t;
	}
	else {
		t = sqrtf(m[2][2] - m[0][0] - m[1][1] + 1.0f) * 2.0f;
		q.z = 0.25f * t;
		q.x = (m[0][2] + m[2][0]) / t;
		q.y = (m[2][1] + m[1][2]) / t;
		q.w = (m[1][0] - m[0][1]) / t;
	}

	// 쿼터니언 정규화 (필요시)
	// return glm::normalize(q);
	return q;
}

// 쿼터니언 곱셈을 위한 __device__ 함수
__device__ __inline__ glm::vec4 quat_multiply(const glm::vec4& q1, const glm::vec4& q2) {
	glm::vec4 result;
	result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
	result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
	result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
	result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
	return result;
}
// glm::mat3를 Eigen::Matrix3f로 변환하는 함수
Eigen::Matrix3f glmToEigen(const glm::mat3& m) {
	Eigen::Matrix3f em;
	// glm은 column-major, Eigen도 기본적으로 column-major이므로 바로 복사
	memcpy(em.data(), &m[0][0], 9 * sizeof(float));
	return em;
}

// Eigen::Matrix3f를 glm::mat3로 변환하는 함수
glm::mat3 eigenToGlm(const Eigen::Matrix3f& em) {
	glm::mat3 m;
	memcpy(&m[0][0], em.data(), 9 * sizeof(float));
	return m;
}

// Eigen::Vector3f를 glm::vec3로 변환하는 함수
glm::vec3 eigenToGlm(const Eigen::Vector3f& ev) {
	return glm::vec3(ev.x(), ev.y(), ev.z());
}


/**
 * @brief glm::mat3 행렬의 SVD를 계산합니다.
 * @param F 입력 행렬 (3x3)
 * @param U 출력 행렬 U (3x3)
 * @param Sigma 출력 특이값 벡터 (3x1)
 * @param V 출력 행렬 V (3x3)
 */
__device__ inline void svd(const glm::mat3& F, glm::mat3& U, glm::vec3& Sigma, glm::mat3& V)
{
	// Transpose F, as the algorithm is formulated for row-major matrices
	glm::mat3 Ft = glm::transpose(F);
	glm::mat3 Vt;

	float s[3];

	// The SVD algorithm starts here
	float c, s_c, s_s;
	float c0, c1, c2;
	float s0, s1, s2;

	c0 = Ft[0][0]; c1 = Ft[0][1]; c2 = Ft[0][2];
	s0 = Ft[1][0]; s1 = Ft[1][1]; s2 = Ft[1][2];

	c = c0; s_c = s0;
	if (abs(s_c) < 1.0e-9f) { c0 = 1.f; s0 = 0.f; }
	else { float t = sqrt(c * c + s_c * s_c); c0 = c / t; s0 = s_c / t; }
	c = c1; s_c = s1;
	c1 = c0 * c + s0 * s_c;
	s1 = -s0 * c + c0 * s_c;
	c = c2; s_c = s2;
	c2 = c0 * c + s0 * s_c;
	s2 = -s0 * c + c0 * s_c;

	c = s1; s_c = Ft[2][1];
	if (abs(s_c) < 1.0e-9f) { s1 = 1.f; Ft[2][1] = 0.f; }
	else { float t = sqrt(c * c + s_c * s_c); s1 = c / t; Ft[2][1] = s_c / t; }
	c = s2; s_c = Ft[2][2];
	s2 = s1 * c + Ft[2][1] * s_c;
	Ft[2][2] = -Ft[2][1] * c + s1 * s_c;
	c = c2; s_c = Ft[2][0];
	c2 = s1 * c + Ft[2][1] * s_c;
	Ft[2][0] = -Ft[2][1] * c + s1 * s_c;

	Vt[0][0] = c0; Vt[0][1] = s0; Vt[0][2] = 0.f;
	Vt[1][0] = -s0 * s1; Vt[1][1] = c0 * s1; Vt[1][2] = -Ft[2][1];
	Vt[2][0] = s0 * Ft[2][1]; Vt[2][1] = -c0 * Ft[2][1]; Vt[2][2] = s1;

	s[0] = c1; s[1] = s2; s[2] = Ft[2][0];

	// Initialize U as identity matrix before accumulation
	U = glm::mat3(1.0f);

	for (int i = 0; i < 9; i++)
	{
		c = s[0]; s_c = s[1];
		if (abs(s_c) < 1.0e-9f) { c0 = 1.f; s0 = 0.f; }
		else { float t = sqrt(c * c + s_c * s_c); c0 = c / t; s0 = s_c / t; }
		s[0] = c0 * c + s0 * s_c;
		s[1] = -s0 * c + c0 * s_c;
		c = c2; s_c = s[2];
		c2 = c0 * c + s0 * s_c;
		s[2] = -s0 * c + c0 * s_c;

		glm::mat3 R_mat(glm::vec3(c0, -s0, 0), glm::vec3(s0, c0, 0), glm::vec3(0, 0, 1));
		U = U * R_mat;

		c = s[0]; s_c = c2;
		if (abs(s_c) < 1.0e-9f) { c0 = 1.f; s0 = 0.f; }
		else { float t = sqrt(c * c + s_c * s_c); c0 = c / t; s0 = s_c / t; }
		s[0] = c0 * c + s0 * s_c;
		c2 = -s0 * c + c0 * s_c;
		c = s[1]; s_c = s[2];
		s[1] = c0 * c + s0 * s_c;
		s[2] = -s0 * c + c0 * s_c;

		glm::mat3 R_mat2(glm::vec3(c0, 0, -s0), glm::vec3(0, 1, 0), glm::vec3(s0, 0, c0));
		U = U * R_mat2;
	}

	Sigma = glm::vec3(s[0], s[1], c2);

	V = glm::transpose(Vt);

	// --------------------------------------------------------------------------
	// --- START OF CORRECTION: Ensure proper rotation matrix ---
	// --------------------------------------------------------------------------

	// Make sure Sigma values are positive
	if (Sigma.x < 0) { Sigma.x = -Sigma.x; V[0][0] = -V[0][0]; V[1][0] = -V[1][0]; V[2][0] = -V[2][0]; }
	if (Sigma.y < 0) { Sigma.y = -Sigma.y; V[0][1] = -V[0][1]; V[1][1] = -V[1][1]; V[2][1] = -V[2][1]; }
	if (Sigma.z < 0) { Sigma.z = -Sigma.z; V[0][2] = -V[0][2]; V[1][2] = -V[1][2]; V[2][2] = -V[2][2]; }

	// Check for reflection and correct it
	// R = U * V^T, so we check det(U) and det(V)
	if (glm::determinant(U) * glm::determinant(V) < 0.0f)
	{
		// Invert the sign of the column of U corresponding to the smallest singular value
		// This flips the sign of det(U) while minimally affecting the matrix
		U[0][2] *= -1.0f;
		U[1][2] *= -1.0f;
		U[2][2] *= -1.0f;
	}
	// --------------------------------------------------------------------------
	// --- END OF CORRECTION ---
	// --------------------------------------------------------------------------
}
// device-side Jacobi for symmetric 3x3
// A_in: row-major symmetric matrix (A_in[i][j])
// outputs: eigenValues (ascending), eigenVectors (columns are eigenvectors)

// glm 매트릭스가 column-major이고 접근은 M[col][row] 이므로 변환에 주의.
// 여기서는 A_in_rowmajor[i][j] = glmM[j][i] 로 복사 (행-열 맞춤)

// device 함수: 3x3 역행렬 (adjoint 방식) + 성공 여부 반환
__device__ bool inverse3x3_safe(const glm::mat3& m, glm::mat3& inv, float eps_det = 1e-12f)
{
	float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
	float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
	float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

	float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);

	if (fabs(det) < eps_det) return false;
	float invdet = 1.0f / det;

	inv[0][0] = (a11 * a22 - a12 * a21) * invdet;
	inv[0][1] = -(a01 * a22 - a02 * a21) * invdet;
	inv[0][2] = (a01 * a12 - a02 * a11) * invdet;

	inv[1][0] = -(a10 * a22 - a12 * a20) * invdet;
	inv[1][1] = (a00 * a22 - a02 * a20) * invdet;
	inv[1][2] = -(a00 * a12 - a02 * a10) * invdet;

	inv[2][0] = (a10 * a21 - a11 * a20) * invdet;
	inv[2][1] = -(a00 * a21 - a01 * a20) * invdet;
	inv[2][2] = (a00 * a11 - a01 * a10) * invdet;

	return true;
}
//// 16개의 이웃(Neighbor) 데이터를 한 번에 처리하여 PPt, QPt를 계산하는 커널
//__global__ void compute_deformation_gradient_tensorcore(
//	// ... 인자들 ...
//	int idx // 현재 처리 중인 가우시안 인덱스
//) {
//	// 워프 설정
//	int warpId = threadIdx.x / 32;
//	int laneId = threadIdx.x % 32;
//
//	// 1. Shared Memory에 이웃 데이터 로딩
//	// Matrix P (16x3): 16개 이웃의 원래 위치 상대좌표
//	// Matrix Q (16x3): 16개 이웃의 변형된 위치 상대좌표
//	__shared__ half smem_P[16 * 16];
//	__shared__ half smem_Q[16 * 16];
//	__shared__ float smem_PPt[16 * 16]; // 결과 1
//	__shared__ float smem_QPt[16 * 16]; // 결과 2
//
//	// laneId 0~15가 각각 이웃 0~15번 데이터를 로딩
//	if (laneId < 16) {
//		// nbr_index에서 이웃 가져오기
//		int nbr_idx = nbr_index[idx * MAX_K + laneId]; // MAX_K는 16 이상이어야 함
//
//		glm::vec3 p_vec = original_pos[nbr_idx] - center_original;
//		glm::vec3 q_vec = deformed_pos[nbr_idx] - center_deformed;
//
//		// P 행렬 채우기 (Row Major)
//		smem_P[laneId * 16 + 0] = __float2half(p_vec.x);
//		smem_P[laneId * 16 + 1] = __float2half(p_vec.y);
//		smem_P[laneId * 16 + 2] = __float2half(p_vec.z);
//		// 나머지 0 패딩...
//
//		// Q 행렬 채우기
//		smem_Q[laneId * 16 + 0] = __float2half(q_vec.x);
//		smem_Q[laneId * 16 + 1] = __float2half(q_vec.y);
//		smem_Q[laneId * 16 + 2] = __float2half(q_vec.z);
//	}
//
//	// Matrix P Transpose (PPt 계산을 위해 필요)
//	// 텐서 코어는 A * B^T 지원 (col_major 로딩 시)
//
//	// 2. 텐서 코어 연산 수행
//	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> p_frag; // P Transpose 처럼 동작
//	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> p_frag_b;
//	wmma::fragment<wmma::accumulator, 16, 16, 16, float> ppt_frag;
//
//	wmma::fill_fragment(ppt_frag, 0.0f);
//	wmma::load_matrix_sync(p_frag, smem_P, 16);   // P^T
//	wmma::load_matrix_sync(p_frag_b, smem_P, 16); // P
//
//	// PPt = P^T * P (3x3 결과가 나옴)
//	// 원래는 Outer Product 합인데, 행렬 곱셈 형태로 변환
//	wmma::mma_sync(ppt_frag, p_frag, p_frag_b, ppt_frag);
//
//	// QPt 계산 (Q * P^T)
//	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> q_frag;
//	wmma::fragment<wmma::accumulator, 16, 16, 16, float> qpt_frag;
//
//	wmma::fill_fragment(qpt_frag, 0.0f);
//	wmma::load_matrix_sync(q_frag, smem_Q, 16);
//
//	wmma::mma_sync(qpt_frag, q_frag, p_frag_b, qpt_frag);
//
//	// 3. 결과 저장 및 A 행렬 계산
//	wmma::store_matrix_sync(smem_PPt, ppt_frag, 16, wmma::mem_row_major);
//	wmma::store_matrix_sync(smem_QPt, qpt_frag, 16, wmma::mem_row_major);
//
//	// 이후 스레드 0번이 3x3 역행렬 계산 및 A = QPt * inv(PPt) 수행
//}






template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* realOrigin_points,
	// chainmail 후 (매 프레임 갱신)
	// --- neighborhood for 3D F ---
	const int* nbr_index,    // size: P*MAX_K
	const int* nbr_count,    // size: P
	int MAX_K,
	const float* nbr_time,

	const glm::vec3* scales,
	const float scale_modifier,

	const float _rotatingModifier_COV2D_Matrix_x,
	const float _rotatingModifier_COV2D_Matrix_y,
	const float _rotatingModifier_COV2D_Matrix_z,

	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	bool antialiasing,
	float t,
	bool enableDeformationCovariance
	)
{
	const float* cov3D;

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// 1. 좌표 정보 가져오기 (올바른 매핑)
	float3 p_original = { realOrigin_points[3 * idx], realOrigin_points[3 * idx + 1], realOrigin_points[3 * idx + 2] };  // 변형 전
	float diff = glm::length(glm::vec3(p_orig.x, p_orig.y, p_orig.z) -
		glm::vec3(p_original.x, p_original.y, p_original.z));



	if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
		p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z)
		return;



	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	int kn = nbr_count[idx];
	kn = min(kn, MAX_K);
	const float deformationEpsilon = 1e-2f;





	/////////////////////////////////////////////////////////////////////////////////
	// Estimate a local affine deformation from rest/deformed neighbor offsets.
	glm::mat3 R;
	glm::mat3 S_final;
	glm::vec3 S_vec;
	bool ok = false;
	if (kn < 3)
	{
		// A 계산을 시도하지 않고, 변형이 없는 항등 행렬을 반환
		ok = false;
	}
	else
	{
		glm::mat3 PPt(0.0f);
		glm::mat3 QPt(0.0f);
		//int valid_delta_count = 0;
		glm::vec3 original_pos = glm::vec3(
			realOrigin_points[3 * idx],
			realOrigin_points[3 * idx + 1],
			realOrigin_points[3 * idx + 2]);
		glm::vec3 deformed_pos = glm::vec3(
			orig_points[3 * idx],
			orig_points[3 * idx + 1],
			orig_points[3 * idx + 2]);
		for (int i = 0; i < kn; ++i) {
			int neighbor_idx = nbr_index[idx * MAX_K + i];
			glm::vec3 nbr_original_pos = glm::vec3(
				realOrigin_points[3 * neighbor_idx],
				realOrigin_points[3 * neighbor_idx + 1],
				realOrigin_points[3 * neighbor_idx + 2]);
			glm::vec3 nbr_deformed_pos = glm::vec3(
				orig_points[3 * neighbor_idx],
				orig_points[3 * neighbor_idx + 1],
				orig_points[3 * neighbor_idx + 2]);
			// 그대로 행렬에 삽입

			  // 중심 기준 상대 좌표 (delta 벡터)
			glm::vec3 p = nbr_original_pos - original_pos;
			glm::vec3 q = nbr_deformed_pos - deformed_pos;

			//glm::vec3 p = nbr_original_pos;
			//glm::vec3 q = nbr_deformed_pos;

			// 2. 벡터 변화 확인 (늘어났는지 줄어들었는지)

			PPt += glm::outerProduct(p, p); // P * P^T
			QPt += glm::outerProduct(q, p); // Q * P^T
		}

		//computePPtQPt_ptx(p_list, q_list, kn, PPt, QPt);
		// 안정성을 위해 PPt에 작은 값을 더해줌 (레귤러라이제이션)
		float tracePPt = PPt[0][0] + PPt[1][1] + PPt[2][2];
		// 시스템 에너지의 0.01%~0.1% 정도만 보정값으로 사용합니다.
		// 이렇게 하면 가우시안이 아주 작아도 그에 맞춰 보정값이 작아집니다.
		float adaptive_alpha = (tracePPt > 0.0f) ? (tracePPt * 1e-4f) : 1e-6f;
		PPt += glm::mat3(adaptive_alpha);
		//const float regularization_alpha = 1e-5f;
		//PPt += glm::mat3(regularization_alpha);
		// === [핵심 수정 2] 레귤러라이제이션 값 조정 ===
		//const float active_regularization = (kn < 6) ? 1e-3f : 1e-5f; // 이웃 개수가 적을 때 더 강하게
		//PPt += glm::mat3(active_regularization);

		glm::mat3 invPPt;
		ok = inverse3x3_safe(PPt, invPPt, 1e-8f);
		glm::mat3 A;

		if (ok) {

			A = QPt * invPPt; //glm::inverse(PPt);
			 // 2. A를 SVD를 통해 U, S, V로 분해
			glm::mat3 AtA = glm::transpose(A) * A;
			glm::mat3 V;
			glm::vec3 S_squared;
			eigenDecomposition_glm(AtA, S_squared, V);
			S_vec = glm::sqrt(glm::max(glm::vec3(0.0f), S_squared));


			const float epsilon = 1.0e-7f;
			const float min_s = 1e-3f;
			S_vec = glm::max(S_vec, glm::vec3(min_s));
			const float max_s = 3.0f;
			S_vec = glm::min(S_vec, glm::vec3(max_s));
			glm::vec3 S_inv_vec = glm::vec3(
				(S_vec.x > epsilon) ? 1.0f / S_vec.x : 0.0f,
				(S_vec.y > epsilon) ? 1.0f / S_vec.y : 0.0f,
				(S_vec.z > epsilon) ? 1.0f / S_vec.z : 0.0f);
			//const float min_singular_value = 1e-4f;
			//if (S_vec.x < min_singular_value) S_vec.x = 1.0f;
			//if (S_vec.y < min_singular_value) S_vec.y = 1.0f;
			//if (S_vec.z < min_singular_value) S_vec.z = 1.0f;
			//// --- 여기까지 수정 ---
			//
			//// 이제 안정화된 S_vec으로 S_inv_vec를 계산
			//glm::vec3 S_inv_vec = glm::vec3(1.0f / S_vec.x, 1.0f / S_vec.y, 1.0f / S_vec.z);
			glm::mat3 U =
				A * V *
				glm::mat3(
					glm::vec3(S_inv_vec.x, 0, 0),
					glm::vec3(0, S_inv_vec.y, 0),
					glm::vec3(0, 0, S_inv_vec.z));

			// 3. 순수 회전(R)과 순수 스케일(S_vec) 분리
			R = U * glm::transpose(V);
			if (glm::determinant(R) < 0.0f) {
				U[0] *= -1.0f;
				R = U * glm::transpose(V);
			}
			// R의 행렬식이 음수일 때, 가장 수치적 영향이 적은(가장 작은 특이값) 축을 보정
			//if (glm::determinant(R) < 0.0f) {
			//	int smallest_idx = 0;
			//	if (S_vec[1] < S_vec[smallest_idx]) smallest_idx = 1;
			//	if (S_vec[2] < S_vec[smallest_idx]) smallest_idx = 2;
			//
			//	U[smallest_idx] *= -1.0f; // 가장 작은 축의 부호를 반전
			//	R = U * glm::transpose(V);
			//}

			// 5. '순수 회전'과 '확대만 남은 스케일'로 새로운 변환 행렬 A_final을 재조립
			// A_final = R * S (Polar Decomposition)
			glm::mat3 S_final_diag = glm::mat3(
				glm::vec3(S_vec.x, 0, 0),
				glm::vec3(0, S_vec.y, 0),
				glm::vec3(0, 0, S_vec.z));
			S_final = V * S_final_diag * glm::transpose(V);

		}
		else {
			// fallback: identity 또는 작게 블렌딩
			A = glm::mat3(1.0f);
		}
	}

	if (enableDeformationCovariance && diff > deformationEpsilon && ok) {
		computeDeformedCovariance(
			scales[idx],
			R,
			S_final,
			scale_modifier,
			rotations[idx],
			cov3Ds + idx * 6);
	}
	else {
		// 변형이 거의 없으면 기존 값 유지
		computeCov3D(
			scales[idx],
			scale_modifier,
			rotations[idx],
			cov3Ds + idx * 6);
	}

	cov3D = cov3Ds + idx * 6;

		// If 3D covariance matrix is precomputed, use it, otherwise compute
		// from scaling and rotation parameters.

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix,

		_rotatingModifier_COV2D_Matrix_x,
		_rotatingModifier_COV2D_Matrix_y,
		_rotatingModifier_COV2D_Matrix_z


		);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	//가우시안이 카메라에서 너무 멀어지면 화면상에서 1픽셀보다 작아져 반짝거리는 현상(Aliasing)이 생김.
	//따라서 행렬대각 성분(x, z)에 강제로 0.3 분산더해 최소한의 크기보장(고전적인 EWA Splatting 기법)
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	//투영된 가우시안이 화면에서 얼마나 크게 퍼지는지 최대 반지름my_radius
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	//p_proj = -1.0 ~ 1.0 사이의 정규화된 가우시안 중심 좌표(NDC)
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };//픽셀 해상도(예: 1920x1080)에 맞게 곱해서, 화면 상의 정확한 픽셀 가우시안 좌표point_image로 변환
	uint2 rect_min, rect_max;


	//반지름(my_radius)을 이용해, 화면을 나눈 16X16 픽셀 타일 상 이 가우시안이 최소 몇 번 타일부터 최대 몇 번 타일 덮는지(Bounding Box)를 계산.
	//가우시안중심점(point_image)에서 상하좌우로 반지름(my_radius)만큼 사각형(Bounding Box).
	//그리고 이 사각형이 타일 그리드(Tile Grid) 상에서 어디에 걸치는지 계산.
    //rect_min: 사각형이 걸친 가장 왼쪽 위 타일의 인덱스(ex x방향 10번째, y방향 5번째 타일  x = 10, y = 5)
	//rect_max : 사각형이 걸친 가장 오른쪽 아래 타일의 인덱스(ex x방향 13번째, y방향 8번째 타일  x = 13, y = 8)
	if (rects == nullptr) 	// More conservative
	{
		getRect(point_image, my_radius, rect_min, rect_max, grid);
	}
	else // Slightly more aggressive, might need a math cleanup
	{
		const int2 my_rect = { (int)ceil(3.f * sqrt(cov.x)), (int)ceil(3.f * sqrt(cov.z)) };
		rects[idx] = my_rect;
		getRect(point_image, my_rect, rect_min, rect_max, grid);
	}
	//화면을 아예 벗어났거나 가리는 타일이 0개라면 더 이상 계산할 필요 없이 버림 return
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = final_opacity;// opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);//총 몇 개의 타일을 덮었는지, 가로로 덮은 타일 수: 13 - 10 = 3개 세로로 덮은 타일 수 : 8 - 5 = 3개 총3x3=9개
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list, //타일별로 깊이 정렬된 가우시안 인덱스 리스트
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ id_buffer)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();//현재 블록 가져옴
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;//가로로 타일 총 개수 ex1920 / 16 = 120개의 타일
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };//우리 팀이 맡은 타일의 왼쪽 위(시작) 픽셀 좌표.  (1, 1)번 타일이면, 시작 픽셀은 (16, 16).
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };//타일의 오른쪽 아래(끝) 픽셀 좌표
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };//스레드가 칠해야 할 정확한 픽셀 좌표(x, y)
	uint32_t pix_id = W * pix.y + pix.x;//모니터 화면 전체를 1차원 배열로 쭉 폈을 때, 내 픽셀이 몇 번째 칸에 있는지(1D 인덱스)
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];//이 타일에 가우시안이 총 몇 개나 겹쳐 있는지, 사전(ranges 배열)에서 찾아오는 과정
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);//공유메모리크기 256 이니 가우시안이 그 이상이면 몇번의 rounds를 해야하는지 256개면 1rounds
	int toDo = range.y - range.x;//총 처리해야하는 가우시안 개수

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];//그걸 배치 단위로 collected_id[BLOCK_SIZE]에 가져와서 쓰는 것
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	int picked_id = -1;//각 픽셀에서 -1로 시작

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)//총 rounds 만큼반복 한 rounds에 256 개 가우시안 처리
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();//공유메모리에 올림 256 개

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)//256개 or 그보다 적은 가우시안이라면 가우시안 개수만큼 반복
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power =
				-0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
				- con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			if (id_buffer != nullptr && picked_id == -1 && alpha > 0.1f)
			{
				//front to back 순서의 가우시안을 돌며 0.1 보다 alpha 가 큰 첫 가우시안 id 를 저장
				picked_id = collected_id[j];//그리고 내부 루프 j는 이번 배치의 j번째 가우시안을 의미함
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		if (id_buffer != nullptr && picked_id != -1)
		{
			id_buffer[pix_id] = picked_id;//pix_id 는 픽셀인덱스 즉 몇번픽셀에 몇번가우시안인지 담기.
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* id_buffer)
{
	//renderCUDA는 한 타일(tile)에 해당하는 가우시안 목록을 shared memory로 배치해서 처리
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,//타일별로 깊이 정렬된 가우시안 인덱스 리스트
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		id_buffer);
}
bool FORWARD::ChainMail::loadGraph(
	ChainMail& cm,
	const std::vector<Pos>& positions,
	const std::vector<Edge>& edges,
	const std::vector<float>& opacities
) {

	// Initialize solver elements from Gaussian positions.
	cm.elements.clear();
	cm.elements.resize(positions.size());
	for (int i = 0; i < positions.size(); ++i) {
		cm.elements[i].pos = positions[i];
		cm.elements[i].vel = glm::vec3(0.0f);
		cm.elements[i].invMass = 1.0f;
		cm.elements[i].density = opacities[i];     // 기본값
		cm.elements[i].time = 1e9f;
		cm.elements[i].offset = 0;
		cm.elements[i].neighborCnt = 0;
	}

	// Convert graph edges to adjacency lists.
	cm.cedges.clear();
	cm.cedges.reserve(edges.size());
	std::vector<std::vector<Neighbor>> tempNeigh(positions.size());
	for (const auto& e : edges) {
		const int a = e.m_vert[0];
		const int b = e.m_vert[1];
		if (a < 0 || b < 0 || a >= (int)positions.size() || b >= (int)positions.size() || a == b) {
			continue;
		}
		tempNeigh[a].emplace_back(b, e.rl, e.st);
		tempNeigh[b].emplace_back(a, e.rl, e.st);
		FORWARD::cEdge ce;
		ce.v1 = a;
		ce.v2 = b;
		ce.dist = e.rl;
		cm.cedges.push_back(ce);
	}

	// Flatten adjacency lists for the solver.
	cm.neighbors.clear();
	int idx = 0, offset = 0;
	for (int i = 0; i < positions.size(); ++i) {
		auto& elem = cm.elements[i];
		elem.offset = offset;
		elem.neighborCnt = static_cast<int>(tempNeigh[i].size());

		for (auto& n : tempNeigh[i]) {
			cm.neighbors.push_back(n);
			idx++;
		}
		offset = idx;
	}
	return true;
}
FORWARD::CMConstraint h_AIR(
	0.1f, 0.1f, 0.1f,   // dx, dy, dz
	0.1f, 0.1f,         // xShearY, xShearZ
	0.1f, 0.1f,         // yShearX, yShearZ
	0.1f, 0.1f          // zShearX, zShearY
);

// SKIN
FORWARD::CMConstraint h_SKIN(
	0.01f, 0.01f, 0.01f,
	0.01f, 0.01f,
	0.01f, 0.01f,
	0.01f, 0.01f
);

// BONE
FORWARD::CMConstraint h_BONE(
	0.0001f, 0.0001f, 0.0001f,
	0.0001f, 0.0001f,
	0.0001f, 0.0001f,
	0.0001f, 0.0001f
);




void FORWARD::ChainMail::resetTime() {
	for (auto& e : elements)
		e.time = 1e9f;
}


void FORWARD::ChainMail::movePointPos(int* idx, const glm::vec3& dpos, std::vector<int>& activeSet) {
	if (g_gpuCommandMode) {
		// Per-frame drag calls can be very frequent; avoid log spam here.
		enqueueGpuCommand(*idx, dpos);
		activeSet.push_back(*idx);
		return;
	}
	// Per-frame drag calls can be very frequent; avoid log spam here.
	elements[*idx].pos = elements[*idx].pos + dpos;
	elements[*idx].time = 0.0f;
	activeSet.push_back(*idx);
}
/**
 * @brief 시작점으로부터 그래프 연결성을 따라 n개의 인접 정점을 수집함
 * @param startNode 마우스 클릭 등으로 선택된 시작 가우시안 인덱스
 * @param targetCount 수집할 총 정점 개수 (예: 50개)
 * @return 수집된 인덱스들의 벡터
 */
#include <vector>
#include <queue>
#include <algorithm>
std::vector<int> FORWARD::ChainMail::collectSeedsBFS(int startNode, int targetCount) {
	std::vector<int> selectedSeeds;
	if (startNode < 0 || startNode >= elements.size()) return selectedSeeds;

	std::queue<int> q;
	std::vector<bool> visited(elements.size(), false);

	// 시작점 설정
	q.push(startNode);
	visited[startNode] = true;

	while (!q.empty() && selectedSeeds.size() < targetCount) {
		int curr = q.front();
		q.pop();

		selectedSeeds.push_back(curr);

		// 현재 노드의 이웃들을 탐색
		const Element& elem = elements[curr];
		for (int i = 0; i < elem.neighborCnt; ++i) {
			int neighborIdx = neighbors[elem.offset + i].idx;

			// 아직 방문하지 않은 이웃만 큐에 삽입
			if (neighborIdx >= 0 && !visited[neighborIdx]) {
				visited[neighborIdx] = true;
				q.push(neighborIdx);
			}

			// 목표 개수를 채우면 즉시 중단
			if (selectedSeeds.size() + q.size() > targetCount * 2) {
				// 큐가 너무 커지는 것을 방지하기 위한 조기 종료 로직 (선택 사항)
			}
		}
	}

	return selectedSeeds;
}
// 여러 점을 동시에 시작점으로 설정하는 방식
void FORWARD::ChainMail::startWaveMultiple(const std::vector<int>& seeds, glm::vec3 delta, std::vector<int>& activeSet)
{
	if (g_gpuCommandMode) {
		printf("g_gpuCommandMode movePointIdx : %d  moveDelta : %f , %f , %f\n", seeds.size(), delta.x, delta.y, delta.z);
		for (int sIdx : seeds) {
			enqueueGpuCommand(sIdx, delta);
			activeSet.push_back(sIdx);
		}
		return;
	}
	else {
		printf("cpuCommandMode movePointIdx : %d  moveDelta : %f , %f , %f\n", seeds.size(), delta.x, delta.y, delta.z);
		for (int sIdx : seeds) {
			elements[sIdx].pos += delta; // 50개 노드를 동시에 delta만큼 이동
			elements[sIdx].time = std::min(elements[sIdx].time, 0.0f);

			//elements[sIdx].time = 0.0f;
			activeSet.push_back(sIdx);
		}
	}



}
void FORWARD::ChainMail::startWavingMultiple(
	const std::vector<int>& seeds,
	glm::vec3 delta,
	std::vector<int>& activeSet)
{
	waveActive.clear();

	int N = seeds.size();

	for (int i = 0; i < N; ++i) {
		int idx = seeds[i];

		//float weight = 1.0f - (i / float(N));  // 여기
		//elements[idx].pos += delta * weight;

		elements[idx].time = 0.0f;
		activeSet.push_back(idx);
		waveActive.push_back(idx);
	}

	waveRunning = true;
}
void FORWARD::ChainMail::startWave(int seed, glm::vec3 delta, std::vector<int>& activeSet)
{
	if (g_gpuCommandMode) {
		enqueueGpuCommand(seed, delta);
		activeSet.push_back(seed);
		waveActive.clear();
		waveActive.push_back(seed);
		return;
	}
	elements[seed].pos += delta;
	elements[seed].time = 0.0f;
	activeSet.push_back(seed);
	waveActive.clear();
	waveActive.push_back(seed);
	//waveRunning = true;
}
void FORWARD::ChainMail::applyWaveOffset(
	const glm::vec3& delta,
	std::vector<int>& activeSet)
{
	if (g_gpuCommandMode) {
		for (int idx : seeds) {
			enqueueGpuCommand(idx, delta);
			activeSet.push_back(idx);
		}
		return;
	}
	for (int idx : seeds) {
		elements[idx].pos += delta;   // 위치만 이동
		activeSet.push_back(idx);     // relax 대상
	}
}


void FORWARD::ChainMail::propagateStep(const std::vector<int>& currentFrontier,std::vector<int>& nextFrontier, std::vector<int>& totalActiveSet)
{
	// static 변수 제거! (매 호출마다 상태가 초기화되어야 함)

	// 이번 단계에서 처리할 노드들에 대해 방문 표시가 필요하다면
	// ChainMail 특성상 time 비교를 하므로 별도 visited 배열이 없어도 되지만,
	// 한 프레임 내 중복 방지를 위해 로컬 visited를 쓰거나 time 체크를 철저히 해야 합니다.

	for (int idx : currentFrontier) {
		Element& elem = elements[idx];

		for (int ni = 0; ni < elem.neighborCnt; ++ni) {
			const Neighbor& neigh = neighbors[elem.offset + ni];
			int nIdx = neigh.idx;
			float dist = neigh.dist;

			Element& neighbor = elements[nIdx];

			float newTime = elem.time + propagationTime(elem, neighbor);

			// 더 빠른 경로(더 강력한 당김)가 발견되면 업데이트
			if (neighbor.time > newTime) {
				neighbor.time = newTime;

				bool moved = false;
				shiftElementPoint(neighbor, elem, dist, moved);

				if (moved) {
					// 이번에 움직였으면 다음 단계에서 얘의 이웃도 검사해야 함
					nextFrontier.push_back(nIdx);

					// Relax를 위해 전체 목록에도 추가
					totalActiveSet.push_back(nIdx);
				}
			}
		}
	}

	// 중복 제거 (선택 사항이나 성능을 위해 권장)
	if (!nextFrontier.empty()) {
		std::sort(nextFrontier.begin(), nextFrontier.end());
		nextFrontier.erase(std::unique(nextFrontier.begin(), nextFrontier.end()), nextFrontier.end());
	}
}

void FORWARD::ChainMail::propagate(std::vector<int>& activeSet)
{
	size_t N = elements.size();

	// --- persistent wave state ---
	static std::vector<int> active;
	static bool initialized = false;

	// 첫 호출 시 seed 초기화
	if (!initialized) {
		active.clear();
		for (size_t i = 0; i < N; ++i)
			if (elements[i].time == 0.0f)
				active.push_back((int)i);
		initialized = true;
	}

	if (active.empty()) {
		initialized = false; // wave 종료 다음 클릭 때 재시작
		return;
	}

	std::vector<int> nextActive;
	std::vector<bool> visited(N, false);

	for (int idx : active)
		visited[idx] = true;

	//  여기서 1 iteration만 수행
	for (int idx : active) {
		Element& elem = elements[idx];

		for (int ni = 0; ni < elem.neighborCnt; ++ni) {
			const Neighbor& neigh = neighbors[elem.offset + ni];
			int nIdx = neigh.idx;
			float dist = neigh.dist;

			Element& neighbor = elements[nIdx];

			float newTime = elem.time + propagationTime(elem, neighbor);

			if (neighbor.time > newTime) {
				neighbor.time = newTime;

				bool moved = false;
				shiftElementPoint(neighbor, elem, dist, moved);

				if (moved) {
					activeSet.push_back(nIdx);
					//printf("propagate activeSet size (실제 움직임):%d\n ", activeSet.size());
				}

				if (!visited[nIdx]) {
					nextActive.push_back(nIdx);
					visited[nIdx] = true;
				}
			}

		}
	}

	active = std::move(nextActive);


}


// //변화(전파) wave가 실제로 도달한 정점 인덱스만 activeSet에 기록 (relax에 활용)
//void FORWARD::ChainMail::propagate(std::vector<int>& activeSet) {
//	size_t N = elements.size();//전체정점 개수
//	std::vector<int> active;// 활성화 된 정점들
//	//activeSet.clear();//결과 배열
//	std::vector<bool> propagated(N, false);  // relax용: propagate로 전파된(방문된) 정점 여부
//	std::vector<bool> movedFlag(N, false);     // 실제 위치 이동이 발생한 정점 기록
//
//	// 초기 활성화: 타임스탬프 0인 정점
//	for (size_t i = 0; i < N; ++i)
//		if (elements[i].time == 0.0f)//초기에 활성화된(변형의 시작점찾기) 타임스탬스0인점
//			active.push_back(static_cast<int>(i));//활성화 목록에 추가
//
//	int iter = 0;
//	// propagation 반복
//	while (!active.empty()) {//활성화 목록에서 모두 처리가끝나면 전파끝
//		std::vector<int> nextActive;//다음 iter 에서 활성화 할 정점들
//		std::vector<bool> visited(N, false); // 같은 iter 에서 중복 생성 방지
//
//		for (int idx : active)
//			visited[idx] = true;
//
//		// #pragma omp parallel for (병렬화시 활성화)
//		for (size_t aidx = 0; aidx < active.size(); ++aidx) {
//			int idx = active[aidx];
//			Element& elem = elements[idx];
//			int off = elem.offset, nCnt = elem.neighborCnt;
//
//			for (int ni = 0; ni < nCnt; ++ni) {
//				const Neighbor& neigh = neighbors[off + ni];
//				int nIdx = neigh.idx;
//				float dist = neigh.dist;
//				//float st = neigh.st;
//
//				Element& neighbor = elements[nIdx];
//
//				// propagation time, 반드시 '내 time + link시간'!
//				float newTime = elem.time + propagationTime(elem, neighbor);
//
//				// 이웃이 현재보다 더 빠른 경로로 도달하면 update
//				if (neighbor.time > newTime) {
//					neighbor.time = newTime;
//
//					bool moved = false;
//					shiftElementPoint(neighbor, elem, dist, moved); // neighbor 이동
//					propagated[nIdx] = true;   // t와 pos 모두 변화
//
//					// activeSet 등록(실제 변화 발생)
//					if (moved) {
//						movedFlag[nIdx] = true;
//					}                    // 활성화 리스트(중복 방지)
//					if (!visited[nIdx]) {
//						nextActive.push_back(nIdx);
//						visited[nIdx] = true;
//					}
//				}
//			}
//		}
//		active = std::move(nextActive);
//		iter++;
//		//std::cout << "Iteration " << iter << ", Active Count: " << active.size() << std::endl;
//	}
//
//	// propagate wave가 실제로 도달해 변화가 일어난 모든 정점의 index를 activeSet에 기록
//	for (size_t i = 0; i < N; ++i)
//		if (propagated[i] && movedFlag[i])
//			activeSet.push_back(static_cast<int>(i));
//	printf("propagate activeSet size (실제 움직임):%d\n ", activeSet.size());
//	//std::cout << "propagate activeSet size (실제 움직임): " << activeSet.size() << std::endl;
//
//}
float FORWARD::ChainMail::propagationTime(const Element& e, const Element& neighbor) {
	// density가 0~1 범위일 때 예시
	float et, nt;
	if (e.density < AIR)        et = 1.0f;        // soft (air)
	else if (e.density < SKIN)   et = 0.3f;        // mid (skin)
	else if (e.density < BONE)  et = 0.05f;       // stiff (bone 등)
	else et = 0.005;

	if (neighbor.density < AIR)      nt = 1.0f;
	else if (neighbor.density < SKIN) nt = 0.3f;
	else if (neighbor.density < BONE) nt = 0.05f;       // stiff (bone 등)
	else nt = 0.005;
	return (et + nt) * 0.5f;

	// 또는 아주 단순하게
	// return 1.0f;   // wave 속도 고정 (실제 변화는 제약값에서 발생)
}
FORWARD::CMConstraint FORWARD::ChainMail::getConstraint(float density) {
	// 0(skin/air) ~ 1(bone)
	float axisC;   // 축 방향 강성
	float shearC;  // 전단 방향 강성

	if (density < FORWARD::AIR) {           // Air
		axisC =  0.3f;
		shearC = 0.3f;
	}
	else if (density < FORWARD::SKIN) {     // Skin
		axisC = 0.06f;
		shearC = 0.06f;// 0.06f;
	}
	else {                         // Bone
		axisC =  0.04f;
		shearC =  0.04f;
	}

	return FORWARD::CMConstraint(
		axisC, axisC, axisC,   // dx, dy, dz
		shearC, shearC,        // xShearY, xShearZ
		shearC, shearC,        // yShearX, yShearZ
		shearC, shearC         // zShearX, zShearY
	);
}


void FORWARD::ChainMail::shiftElementPoint(Element& elem, const Element& n, float targDist,  bool& moved) {
	// 간단화: density별로 constraint 하드코딩 적용(실제값은 현업코드 참조)
	CMConstraint nConstraint = getConstraint(n.density);
	// st=1이면 매우 쫀쫀(허용오차 작게), st=0이면 느슨(허용오차 크게)


	//glm::vec3 dir = n.pos - elem.pos;
	//float len = dir.length();
	//glm::vec3 nDir = norm(dir);// (len > 1e-6f) ? glm::normalize(dir) : glm::vec3();
	//
	//Vec3 dir(n.pos.x - elem.pos.x, n.pos.y - elem.pos.y, n.pos.z - elem.pos.z);
	glm::vec3 Dir = n.pos - elem.pos;
	Vec3 dir(Dir.x, Dir.y, Dir.z);
	float len = dir.length();
	glm::vec3 nDir = glm::normalize(Dir);

	float delta = 0.0f;
	float alpha = 1.0f; // 0 < alpha <= 1.0, 낮을수록 부드럽게
	if (len < targDist - nConstraint.dx) {
		// 너무 가까움 멀어져야 함 -nDir 방향
		delta = (targDist - nConstraint.dx) - len;
		elem.pos = elem.pos - nDir * delta;
		moved = true;
	}
	else if (len > targDist + nConstraint.dx) {
		// 너무 멀음  가까워져야 함 +nDir 방향
		delta = len - (targDist + nConstraint.dx);
		elem.pos = elem.pos + nDir * delta;
		moved = true;
	}
	else {
		moved = false;
	}


	//if (len < targDist - nConstraint.dx) {//제약 범위보다 가깝거나
	//    delta = (targDist - nConstraint.dx) - len;//제약범위내에서 가능한 가깝게 설정
	//}
	//else if (len > targDist + nConstraint.dx) {//제약범위보다 멀면
	//    delta = (targDist + nConstraint.dx) - len;//거리 제약범위 내에서 최대거리로 설정
	//}

	//// 한 번에 다 보내지 않고 조금씩 따라오게!
	//if (std::abs(delta) > 1e-6f) {
	//    float alpha = 1.0f; // 0 < alpha <= 1.0, 낮을수록 부드럽게
	//    elem.pos = elem.pos + nDir * (delta * alpha);
	//    moved = true;
	//}
	//else {
	//    moved = false;
	//}
}


const FORWARD::Element& FORWARD::ChainMail::getElement(int i) const {
	return elements[i];
}

FORWARD::Element& FORWARD::ChainMail::getElement(int i) {
	return elements[i];
}
const FORWARD::Neighbor& FORWARD::ChainMail::getNeighbor(int i) const {
	return neighbors[i];
}

FORWARD::Neighbor& FORWARD::ChainMail::getNeighbor(int i) {
	return neighbors[i];
}

const std::vector<FORWARD::cEdge>& FORWARD::ChainMail::getEdges() const
{
	return cedges;
}
void FORWARD::ChainMail::B_relax(const std::vector<int>& activeSet) {
	std::vector<glm::vec3> newPos(elements.size());

	float a = 0.01f;
	float b = 0.01f;
	for (int idx : activeSet) {
		Element& e = elements[idx];

		// 자기 밀도에 따른 제약값 (getConstraint 대체)
		CMConstraint eConstraint;
		if (e.density < AIR) {          // AIR
			eConstraint.dx = 0.8f;  eConstraint.dy = 0.8f;  eConstraint.dz = 0.8f;
			eConstraint.xShearY = eConstraint.xShearZ = 0.8f;
			eConstraint.yShearX = eConstraint.yShearZ = 0.8f;
			eConstraint.zShearX = eConstraint.zShearY = 0.8f;
		}
		else if (e.density < SKIN) {    // SKIN
			eConstraint.dx = a;  eConstraint.dy = a;  eConstraint.dz = a;
			eConstraint.xShearY = eConstraint.xShearZ = a;
			eConstraint.yShearX = eConstraint.yShearZ = a;
			eConstraint.zShearX = eConstraint.zShearY = a;
		}
		else {                          // BONE
			eConstraint.dx = b;  eConstraint.dy = b;  eConstraint.dz = b;
			eConstraint.xShearY = eConstraint.xShearZ = b;
			eConstraint.yShearX = eConstraint.yShearZ = b;
			eConstraint.zShearX = eConstraint.zShearY = b;
		}

		glm::vec3 sumPos(0, 0, 0);
		glm::vec3 totalWeight(0, 0, 0);
		int nCnt = 0;

		// 6방향 이웃 처리
		for (int j = 0; j < e.neighborCnt; ++j) {
			const Neighbor& n = neighbors[e.offset + j];
			const Element& nb = elements[n.idx];
			float targetDist = n.dist;

			// 이웃 밀도에 따른 제약값
			CMConstraint nConstraint;
			if (nb.density < AIR) {
				nConstraint.dx = 0.8f;  nConstraint.dy = 0.8f;  nConstraint.dz = 0.8f;
				nConstraint.xShearY = nConstraint.xShearZ = 0.8f;
				nConstraint.yShearX = nConstraint.yShearZ = 0.8f;
				nConstraint.zShearX = nConstraint.zShearY = 0.8f;
			}
			else if (nb.density < SKIN) {
				nConstraint.dx = a;  nConstraint.dy = a;  nConstraint.dz = a;
				nConstraint.xShearY = nConstraint.xShearZ = a;
				nConstraint.yShearX = nConstraint.yShearZ = a;
				nConstraint.zShearX = nConstraint.zShearY = a;
			}
			else {
				nConstraint.dx = b;  nConstraint.dy = b;  nConstraint.dz = b;
				nConstraint.xShearY = nConstraint.xShearZ = b;
				nConstraint.yShearX = nConstraint.yShearZ = b;
				nConstraint.zShearX = nConstraint.zShearY = b;
			}

			// 방향별 가중치 계산 (간단화)
			float wX = 1.0f / ((eConstraint.dx + nConstraint.dx+targetDist) * 0.5f + 1e-6f);
			float wY = 1.0f / ((eConstraint.dy + nConstraint.dy+targetDist) * 0.5f + 1e-6f);
			float wZ = 1.0f / ((eConstraint.dz + nConstraint.dz+targetDist) * 0.5f + 1e-6f);

			sumPos.x += nb.pos.x * wX;
			sumPos.y += nb.pos.y * wY;
			sumPos.z += nb.pos.z * wZ;

			totalWeight.x += wX;
			totalWeight.y += wY;
			totalWeight.z += wZ;

			nCnt++;
		}

		if (nCnt > 0) {
			glm::vec3 newE;
			newE.x = sumPos.x / totalWeight.x;
			newE.y = sumPos.y / totalWeight.y;
			newE.z = sumPos.z / totalWeight.z;
			newPos[idx] = newE;
		}
		else {
			newPos[idx] = e.pos;
		}
	}

	// 위치 갱신
	for (int idx : activeSet) {
		elements[idx].pos = newPos[idx];
	}
}

void FORWARD::ChainMail::Stabilize(const std::vector<int>& activeSet) {
	// 거리 제약 기반 위치 보정
	for (int idx : activeSet) {
		Element& e = elements[idx];

		for (int j = 0; j < e.neighborCnt; ++j) {
			Neighbor& n = neighbors[e.offset + j];
			Element& nb = elements[n.idx];

			glm::vec3 delta = e.pos - nb.pos;
			float dist = delta.length();
			if (dist < 1e-6f) continue;

			float diff = dist - n.dist; // targetDist와 현재 거리 차이
			// correction 비율 (0.5 → 양쪽 절반씩 이동)
			glm::vec3 correction = (diff / dist) * 0.5f * delta;

			// 밀도/제약 고려해서 움직임 크기 제한 가능
			e.pos -= correction;
			nb.pos += correction;
		}
	}
}
void FORWARD::ChainMail::relax2(const std::vector<int>& activeSet) {
	// [변경 1] newPos 벡터 제거 (메모리 절약 + 속도 향상)
	// std::vector<glm::vec3> newPos(elements.size());

	float stiffness = 0.9f; // 원하시는 대로 유지

	for (int idx : activeSet) {
		Element& e = elements[idx];
		CMConstraint eConstraint = getConstraint(e.density);

		glm::vec3 accumCorrection(0.0f);
		int corrCount = 0;
		float sumTargetDist = 0.0f;
		int distCount = 0;

		for (int j = 0; j < e.neighborCnt; ++j) {
			const Neighbor& n = neighbors[e.offset + j];

			// [핵심] n.idx의 위치를 가져올 때, 앞서 계산된 이웃이라면
			// 이미 보정된 '최신 위치'를 가져오게 되어 전파 속도가 2배 이상 빨라집니다.
			const Element& nb = elements[n.idx];
			CMConstraint nConstraint = getConstraint(nb.density);

			float targDist = n.dist;
			float tol = 0.1f * (eConstraint.dx + nConstraint.dx) * 0.02f;

			glm::vec3 dir = nb.pos - e.pos;
			float len = glm::length(dir);
			if (len < 1e-6f) continue;

			glm::vec3 nDir = dir / len;

			if (len < targDist - tol) { // 너무 가까움 (압축)
				float delta = (targDist - tol) - len;
				accumCorrection -= nDir * delta;
				corrCount++;
			}
			else if (len > targDist + tol) { // 너무 멈 (인장)
				float delta = len - (targDist + tol);
				accumCorrection += nDir * delta;
				corrCount++;
			}

			sumTargetDist += targDist;
			distCount++;
		}

		if (corrCount > 0) {
			glm::vec3 avgCorr = accumCorrection / float(corrCount);

			// --- [변경 2] 속도 제한(Limit) 완화 ---
			// 기존 1.5배는 돌아오려는 움직임을 너무 강하게 브레이크 잡습니다.
			// 이걸 4.0배 정도로 늘려주면 안정성은 유지하되 '확' 돌아옵니다.
			float maxStep = 0.8f;
			if (distCount > 0) {
				float avgTarget = sumTargetDist / float(distCount);
				const float maxStepFactor = 8.0f; // 기존 1.5f -> 4.0f로 증가 (이게 속도의 핵심)
				maxStep = maxStepFactor * avgTarget;
			}

			// 너무 큰 보정은 잘라낸다 (폭발 방지용 안전장치)
			float corrLen = glm::length(avgCorr);
			if (corrLen > 1e-6f) {
				if (corrLen > maxStep) {
					avgCorr = avgCorr * (maxStep / corrLen);
				}

				// [변경 3] 즉시 적용 (Gauss-Seidel)
				// newPos에 넣지 않고 내 위치를 바로 바꿉니다.
				e.pos += stiffness * avgCorr;
			}
		}
	}

	// [변경 4] 마지막 복사 루프 제거 (위에서 이미 적용함)
	// for (int idx : activeSet) elements[idx].pos = newPos[idx];
}
void FORWARD::ChainMail::relax(const std::vector<int>& activeSet) {
	std::vector<glm::vec3> newPos(elements.size());
	float stiffness = 0.8f; // 권장 범위 0.2 ~ 0.5

	for (int idx : activeSet) {
		Element& e = elements[idx];
		CMConstraint eConstraint = getConstraint(e.density);

		glm::vec3 accumCorrection(0.0f);
		int corrCount = 0;
		float sumTargetDist = 0.0f;
		int distCount = 0;

		for (int j = 0; j < e.neighborCnt; ++j) {
			const Neighbor& n = neighbors[e.offset + j];
			const Element& nb = elements[n.idx];
			CMConstraint nConstraint = getConstraint(nb.density);

			float targDist = n.dist;
			float tol = 0.1*(eConstraint.dx + nConstraint.dx) * 0.02f;

			glm::vec3 dir = nb.pos - e.pos;
			float len = glm::length(dir);
			if (len < 1e-6f) continue;

			glm::vec3 nDir = dir / len;

			if (len < targDist - tol) {//너무 가까울시 이웃반대방향으로 초과된범위만큼 멀어짐
				float delta = (targDist - tol) - len;
				accumCorrection -= nDir * delta;
				corrCount++;
			}
			else if (len > targDist + tol) {//너무 멀때 이웃방향으로 제약초과범위만큼 가까워짐
				float delta = len - (targDist + tol);
				accumCorrection += nDir * delta;
				corrCount++;
			}

			// 로컬 평균 목표거리 수집
			sumTargetDist += targDist;
			distCount++;
		}

		if (corrCount > 0) {
			glm::vec3 avgCorr = accumCorrection / float(corrCount);

			// --- 로컬 maxStep 계산 ---
			float maxStep = 0.6f; // 기본 fallback
			if (distCount > 0) {
				float avgTarget = sumTargetDist / float(distCount);
				const float maxStepFactor = 1.5f; // avgTarget의 몇 배를 최대 스텝으로 허용할지
				maxStep = maxStepFactor * avgTarget;
			}

			// 너무 큰 보정은 잘라낸다
			float corrLen = glm::length(avgCorr);
			if (corrLen > 1e-6f && corrLen > maxStep) {
				avgCorr = avgCorr * (maxStep / corrLen);
			}

			glm::vec3 corrected = e.pos + stiffness * avgCorr;
			newPos[idx] = corrected;
		}
		else {
			newPos[idx] = e.pos;
		}
	}

	for (int idx : activeSet) elements[idx].pos = newPos[idx];
}


size_t FORWARD::ChainMail::numElements() const

{
	return elements.size();
}
// ===============================
// GPU persistent chainmail state
// ===============================
static float3* d_pos_curr = nullptr;
static float3* d_pos_next = nullptr;
static float3* d_pos_prev = nullptr;
static float3* d_vel = nullptr;
static float* d_density = nullptr;
static float* d_invMass = nullptr;
static float* d_time_curr = nullptr;
static float* d_time_next = nullptr;
static int* d_offset = nullptr;
static int* d_nbrCount = nullptr;
static int* d_nbrIdx = nullptr;
static float* d_nbrDist = nullptr;
static float* d_nbrStiff = nullptr;
static int* d_active_map = nullptr;
static int* d_next_map = nullptr;
static int* d_active_count = nullptr;
static int* d_best_from = nullptr;
// Latest deformed positions used for rendering (xyz packed, 3*P floats).
// Updated every frame in preprocess().
static float* g_latest_deformed_xyz = nullptr;
static int g_latest_deformed_count = 0;
// Latest screen-space projection buffers used by render path.
static float2* g_latest_means2d = nullptr;
static int* g_latest_radii = nullptr;
static int g_latest_project_count = 0;
static int g_latest_render_w = 0;
static int g_latest_render_h = 0;
static int cm_num_elements = 0;
static int cm_num_neighbors = 0;
static bool chainmailInit = false;

void FORWARD::setPickingParams(int hops)
{
	bfsHops = (hops < 0) ? 0 : hops;

}

void FORWARD::getPickingParams(int* hops)
{
	if (hops) *hops = bfsHops;

}

bool FORWARD::copyCurrentDeformedPositions(float* outXYZ, int pointCount)
{
	if (!outXYZ || pointCount <= 0 || !g_latest_deformed_xyz || g_latest_deformed_count <= 0) {
		return false;
	}
	const int copyCount = (pointCount < g_latest_deformed_count) ? pointCount : g_latest_deformed_count;
	cudaMemcpy(outXYZ, g_latest_deformed_xyz, sizeof(float) * 3 * copyCount, cudaMemcpyDeviceToHost);
	return true;
}

bool FORWARD::copyCurrentScreenProjection(
	float* outXY,
	int* outRadii,
	int pointCount,
	int* outRenderW,
	int* outRenderH)
{
	if (!outXY || pointCount <= 0 || !g_latest_means2d || g_latest_project_count <= 0) {
		return false;
	}
	const int copyCount = (pointCount < g_latest_project_count) ? pointCount : g_latest_project_count;
	cudaMemcpy(outXY, g_latest_means2d, sizeof(float2) * copyCount, cudaMemcpyDeviceToHost);
	if (outRadii && g_latest_radii) {
		cudaMemcpy(outRadii, g_latest_radii, sizeof(int) * copyCount, cudaMemcpyDeviceToHost);
	}
	if (outRenderW) *outRenderW = g_latest_render_w;
	if (outRenderH) *outRenderH = g_latest_render_h;
	return true;
}

void FORWARD::setChainmailActiveMapEnabled(bool enabled)
{
	g_useActiveMap = enabled;
	g_activeMapReset = true;
}

float FORWARD::getChainmailActiveRatio()
{
	return g_lastActiveRatio;
}

int FORWARD::getChainmailActiveCount()
{
	return g_lastActiveCount;
}

void FORWARD::setGpuChainmailMode(int mode)
{
	g_GpuChainmailMode = (mode == 1) ? 1 : 0;
	printf("g_GpuChainmailMode : %d \n", g_GpuChainmailMode);
}

int FORWARD::getGpuChainmailMode()
{
	return g_GpuChainmailMode;
}

void FORWARD::setChainmailParams(
	int propIters,
	int relaxIters,
	float propStrength,
	float stiffness,
	float damping)
{
	g_cmPropIters = (propIters < 0) ? 0 : propIters;
	g_cmRelaxIters = (relaxIters < 0) ? 0 : relaxIters;
	g_cmPropStrength = fmaxf(0.0f, propStrength);
	g_cmStiffness = fmaxf(0.0f, stiffness);
	g_cmDamping = fmaxf(0.0f, damping);
}

void FORWARD::getChainmailParams(
	int* propIters,
	int* relaxIters,
	float* propStrength,
	float* stiffness,
	float* damping)
{
	if (propIters) *propIters = g_cmPropIters;
	if (relaxIters) *relaxIters = g_cmRelaxIters;
	if (propStrength) *propStrength = g_cmPropStrength;
	if (stiffness) *stiffness = g_cmStiffness;
	if (damping) *damping = g_cmDamping;
}

void FORWARD::setChainmailMaterialParams(
	float constraintGlobalScale,
	float airScale,
	float skinScale,
	float boneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	g_cmConstraintGlobalScale = fmaxf(1e-4f, constraintGlobalScale);
	g_cmConstraintAirScale = fmaxf(1e-4f, airScale);
	g_cmConstraintSkinScale = fmaxf(1e-4f, skinScale);
	g_cmConstraintBoneScale = fmaxf(1e-4f, boneScale);
	g_cmUseEdgeStiffness = useEdgeStiffness;
	g_cmEdgeStiffnessInfluence = fmaxf(0.0f, edgeStiffnessInfluence);
}

void FORWARD::getChainmailMaterialParams(
	float* constraintGlobalScale,
	float* airScale,
	float* skinScale,
	float* boneScale,
	bool* useEdgeStiffness,
	float* edgeStiffnessInfluence)
{
	if (constraintGlobalScale) *constraintGlobalScale = g_cmConstraintGlobalScale;
	if (airScale) *airScale = g_cmConstraintAirScale;
	if (skinScale) *skinScale = g_cmConstraintSkinScale;
	if (boneScale) *boneScale = g_cmConstraintBoneScale;
	if (useEdgeStiffness) *useEdgeStiffness = g_cmUseEdgeStiffness;
	if (edgeStiffnessInfluence) *edgeStiffnessInfluence = g_cmEdgeStiffnessInfluence;
}

void FORWARD::setChainmailDynamicsParams(
	float inertiaGain,
	float velocityRetention,
	float velocityClamp)
{
	g_cmInertiaGain = fmaxf(0.0f, inertiaGain);
	g_cmVelocityRetention = fminf(fmaxf(0.0f, velocityRetention), 0.9999f);
	g_cmVelocityClamp = fmaxf(1e-5f, velocityClamp);
}

void FORWARD::getChainmailDynamicsParams(
	float* inertiaGain,
	float* velocityRetention,
	float* velocityClamp)
{
	if (inertiaGain) *inertiaGain = g_cmInertiaGain;
	if (velocityRetention) *velocityRetention = g_cmVelocityRetention;
	if (velocityClamp) *velocityClamp = g_cmVelocityClamp;
}

__device__ __forceinline__ void mat3Identity(float m[9]) {
	m[0] = 1.0f; m[1] = 0.0f; m[2] = 0.0f;
	m[3] = 0.0f; m[4] = 1.0f; m[5] = 0.0f;
	m[6] = 0.0f; m[7] = 0.0f; m[8] = 1.0f;
}

__device__ __forceinline__ float mat3Det(const float m[9]) {
	return m[0] * (m[4] * m[8] - m[5] * m[7])
		- m[1] * (m[3] * m[8] - m[5] * m[6])
		+ m[2] * (m[3] * m[7] - m[4] * m[6]);
}

__device__ __forceinline__ void mat3Transpose(const float in[9], float out[9]) {
	out[0] = in[0]; out[1] = in[3]; out[2] = in[6];
	out[3] = in[1]; out[4] = in[4]; out[5] = in[7];
	out[6] = in[2]; out[7] = in[5]; out[8] = in[8];
}

__device__ __forceinline__ bool mat3Inverse(const float m[9], float invOut[9]) {
	const float det = mat3Det(m);
	if (fabsf(det) < 1e-8f) {
		return false;
	}
	const float invDet = 1.0f / det;
	invOut[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
	invOut[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
	invOut[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
	invOut[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
	invOut[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
	invOut[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
	invOut[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
	invOut[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
	invOut[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
	return true;
}

__device__ __forceinline__ float3 mat3MulVec(const float m[9], const float3& v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2] * v.z,
		m[3] * v.x + m[4] * v.y + m[5] * v.z,
		m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

__device__ __forceinline__ void polarRotation3x3(const float A[9], float R[9]) {
	for (int i = 0; i < 9; ++i) {
		R[i] = A[i];
	}

	for (int iter = 0; iter < 5; ++iter) {
		float invR[9];
		if (!mat3Inverse(R, invR)) {
			mat3Identity(R);
			return;
		}
		float invRT[9];
		mat3Transpose(invR, invRT);
		for (int i = 0; i < 9; ++i) {
			R[i] = 0.5f * (R[i] + invRT[i]);
		}
	}

	// Keep a proper rotation.
	if (mat3Det(R) < 0.0f) {
		R[2] = -R[2];
		R[5] = -R[5];
		R[8] = -R[8];
	}
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value)
{
	int* address_as_i = reinterpret_cast<int*>(addr);
	int old = *address_as_i;
	while (__int_as_float(old) > value) {
		const int assumed = old;
		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
		if (assumed == old) {
			break;
		}
	}
	return __int_as_float(old);
}

struct CMConstraintGPU {
	float dx, dy, dz;
	float xShearY, xShearZ;
	float yShearX, yShearZ;
	float zShearX, zShearY;
};

__device__ __forceinline__ CMConstraintGPU cm_getConstraint(
	float density,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale) {
	float axisC;
	float shearC;
	float materialScale;
	if (density < FORWARD::AIR) {
		axisC = constraintAirScale;
		shearC = constraintAirScale;

		//axisC = 0.3f;
		//shearC = 0.3f;
		//materialScale = constraintAirScale;
	}
	else if (density < FORWARD::SKIN) {
		axisC = constraintSkinScale;
		shearC = constraintSkinScale;

		//axisC = 0.06f;
		//shearC = 0.06f;
		//materialScale = constraintSkinScale;
	}
	else {
		axisC = constraintBoneScale;
		shearC = constraintBoneScale;

		//axisC = 0.04f;
		//shearC = 0.04f;
		//materialScale = constraintBoneScale;
	}
	const float scale = fmaxf(1e-4f, constraintGlobalScale);
	axisC *= scale;
	shearC *= scale;
	CMConstraintGPU c;
	c.dx = axisC; c.dy = axisC; c.dz = axisC;
	c.xShearY = shearC; c.xShearZ = shearC;
	c.yShearX = shearC; c.yShearZ = shearC;
	c.zShearX = shearC; c.zShearY = shearC;
	return c;
}

__device__ __forceinline__ float cm_edgeScaleFromStiff(
	float edgeStiff,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	if (!useEdgeStiffness || edgeStiffnessInfluence <= 0.0f) {
		return 1.0f;
	}
	const float s = fminf(fmaxf(edgeStiff, 0.05f), 1.0f);
	const float t = (s - 0.05f) / 0.95f;
	const float scale = 1.0f + edgeStiffnessInfluence * (t * 2.0f - 1.0f);
	return fminf(fmaxf(scale, 0.1f), 4.0f);
}

__device__ __forceinline__ float cm_propagationTime(float d0, float d1) {
	float et, nt;
	if (d0 < FORWARD::AIR) et = 1.0f;
	else if (d0 < FORWARD::SKIN) et = 0.3f;
	else if (d0 < FORWARD::BONE) et = 0.05f;
	else et = 0.005f;

	if (d1 < FORWARD::AIR) nt = 1.0f;
	else if (d1 < FORWARD::SKIN) nt = 0.3f;
	else if (d1 < FORWARD::BONE) nt = 0.05f;
	else nt = 0.005f;
	return (et + nt) * 0.5f;
}

__device__ __forceinline__ float3 cm_correctionFromNeighbor(
	const float3& epos,
	const float3& npos,
	float targDist,
	float neighborDensity,
	float edgeStiff,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale)
{
	const CMConstraintGPU nC = cm_getConstraint(
		neighborDensity,
		constraintGlobalScale,
		constraintAirScale,
		constraintSkinScale,
		constraintBoneScale);
	const float edgeScale = cm_edgeScaleFromStiff(edgeStiff, useEdgeStiffness, edgeStiffnessInfluence);
	const float effectiveDx = nC.dx;// / edgeScale;
	float3 dir = f3_sub(epos, npos);
	float len = f3_len(dir);
	if (len < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);
	float invLen = 1.0f / len;
	float3 nDir = f3_mul(dir, invLen);

	if (len < targDist - effectiveDx) {//너무 가까울때
		float delta = (targDist - effectiveDx) - len;// 임계보다 가까운만큼의 차이를 delta로
		return f3_mul(nDir, delta * edgeScale); // push away
	}
	else if (len > targDist + effectiveDx) {// 너무 멀때
		float delta = len - (targDist + effectiveDx);
		return f3_mul(nDir, -(delta * edgeScale)); // pull closer
	}
	return make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void chainmailPropagateKernel(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float propStrength,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const float tcur = time_in[idx];

	float bestTime = tcur;
	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float cand = ntime + cm_propagationTime(edens, ndens);// 자신과 이웃의 density 를 넣음
		if (cand < bestTime) bestTime = cand;

		if (cand < tcur) {
			const float3 npos = pos_in[nIdx];
			const float targDist = nbrDist[off + j];
			const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
			float3 corr = cm_correctionFromNeighbor(
				epos, npos, targDist, ndens, edgeStiff, useEdgeStiffness,
				edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
				constraintSkinScale, constraintBoneScale);
			accum = f3_add(accum, corr);
			corrCount++;
		}
	}

	float3 outPos = epos;
	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));//이웃들로부터 받은 보정량을 평균해서 과도한 이동을 막기위한 안정화. 이웃들이 많거나 적으면 차이가 있을수있기때문.
		outPos = f3_add(epos, f3_mul(avgCorr, propStrength));
	}

	pos_out[idx] = outPos;
	time_out[idx] = bestTime;
}

// HP-ChainMail style: pick the single fastest neighbor (smallest cand) and move only w.r.t. that neighbor.
__global__ void chainmailPropagateKernelBest(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float propStrength,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const float tcur = time_in[idx];

	float bestTime = tcur;
	int bestIdx = -1;
	float bestDist = 0.0f;
	float bestNDens = 0.0f;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float cand = ntime + cm_propagationTime(edens, ndens);
		if (cand < bestTime) {
			bestTime = cand;
			bestIdx = nIdx;
			bestDist = nbrDist[off + j];
			bestNDens = ndens;
		}
	}

	float3 outPos = epos;
	if (bestIdx >= 0 && bestTime < tcur) {
		const float3 npos = pos_in[bestIdx];
		float bestEdgeStiff = 1.0f;
		for (int j = 0; j < cnt; ++j) {
			if (nbrIdx[off + j] == bestIdx) {
				bestEdgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
				break;
			}
		}
		const float3 corr = cm_correctionFromNeighbor(
			epos, npos, bestDist, bestNDens, bestEdgeStiff, useEdgeStiffness,
			edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
			constraintSkinScale, constraintBoneScale);
		if (f3_len(corr) > 0.0f) {
			outPos = f3_add(epos, f3_mul(corr, propStrength));
		}
	}

	pos_out[idx] = outPos;
	time_out[idx] = bestTime;
}

__global__ void chainmailRelaxKernel(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float stiffness,
	float damping,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const CMConstraintGPU eC = cm_getConstraint(
		edens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);

	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;
	float sumTarget = 0.0f;
	int distCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float3 npos = pos_in[nIdx];
		const float ndens = density[nIdx];
		const CMConstraintGPU nC = cm_getConstraint(
			ndens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);
		const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
		const float edgeScale = cm_edgeScaleFromStiff(edgeStiff, useEdgeStiffness, edgeStiffnessInfluence);

		const float targDist = nbrDist[off + j];
		// Relax tolerance directly controls "hard/soft" perception.
		// Previous factor (0.002x) made UI scaling visually subtle.
		const float relaxTolScale = 0.0000001f;
		const float tol = (relaxTolScale * (eC.dx + nC.dx)) / edgeScale;

		float3 dir = f3_sub(npos, epos);
		float len = f3_len(dir);
		if (len < 1e-6f) continue;

		float invLen = 1.0f / len;
		float3 nDir = f3_mul(dir, invLen);

		if (len < targDist - tol) {
			float delta = (targDist - tol) - len;
			accum = f3_sub(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}
		else if (len > targDist + tol) {
			float delta = len - (targDist + tol);
			accum = f3_add(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}

		sumTarget += targDist;
		distCount++;
	}

	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));

		float maxStep = 0.6f;
		if (distCount > 0) {
			float avgTarget = sumTarget / float(distCount);
			const float maxStepFactor = 1.5f;
			maxStep = maxStepFactor * avgTarget;
		}

		float corrLen = f3_len(avgCorr);
		if (corrLen > 1e-6f && corrLen > maxStep) {
			avgCorr = f3_mul(avgCorr, maxStep / corrLen);
		}

		float3 corrected = f3_add(epos, f3_mul(avgCorr, stiffness));
		// damping: reduce correction magnitude (0 = no damping, 1 = fully frozen)
		pos_out[idx] = f3_add(epos, f3_mul(f3_sub(corrected, epos), 1.0f - damping));
	}
	else {
		pos_out[idx] = epos;
	}
}

__global__ void chainmailPropagateKernelActive(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float propStrength,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	const int* active_map,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	if (!active_map || active_map[idx] == 0) {
		pos_out[idx] = pos_in[idx];
		time_out[idx] = time_in[idx];
		return;
	}

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const float tcur = time_in[idx];

	float bestTime = tcur;
	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float propTime = cm_propagationTime(edens, ndens);

		const float cand = ntime + propTime;
		if (cand < bestTime) bestTime = cand;

		const float3 npos = pos_in[nIdx];
		const float targDist = nbrDist[off + j];
		const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
		float3 corr = cm_correctionFromNeighbor(
			epos, npos, targDist, ndens, edgeStiff, useEdgeStiffness,
			edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
			constraintSkinScale, constraintBoneScale);
		if (f3_len(corr) > 0.0f) {
			accum = f3_add(accum, corr);
			corrCount++;
			// Spread activity to neighbors that actually need correction.
			next_map[nIdx] = 1;
		}
	}

	const float baseTime = (bestTime < tcur) ? bestTime : tcur;
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float propTime = cm_propagationTime(edens, ndens);
		if (baseTime + propTime < ntime) {
			next_map[nIdx] = 1;
		}
	}

	float3 outPos = epos;
	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));
		outPos = f3_add(epos, f3_mul(avgCorr, propStrength));
	}

	pos_out[idx] = outPos;
	time_out[idx] = bestTime;

	if (corrCount > 0 || bestTime < tcur) {
		next_map[idx] = 1;
	}
}

__global__ void chainmailPropagateScatterKernel(
	int N,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const int* active_map,
	int* best_from,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (!active_map || active_map[idx] == 0) return;

	const float tcur = time_in[idx];
	const float edens = density[idx];
	const int off = offset[idx];
	const int cnt = nbrCount[idx];

	next_map[idx] = 1;

	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ndens = density[nIdx];
		const float newTime = tcur + cm_propagationTime(edens, ndens);
		const float old = atomicMinFloat(time_out + nIdx, newTime);
		if (newTime < old) {
			best_from[nIdx] = idx;
			next_map[nIdx] = 1;
		}
	}
}

__global__ void applyBestFromKernel(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	const int* best_from,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const int src = best_from[idx];
	if (src < 0) {
		pos_out[idx] = pos_in[idx];
		return;
	}

	float targStiff = 1.0f;
	float targDist = -1.0f;
	const int offSrc = offset[src];
	const int cntSrc = nbrCount[src];
	for (int j = 0; j < cntSrc; ++j) {
		if (nbrIdx[offSrc + j] == idx) {
			targDist = nbrDist[offSrc + j];
			targStiff = nbrStiff ? nbrStiff[offSrc + j] : 1.0f;
			break;
		}
	}
	if (targDist < 0.0f) {
		const int off = offset[idx];
		const int cnt = nbrCount[idx];
		for (int j = 0; j < cnt; ++j) {
			if (nbrIdx[off + j] == src) {
				targDist = nbrDist[off + j];
				targStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
				break;
			}
		}
	}
	if (targDist < 0.0f) {
		pos_out[idx] = pos_in[idx];
		return;
	}

	const float3 epos = pos_in[idx];
	const float3 npos = pos_in[src];
	const float ndens = density[src];
	const float3 corr = cm_correctionFromNeighbor(
		epos, npos, targDist, ndens, targStiff, useEdgeStiffness,
		edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
		constraintSkinScale, constraintBoneScale);
	if (f3_len(corr) > 0.0f) {
		pos_out[idx] = f3_add(epos, corr);
		next_map[idx] = 1;
	}
	else {
		pos_out[idx] = epos;
	}
}

__global__ void chainmailRelaxKernelActive(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float stiffness,
	float damping,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	const int* active_map,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	if (!active_map || active_map[idx] == 0) {
		pos_out[idx] = pos_in[idx];
		return;
	}

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const CMConstraintGPU eC = cm_getConstraint(
		edens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);

	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;
	float sumTarget = 0.0f;
	int distCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float3 npos = pos_in[nIdx];
		const float ndens = density[nIdx];
		const CMConstraintGPU nC = cm_getConstraint(
			ndens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);
		const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
		const float edgeScale = cm_edgeScaleFromStiff(edgeStiff, useEdgeStiffness, edgeStiffnessInfluence);

		const float targDist = nbrDist[off + j];
		// Relax tolerance directly controls "hard/soft" perception.
		// Previous factor (0.002x) made UI scaling visually subtle.
		const float relaxTolScale = 0.10f;
		const float tol = (relaxTolScale * (eC.dx + nC.dx)) / edgeScale;

		float3 dir = f3_sub(npos, epos);
		float len = f3_len(dir);
		if (len < 1e-6f) continue;

		float invLen = 1.0f / len;
		float3 nDir = f3_mul(dir, invLen);

		if (len < targDist - tol) {
			float delta = (targDist - tol) - len;
			accum = f3_sub(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}
		else if (len > targDist + tol) {
			float delta = len - (targDist + tol);
			accum = f3_add(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}

		sumTarget += targDist;
		distCount++;
	}

	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));

		float maxStep = 0.6f;
		if (distCount > 0) {
			float avgTarget = sumTarget / float(distCount);
			const float maxStepFactor = 1.5f;
			maxStep = maxStepFactor * avgTarget;
		}

		float corrLen = f3_len(avgCorr);
		if (corrLen > 1e-6f && corrLen > maxStep) {
			avgCorr = f3_mul(avgCorr, maxStep / corrLen);
		}

		float3 corrected = f3_add(epos, f3_mul(avgCorr, stiffness));
		pos_out[idx] = f3_add(epos, f3_mul(f3_sub(corrected, epos), 1.0f - damping));
		next_map[idx] = 1;
	}
	else {
		pos_out[idx] = epos;
	}
}

__global__ void chainmailInertiaKernel(
	int N,
	const float3* pos_prev,
	float3* pos_curr,
	float3* vel,
	const float* invMass,
	float inertiaGain,// 총 계산된 관성 반영 파라메터
	float velocityRetention,//에너지 반영 파라메터(낮을수록 제동걸림)
	float velocityClamp//충격량 파라메터(낮을수록 적용되는 속도 제한)
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	if (invMass && invMass[idx] <= 0.0f) {
		vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}

	const float3 disp = f3_sub(pos_curr[idx], pos_prev[idx]);//이번 프레임에서 체인메일 알고리즘(전파/안정화)에 의해 가우시안이 실제로 이동한 거리와 방향
	float3 v = f3_add(f3_mul(vel[idx], velocityRetention), disp);//이전 프레임에서 가지고 있던 속도(vel)를 일정 비율 유지하고, 여기에 이번에 새로 발생한 움직임(disp)을 더함
	const float vLen = f3_len(v);//이를 통해 가우시안은 "방금 움직였던 방향으로 계속 가려는 성질"을 기억
	if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) {
		v = make_float3(0.0f, 0.0f, 0.0f);
	}
	else if (vLen > velocityClamp && vLen > 1e-8f) {
		v = f3_mul(v, velocityClamp / vLen);
	}
	vel[idx] = v;
	pos_curr[idx] = f3_add(pos_curr[idx], f3_mul(v, inertiaGain));
}

__global__ void countActiveKernel(int N, const int* active_map, int* out_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (active_map && active_map[idx]) {
		atomicAdd(out_count, 1);
	}
}

__global__ void resetTimeKernel(int N, float* time, float value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	time[idx] = value;
}

__global__ void applySeedCommandsKernel(
	int n,
	const int* idx,
	const float3* delta,
	float3* pos,
	float* time,
	int* active_map)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const int id = idx[i];
	const float3 d = delta[i];
	pos[id] = f3_add(pos[id], d);
	time[id] = 0.0f;
	if (active_map) {
		active_map[id] = 1;
	}
}

__global__ void PackKernel(
	int N,
	const float3* pos,
	const float* time,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	int Ncount,
	float* out_means3D,
	int* out_nbr_index,
	int* out_nbr_count,
	float* out_nbr_time)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 p = pos[idx];
	out_means3D[3 * idx + 0] = p.x;
	out_means3D[3 * idx + 1] = p.y;
	out_means3D[3 * idx + 2] = p.z;

	out_nbr_time[idx] = time[idx];
	out_nbr_count[idx] = nbrCount[idx];

	const int off = offset[idx];
	for (int k = 0; k < Ncount; ++k) {
		if (k < nbrCount[idx]) {
			out_nbr_index[idx * Ncount + k] = nbrIdx[off + k];
		}
		else {
			out_nbr_index[idx * Ncount + k] = -1;
		}
	}
}

static int computeTotalNeighbors(const FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	int total = 0;
	for (int i = 0; i < N; ++i) {
		const auto& e = cm.getElement(i);
		const int end = e.offset + e.neighborCnt;
		if (end > total) total = end;
	}
	return total;
}

static void ensureChainmailGPU(const FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	const int totalNeighbors = computeTotalNeighbors(cm);

	if (chainmailInit && N == cm_num_elements && totalNeighbors == cm_num_neighbors) {//초기화 1회만 하기
		return;
	}

	if (d_pos_curr) cudaFree(d_pos_curr);
	if (d_pos_next) cudaFree(d_pos_next);
	if (d_pos_prev) cudaFree(d_pos_prev);
	if (d_vel) cudaFree(d_vel);
	if (d_density) cudaFree(d_density);
	if (d_invMass) cudaFree(d_invMass);
	if (d_time_curr) cudaFree(d_time_curr);
	if (d_time_next) cudaFree(d_time_next);
	if (d_offset) cudaFree(d_offset);
	if (d_nbrCount) cudaFree(d_nbrCount);
	if (d_nbrIdx) cudaFree(d_nbrIdx);
	if (d_nbrDist) cudaFree(d_nbrDist);
	if (d_nbrStiff) cudaFree(d_nbrStiff);
	if (d_active_map) cudaFree(d_active_map);
	if (d_next_map) cudaFree(d_next_map);
	if (d_active_count) cudaFree(d_active_count);
	if (d_best_from) cudaFree(d_best_from);

	cm_num_elements = N;
	cm_num_neighbors = totalNeighbors;

	cudaMalloc(&d_pos_curr, sizeof(float3) * N);
	cudaMalloc(&d_pos_next, sizeof(float3) * N);
	cudaMalloc(&d_pos_prev, sizeof(float3) * N);
	cudaMalloc(&d_vel, sizeof(float3) * N);
	cudaMalloc(&d_density, sizeof(float) * N);
	cudaMalloc(&d_invMass, sizeof(float) * N);
	cudaMalloc(&d_time_curr, sizeof(float) * N);
	cudaMalloc(&d_time_next, sizeof(float) * N);
	cudaMalloc(&d_offset, sizeof(int) * N);
	cudaMalloc(&d_nbrCount, sizeof(int) * N);
	cudaMalloc(&d_nbrIdx, sizeof(int) * totalNeighbors);
	cudaMalloc(&d_nbrDist, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_nbrStiff, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_active_map, sizeof(int) * N);
	cudaMalloc(&d_next_map, sizeof(int) * N);
	cudaMalloc(&d_active_count, sizeof(int));
	cudaMalloc(&d_best_from, sizeof(int) * N);
	cudaMemset(d_active_map, 0, sizeof(int) * N);
	cudaMemset(d_next_map, 0, sizeof(int) * N);
	cudaMemset(d_active_count, 0, sizeof(int));
	cudaMemset(d_best_from, 0xFF, sizeof(int) * N);

	std::vector<float3> h_pos(N);
	std::vector<float3> h_vel(N);
	std::vector<float> h_density(N);
	std::vector<float> h_invMass(N);
	std::vector<float> h_time(N);
	std::vector<int> h_offset(N);
	std::vector<int> h_count(N);
	std::vector<int> h_idx(totalNeighbors);
	std::vector<float> h_dist(totalNeighbors);
	std::vector<float> h_stiff(totalNeighbors);
	cudaMemcpy(d_pos_curr, h_pos.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, h_vel.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_density, h_density.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_invMass, h_invMass.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_time_curr, h_time.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, h_offset.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrCount, h_count.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrIdx, h_idx.data(), sizeof(int) * totalNeighbors, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrDist, h_dist.data(), sizeof(float) * totalNeighbors, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrStiff, h_stiff.data(), sizeof(float) * totalNeighbors, cudaMemcpyHostToDevice);

	chainmailInit = true;
}

static void uploadChainmailPositions(const FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	std::vector<float3> h_pos(N);
	std::vector<float> h_time(N);
	for (int i = 0; i < N; ++i) {
		const auto& e = cm.getElement(i);
		h_pos[i] = make_float3(e.pos.x, e.pos.y, e.pos.z);
		h_time[i] = e.time;
	}
	cudaMemcpy(d_pos_curr, h_pos.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_time_curr, h_time.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
}

static void uploadChainmailPositionsIndexed(const FORWARD::ChainMail& cm, const std::vector<int>& indices)
{
	const int N = static_cast<int>(cm.numElements());
	if (indices.empty() || N == 0) {
		return;
	}

	std::vector<int> uniq = indices;
	std::sort(uniq.begin(), uniq.end());
	uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

	for (int idx : uniq) {
		if (idx < 0 || idx >= N) {
			continue;
		}
		const auto& e = cm.getElement(idx);
		const float3 p = make_float3(e.pos.x, e.pos.y, e.pos.z);
		cudaMemcpy(d_pos_curr + idx, &p, sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_time_curr + idx, &e.time, sizeof(float), cudaMemcpyHostToDevice);
	}
}

static void downloadChainmailPositions(FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	std::vector<float3> h_pos(N);
	std::vector<float> h_time(N);
	cudaMemcpy(h_pos.data(), d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_time.data(), d_time_curr, sizeof(float) * N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i) {
		auto& e = cm.getElement(i);
		e.pos = glm::vec3(h_pos[i].x, h_pos[i].y, h_pos[i].z);
		e.time = h_time[i];
	}
}

//static void applySeedCommandsGPU(FORWARD::ChainMail& cm)
//{
//	const int N = static_cast<int>(cm.numElements());
//	if (N <= 0) return;
//
//	if (g_useActiveMap && g_activeMapReset) {
//		cudaMemset(d_active_map, 0, sizeof(int) * N);
//		cudaMemset(d_next_map, 0, sizeof(int) * N);
//		g_activeMapReset = false;
//	}
//
//	std::vector<int> cmdIdx;
//	std::vector<glm::vec3> cmdDelta;
//	drainGpuCommands(cmdIdx, cmdDelta);
//	if (cmdIdx.empty()) return;
//
//	struct Cmd { int idx; glm::vec3 delta; };
//	std::vector<Cmd> cmds;
//	cmds.reserve(cmdIdx.size());
//	for (size_t i = 0; i < cmdIdx.size(); ++i) {
//		const int idx = cmdIdx[i];
//		if (idx < 0 || idx >= N) continue;
//		cmds.push_back({ idx, cmdDelta[i] });
//	}
//	if (cmds.empty()) return;
//
//	std::sort(cmds.begin(), cmds.end(), [](const Cmd& a, const Cmd& b) {
//		return a.idx < b.idx;
//	});
//
//	std::vector<int> h_idx;
//	std::vector<float3> h_delta;
//	h_idx.reserve(cmds.size());
//	h_delta.reserve(cmds.size());
//
//	Cmd current = cmds[0];
//	for (size_t i = 1; i < cmds.size(); ++i) {
//		if (cmds[i].idx == current.idx) {
//			current.delta += cmds[i].delta;
//		}
//		else {
//			h_idx.push_back(current.idx);
//			h_delta.push_back(make_float3(current.delta.x, current.delta.y, current.delta.z));
//			current = cmds[i];
//		}
//	}
//	h_idx.push_back(current.idx);
//	h_delta.push_back(make_float3(current.delta.x, current.delta.y, current.delta.z));
//
//	if (g_useActiveMap) {
//		const int threads = 256;
//		const int blocksN = (N + threads - 1) / threads;
//		resetTimeKernel << <blocksN, threads >> > (N, d_time_curr, 1e9f);
//		resetTimeKernel << <blocksN, threads >> > (N, d_time_next, 1e9f);
//		cudaMemset(d_active_map, 0, sizeof(int) * N);
//		cudaMemset(d_next_map, 0, sizeof(int) * N);
//	}
//
//	int* d_idx = nullptr;
//	float3* d_delta = nullptr;
//	const int count = static_cast<int>(h_idx.size());
//	cudaMalloc(&d_idx, sizeof(int) * count);
//	cudaMalloc(&d_delta, sizeof(float3) * count);
//	cudaMemcpy(d_idx, h_idx.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_delta, h_delta.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);
//
//	const int threads = 256;
//	const int blocks = (count + threads - 1) / threads;
//	applySeedCommandsKernel << <blocks, threads >> > (
//		count,
//		d_idx,
//		d_delta,
//		d_pos_curr,
//		d_time_curr,
//		g_useActiveMap ? d_active_map : nullptr);
//	cudaDeviceSynchronize();
//
//	cudaFree(d_idx);
//	cudaFree(d_delta);
//}
static int g_currentAnchorIdx = -1;
// 매 프레임 할당을 피하기 위한 정적 버퍼
static int* d_idx_static = nullptr;
static float3* d_delta_static = nullptr;
static const int MAX_SEEDS = 10000;

static void applySeedCommandsGPU(FORWARD::ChainMail& cm)
{
    const int N = static_cast<int>(cm.numElements());
    if (N <= 0) return;

    // 1. 명령어 가져오기
    std::vector<int> cmdIdx;
    std::vector<glm::vec3> cmdDelta;
    drainGpuCommands(cmdIdx, cmdDelta);

    // 명령어가 없으면 앵커 해제 후 종료
    if (cmdIdx.empty()) {
        g_currentAnchorIdx = -1;
        return;
    }

    // 2. BFS 이웃 확장

    std::set<int> selectedNodes;
    std::queue<std::pair<int, int>> q;
    glm::vec3 totalDelta(0.0f);

    for (size_t i = 0; i < cmdIdx.size(); ++i) {
        int idx = cmdIdx[i];
        if (idx >= 0 && idx < N) {
            if (selectedNodes.insert(idx).second) {
                q.push({ idx, 0 });
                totalDelta += cmdDelta[i];
            }
        }
    }

    if (selectedNodes.empty()) return;

    // 평균 이동량 및 대표 앵커 설정
    glm::vec3 avgDelta = totalDelta / (float)cmdIdx.size();
    g_currentAnchorIdx = cmdIdx[0];

    while (!q.empty()) {
        auto current = q.front(); q.pop();
        int curIdx = current.first;
        int depth = current.second;

        if (depth >= bfsHops) continue;

        const auto& elem = cm.getElement(curIdx);
        for (int k = 0; k < elem.neighborCnt; ++k) {
            int nIdx = cm.getNeighbor(elem.offset + k).idx;
            if (nIdx >= 0 && nIdx < N && selectedNodes.insert(nIdx).second) {
                q.push({ nIdx, depth + 1 });
            }
        }
    }

    // 3. [수정됨] BFS로 모은 모든 이웃을 h_idx에 담기
    std::vector<int> h_idx;
    std::vector<float3> h_delta;
    h_idx.reserve(selectedNodes.size());
    h_delta.reserve(selectedNodes.size());

    float3 f3Delta = make_float3(avgDelta.x, avgDelta.y, avgDelta.z);
    for (int nodeIdx : selectedNodes) {
        h_idx.push_back(nodeIdx);
        h_delta.push_back(f3Delta); // 모두 같은 평균 이동량 적용
    }

    // 4. GPU 전송 및 커널 실행
    const int count = static_cast<int>(h_idx.size());

    // 정적 버퍼 초기화 (1회)
    if (!d_idx_static) {
        cudaMalloc(&d_idx_static, sizeof(int) * MAX_SEEDS);
        cudaMalloc(&d_delta_static, sizeof(float3) * MAX_SEEDS);
    }

    int safeCount = std::min(count, MAX_SEEDS);
    cudaMemcpy(d_idx_static, h_idx.data(), sizeof(int) * safeCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_static, h_delta.data(), sizeof(float3) * safeCount, cudaMemcpyHostToDevice);

    if (g_useActiveMap) {
        const int threads = 256;
        const int blocksN = (N + threads - 1) / threads;
        resetTimeKernel << <blocksN, threads >> > (N, d_time_curr, 1e9f);
        cudaMemset(d_active_map, 0, sizeof(int) * N);
    }

    const int threads = 256;
    const int blocks = (safeCount + threads - 1) / threads;
    applySeedCommandsKernel << <blocks, threads >> > (
        safeCount,
        d_idx_static,
        d_delta_static,
        d_pos_curr,
        d_time_curr,
        g_useActiveMap ? d_active_map : nullptr);

    cudaDeviceSynchronize();
}
static void runChainmailRelaxGPU(FORWARD::ChainMail& cm, int iterations, float stiffness, float damping, bool uploadFromCPU)
{
	ensureChainmailGPU(cm);
	if (uploadFromCPU) {
		uploadChainmailPositions(cm);
	}

	const int N = static_cast<int>(cm.numElements());
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	const bool useInertia = (g_cmInertiaGain > 0.0f);

	if (useInertia) {
		cudaMemcpy(d_pos_prev, d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
	}

	for (int it = 0; it < iterations; ++it) {
		chainmailRelaxKernel << <blocks, threads >> > (
			N,
			d_pos_curr,
			d_pos_next,
			d_density,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			d_nbrDist,
			d_nbrStiff,
			stiffness,
			damping,
			g_cmConstraintGlobalScale,
			g_cmConstraintAirScale,
			g_cmConstraintSkinScale,
			g_cmConstraintBoneScale,
			g_cmUseEdgeStiffness,
			g_cmEdgeStiffnessInfluence
			);
		std::swap(d_pos_curr, d_pos_next);
	}

	if (useInertia) {
		chainmailInertiaKernel << <blocks, threads >> > (
			N,
			d_pos_prev,
			d_pos_curr,
			d_vel,
			d_invMass,
			g_cmInertiaGain,
			g_cmVelocityRetention,
			g_cmVelocityClamp
			);
	}

	downloadChainmailPositions(cm);
}

static void runChainmailGPU(FORWARD::ChainMail& cm, int propIters, int relaxIters, float propStrength, float relaxStiffness, float relaxDamping, bool uploadFromCPU, bool downloadToCPU, const std::vector<int>* uploadIndices)
{
	//ensureChainmailGPU(cm);
	if (uploadFromCPU) {
		uploadChainmailPositions(cm);
	}
	else if (uploadIndices && !uploadIndices->empty()) {
		//uploadChainmailPositionsIndexed(cm, *uploadIndices);
	}

	const int N = static_cast<int>(cm.numElements());// 처리할 가우시안 개수
	const int threads = 256;// 블록당 스레드 수
	const int blocks = (N + threads - 1) / threads;// N 개를 256개 씩 나눠서 처리, 단 나머지가 있을 시 블록 하나 더 생성.
	const bool useInertia = (g_cmInertiaGain > 0.0f);
	if (useInertia) {
		cudaMemcpy(d_pos_prev, d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
	}

	const bool useActiveMap = g_useActiveMap && !uploadFromCPU;
	if (useActiveMap) {
		if (g_activeMapReset) {
			cudaMemset(d_active_map, 0, sizeof(int) * N);
			cudaMemset(d_next_map, 0, sizeof(int) * N);
			g_activeMapReset = false;
		}

		for (int it = 0; it < propIters; ++it) {
			cudaMemcpy(d_time_next, d_time_curr, sizeof(float) * N, cudaMemcpyDeviceToDevice);
			cudaMemset(d_best_from, 0xFF, sizeof(int) * N);
			cudaMemset(d_next_map, 0, sizeof(int) * N);
			chainmailPropagateScatterKernel << <blocks, threads >> > (
				N,
				d_time_curr,
				d_time_next,
				d_density,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_active_map,
				d_best_from,
				d_next_map
				);
			applyBestFromKernel << <blocks, threads >> > (
				N,
				d_pos_curr,
				d_pos_next,
				d_density,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_nbrDist,
				d_nbrStiff,
				g_cmConstraintGlobalScale,
				g_cmConstraintAirScale,
				g_cmConstraintSkinScale,
				g_cmConstraintBoneScale,
				g_cmUseEdgeStiffness,
				g_cmEdgeStiffnessInfluence,
				d_best_from,
				d_next_map
				);
			std::swap(d_pos_curr, d_pos_next);
			std::swap(d_time_curr, d_time_next);
			std::swap(d_active_map, d_next_map);
		}

		for (int it = 0; it < relaxIters; ++it) {
			cudaMemset(d_next_map, 0, sizeof(int) * N);
			chainmailRelaxKernelActive << <blocks, threads >> > (
				N,
				d_pos_curr,
				d_pos_next,
				d_density,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_nbrDist,
				d_nbrStiff,
				relaxStiffness,
				relaxDamping,
				g_cmConstraintGlobalScale,
				g_cmConstraintAirScale,
				g_cmConstraintSkinScale,
				g_cmConstraintBoneScale,
				g_cmUseEdgeStiffness,
				g_cmEdgeStiffnessInfluence,
				d_active_map,
				d_next_map
				);
			std::swap(d_pos_curr, d_pos_next);
			std::swap(d_active_map, d_next_map);
		}

		if (useInertia) {
			chainmailInertiaKernel << <blocks, threads >> > (
				N,
				d_pos_prev,
				d_pos_curr,
				d_vel,
				d_invMass,
				g_cmInertiaGain,
				g_cmVelocityRetention,
				g_cmVelocityClamp
				);
		}
		cudaMemset(d_active_count, 0, sizeof(int));
		countActiveKernel << <blocks, threads >> > (N, d_active_map, d_active_count);
		cudaMemcpy(&g_lastActiveCount, d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
		g_lastActiveRatio = (N > 0) ? (float)g_lastActiveCount / float(N) : 0.0f;
		return;
	}

	for (int it = 0; it < propIters; ++it) {
		chainmailPropagateKernelBest << <blocks, threads >> > (
			N,
			d_pos_curr,
			d_pos_next,
			d_time_curr,
			d_time_next,
			d_density,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			d_nbrDist,
			d_nbrStiff,
			propStrength,
			g_cmConstraintGlobalScale,
			g_cmConstraintAirScale,
			g_cmConstraintSkinScale,
			g_cmConstraintBoneScale,
			g_cmUseEdgeStiffness,
			g_cmEdgeStiffnessInfluence
			);
		std::swap(d_pos_curr, d_pos_next);
		std::swap(d_time_curr, d_time_next);
	}

	for (int it = 0; it < relaxIters; ++it) {
		chainmailRelaxKernel << <blocks, threads >> > (
			N,
			d_pos_curr,
			d_pos_next,
			d_density,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			d_nbrDist,
			d_nbrStiff,
			relaxStiffness,
			relaxDamping,
			g_cmConstraintGlobalScale,
			g_cmConstraintAirScale,
			g_cmConstraintSkinScale,
			g_cmConstraintBoneScale,
			g_cmUseEdgeStiffness,
			g_cmEdgeStiffnessInfluence
			);
		std::swap(d_pos_curr, d_pos_next);
	}

	if (useInertia) {
		chainmailInertiaKernel << <blocks, threads >> > (// 관성에대해서 효과를 주기위한 커널.
			N,
			d_pos_prev,
			d_pos_curr,
			d_vel,
			d_invMass,
			g_cmInertiaGain,
			g_cmVelocityRetention,
			g_cmVelocityClamp
			);
	}
	chainmailRelaxKernel << <blocks, threads >> > (
		N,
		d_pos_curr,
		d_pos_next,
		d_density,
		d_offset,
		d_nbrCount,
		d_nbrIdx,
		d_nbrDist,
		d_nbrStiff,
		relaxStiffness,
		relaxDamping,
		g_cmConstraintGlobalScale,
		g_cmConstraintAirScale,
		g_cmConstraintSkinScale,
		g_cmConstraintBoneScale,
		g_cmUseEdgeStiffness,
		g_cmEdgeStiffnessInfluence
		);
	std::swap(d_pos_curr, d_pos_next);
	g_lastActiveCount = N;
	g_lastActiveRatio = (N > 0) ? 1.0f : 0.0f;

	//if (downloadToCPU) {
	//	downloadChainmailPositions(cm);
	//}
}

#include <chrono>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
void FORWARD::preprocess(
	FORWARD::ChainMail& cm,
	std::vector<int>& activeSet,

	int P, int D, int M,
	 float* means3D,
	int nbr_K,
	const glm::vec3* scales,
	const float scale_modifier,
	const float _rotatingModifier_COV2D_Matrix_x,
	const float _rotatingModifier_COV2D_Matrix_y,
	const float _rotatingModifier_COV2D_Matrix_z,	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	bool antialiasing,float t,
	bool enableDeformationCovariance
	)
{
	const bool useGpuChainmail = (g_GpuChainmailMode == 1);
	//
	const int cmRelaxIters = g_cmRelaxIters;
	const float cmStiffness = g_cmStiffness;
	const float cmDamping = g_cmDamping;
	const int cmPropIters = g_cmPropIters;
	const float cmPropStrength = g_cmPropStrength;
	//
	//
	const bool cmSyncCPU = false;
	setGpuCommandMode(useGpuChainmail && !cmSyncCPU);
	// ==========================================
	// [측정용 변수] 함수 내부에 static으로 선언
	// ==========================================
	static std::vector<float> timeLog_Prop;
	static std::vector<float> timeLog_Cov;
	static std::vector<float> timeLog_Input;
	static int measureCount = 0;

	// 측정 시작: Propagation (ChainMail)
	static cudaEvent_t ev_prop_start = nullptr;
	static cudaEvent_t ev_prop_end = nullptr;
	static cudaEvent_t ev_cov_start = nullptr;
	static cudaEvent_t ev_cov_end = nullptr;
	static bool ev_init = false;
	if (!ev_init) {
		cudaEventCreate(&ev_prop_start);
		cudaEventCreate(&ev_prop_end);
		cudaEventCreate(&ev_cov_start);
		cudaEventCreate(&ev_cov_end);
		ev_init = true;
	}
	auto cpu_input_start = std::chrono::high_resolution_clock::now();
	float cpu_input_ms = 0.0f;
	float cpu_prop_ms = 0.0f;
	bool prop_timing_valid = false;
	// ----------------------------------------------------------------
	// (1) ChainMail Propagation 로직 (기존 코드)
	// ----------------------------------------------------------------
	std::vector<int> frameActive;

	static float prevD = 0.0f;
	static bool applyForce = true;
	float amp = 0.004f;
	float speed = 8.0f;
	float d = amp * sin(t * speed);
	float maxAngle = 0.02f;
	float angle = maxAngle * sin(t * speed); // 현재 프레임의 회전 각도
	if (cm.singleDeformTask.isRunning) {
	cm.startWave(cm.singleDeformTask.seedIdx, glm::vec3(d, 0, 0), activeSet);
	}

	else {
		// ... (기존 else 구문 유지) ...
		if (useGpuChainmail) {
			if (!cmSyncCPU) {
				cpu_input_start = std::chrono::high_resolution_clock::now();
				ensureChainmailGPU(cm);
				applySeedCommandsGPU(cm);
				cpu_input_ms = std::chrono::duration<float, std::milli>(
					std::chrono::high_resolution_clock::now() - cpu_input_start).count();
			}
			cudaEventRecord(ev_prop_start);
			runChainmailGPU(cm, cmPropIters, cmRelaxIters, cmPropStrength, cmStiffness, cmDamping, cmSyncCPU, cmSyncCPU, nullptr);
			cudaEventRecord(ev_prop_end);
			cudaEventSynchronize(ev_prop_end);
			prop_timing_valid = true;
		}
		else {
			//cpu cm
			auto cpu_prop_start = std::chrono::high_resolution_clock::now();
			cm.propagate(frameActive);
			activeSet.insert(activeSet.end(), frameActive.begin(), frameActive.end());
			std::sort(activeSet.begin(), activeSet.end());
			activeSet.erase(std::unique(activeSet.begin(), activeSet.end()), activeSet.end());
			cm.relax(activeSet);
			cpu_prop_ms = std::chrono::duration<float, std::milli>(
				std::chrono::high_resolution_clock::now() - cpu_prop_start).count();
		}
	}

	// 측정 종료: Propagation


	// 측정 시작: Covariance Update (CUDA Kernel)
	cudaEventRecord(ev_cov_start);
	//activeSet.clear();  // 프레임 단위 reset

	//if (cm.isWaveRunning())
		//cm.propagateStep(activeSet);

	//cm.propagate(activeSet);
	//cm.relax(activeSet);

	// Allocate the per-scene deformation buffers on demand.
	static bool initialized = false;
	static int allocatedPointCount = 0;
	static int allocatedNeighborCapacity = 0;
	static float* d_means3D = nullptr;
	static int* nbr_index = nullptr;
	static int* nbr_count = nullptr;
	static float* nbr_time = nullptr;

	const int Ncount = std::max(1, nbr_K);
	if (!initialized || allocatedPointCount != P || allocatedNeighborCapacity != Ncount) {
		cudaFree(d_means3D);
		cudaFree(nbr_index);
		cudaFree(nbr_count);
		cudaFree(nbr_time);
		cudaMallocManaged(&d_means3D, sizeof(float) * 3 * P);
		cudaMallocManaged(&nbr_index, sizeof(int) * P * Ncount);
		cudaMallocManaged(&nbr_count, sizeof(int) * P);
		cudaMallocManaged(&nbr_time, sizeof(float) * P);
		allocatedPointCount = P;
		allocatedNeighborCapacity = Ncount;
		initialized = true;
	}
	// Expose the latest deformed positions buffer for debug visualization.
	g_latest_deformed_xyz = d_means3D;
	g_latest_deformed_count = P;
	// Expose the exact render-path 2D projection buffers for debug visualization.
	g_latest_means2d = means2D;
	g_latest_radii = radii;
	g_latest_project_count = P;
	g_latest_render_w = W;
	g_latest_render_h = H;
	if (useGpuChainmail) {
		const int threads = 256;
		const int blocks = (P + threads - 1) / threads;
		PackKernel << <blocks, threads >> > (
			P,
			d_pos_curr,
			d_time_curr,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			Ncount,
			d_means3D,
			nbr_index,
			nbr_count,
			nbr_time
			);
		cudaDeviceSynchronize();
	}
	else {
		for (int i = 0; i < P; ++i) {
			Element E = cm.getElement(i);
			const int neighborCount = std::min(E.neighborCnt, Ncount);
			for (int k = 0; k < neighborCount; ++k) {
				Neighbor n = cm.getNeighbor(E.offset + k);
				nbr_index[i * Ncount + k] = n.idx;
			}
			nbr_count[i] = neighborCount;
			d_means3D[3 * i + 0] = E.pos.x;
			d_means3D[3 * i + 1] = E.pos.y;
			d_means3D[3 * i + 2] = E.pos.z;
			nbr_time[i] = E.time;
		}
		cudaDeviceSynchronize();
	}

	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		d_means3D,
		means3D,
		// --- neighborhood for 3D F ---
		nbr_index,    // size: P*MAX_K
		nbr_count,    // size: P
		Ncount,
		nbr_time,
		scales,
		scale_modifier,


		_rotatingModifier_COV2D_Matrix_x,
		_rotatingModifier_COV2D_Matrix_y,
		_rotatingModifier_COV2D_Matrix_z,		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		rects,
		boxmin,
		boxmax,
		antialiasing,
		t,
		enableDeformationCovariance
		);
	cudaDeviceSynchronize();

	// 측정 종료: Covariance Update
	cudaEventRecord(ev_cov_end);
	cudaEventSynchronize(ev_cov_end);

	if(cm.FPS){

		// ==========================================
		// [로그 출력 로직]
		// ==========================================
		// 상호작용이 있을 때만(isWaveRunning) 혹은 항상 측정할지 결정
		float ms_prop_gpu = 0.0f;
		float ms_cov_gpu = 0.0f;
		if (useGpuChainmail && prop_timing_valid) {
			cudaEventElapsedTime(&ms_prop_gpu, ev_prop_start, ev_prop_end);
		}
		cudaEventElapsedTime(&ms_cov_gpu, ev_cov_start, ev_cov_end);
		timeLog_Prop.push_back(useGpuChainmail ? ms_prop_gpu : cpu_prop_ms);
		timeLog_Cov.push_back(ms_cov_gpu);
		timeLog_Input.push_back(cpu_input_ms);
		measureCount++;

		// 100 프레임마다 평균 출력 (너무 자주 출력하면 느려짐)
		if (cm.FPS && measureCount >= 100) {
			float sum_prop = std::accumulate(timeLog_Prop.begin(), timeLog_Prop.end(), 0.0f);
			float sum_cov = std::accumulate(timeLog_Cov.begin(), timeLog_Cov.end(), 0.0f);
			float sum_input = std::accumulate(timeLog_Input.begin(), timeLog_Input.end(), 0.0f);
			float avg_prop = sum_prop / timeLog_Prop.size();
			float avg_cov = sum_cov / timeLog_Cov.size();
			float avg_input = sum_input / timeLog_Input.size();
			const char* propLabel = useGpuChainmail ? "ChainMail (GPU)" : "ChainMail (CPU)";

			printf("\n==============================================\n");
			printf(" [ Performance Analysis - Avg of 100 Frames ] \n");
			printf(" * Physics (%s) : %.4f ms\n", propLabel, avg_prop);//ChainMail propagation time. 물리적 복잡도
			printf(" * Covariance (GPU) : %.4f ms\n", avg_cov);//3DGS 기하학적 업데이트 시간
			if(!useGpuChainmail)printf(" * Input (CPU)       : %.4f ms\n", avg_input);//CPU에서 GPU로 데이터를 준비/전송하는 시간
			const float total_ms = avg_prop + avg_cov + avg_input;
			printf(" * Total Latency         : %.4f ms\n", total_ms);//위 세 가지를 합친 전체 파이프라인 시간. 한 프레임의 연산이 완료되는 총 시간
			printf(" * Est. FPS              : %.2f FPS", 1000.0f / total_ms);
			printf("\n==============================================\n\n");

			timeLog_Prop.clear();
			timeLog_Cov.clear();
			timeLog_Input.clear();
			measureCount = 0;
		}



	}else {
		// 버튼 껐을 때는 데이터 초기화 (찌꺼기 데이터 방지)
		if (!timeLog_Prop.empty()) {
			timeLog_Prop.clear();
			timeLog_Cov.clear();
			timeLog_Input.clear();
			measureCount = 0;
		}
	}



}
