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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED
#include <cuda.h>
#include <vector>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	#pragma once


	// ===== ?먮즺援ъ“ =====
	struct Edge {
		int m_vert[2];
		float st;
		float rl;
		Edge(int v0, int v1, float restlen, float stiff = 1.0f)
			: st(stiff), rl(restlen) {
			m_vert[0] = v0; m_vert[1] = v1;
		}
	};

	typedef glm::vec3 Pos;

	template<int D>
	struct SHs
	{
		float shs[(D + 1) * (D + 1) * 3];
	};
	struct Scale
	{
		float scale[3];
	};
	struct Rot
	{
		float rot[4];
		float& operator[](int idx) { return rot[idx]; }
		float operator[](int idx) const { return rot[idx]; }
	};
	// GaussianView.cpp?먯꽌 ?щ∼+洹몃옒???앹꽦(踰≫꽣)???앸궡怨???踰덈쭔 ?몄텧
	/*void SetChainMailGraphFromVectors(
		const std::vector<Pos>& cropped_pos,
		const std::vector<Edge>& cropped_edges,
		const std::vector<float>& cropped_opacity
	);*/


	constexpr float AIR = 0.13f;
	constexpr float SKIN = 0.45f;
	constexpr float BONE = 0.75f;

	struct Vec3 {
		float x, y, z;
		Vec3() : x(0), y(0), z(0) {}
		Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

		Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
		Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
		Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

		float length() const { return std::sqrt(x * x + y * y + z * z); }
		Vec3 normalized() const {
			float l = length();
			if (l < 1e-6f) return Vec3(0, 0, 0);
			return Vec3(x / l, y / l, z / l);
		}
	};

	struct CMConstraint {
		float dx, dy, dz;
		float xShearY, xShearZ;       // X諛⑺뼢 ?꾩튂?????Y, Z異??꾨떒
		float yShearX, yShearZ;       // Y諛⑺뼢 ?꾩튂?????X, Z異??꾨떒
		float zShearX, zShearY;       // Z諛⑺뼢 ?꾩튂?????X, Y異??꾨떒
		CMConstraint()
			: dx(0.01f), dy(0.01f), dz(0.01f),
			xShearY(0.01f), xShearZ(0.01f),
			yShearX(0.01f), yShearZ(0.01f),
			zShearX(0.01f), zShearY(0.01f)
		{}
		CMConstraint(float dx, float dy, float dz,
			float xSy, float xSz,
			float ySx, float ySz,
			float zSx, float zSy)
			: dx(dx), dy(dy), dz(dz),
			xShearY(xSy), xShearZ(xSz),
			yShearX(ySx), yShearZ(ySz),
			zShearX(zSx), zShearY(zSy) {}
	};
	struct float3x3 {
		float m[9]; // ?됰젹 ?먯냼瑜????곗꽑(row-major) ?먮뒗 ???곗꽑(column-major)?쇰줈 ??? ?먰븯??????쒖꽌??留욎떠 ?ъ슜
		__host__ __device__ float& operator()(int row, int col) { return m[row * 3 + col]; }
		__host__ __device__ float operator()(int row, int col) const { return m[row * 3 + col]; }
	};

	struct Neighbor {
		int idx;    // neighbor index
		float dist; // rest distance
		float st; // ?댁썐 媛?affinity
		Neighbor() : idx(-1), dist(0), st(0){}
		Neighbor(int idx, float dist, float st) : idx(idx), dist(dist), st(st) {}
	};

	struct Element {
		glm::vec3 pos;
		glm::vec3 vel;
		float invMass;
		float density;
		float time;
		int offset;      // neighbor array start index
		int neighborCnt; // number of neighbors	
		Element(): pos(), vel(0.0f), invMass(1.0f), density(0), time(1e9f), offset(0), neighborCnt(0) {}
	};
	struct cEdge {
		int v1, v2;
		float dist;
	};
	struct SeedGroup {
		std::string name;          // 援щ텇???대쫫 (?? "LeftEar", "RightEar")
		std::vector<int> indices;  // ?먮뱾???몃뜳??紐⑸줉
		glm::vec3 pivot;           // ??洹몃９???뚯쟾 以묒떖??(留ㅼ슦 以묒슂!)

		// ?앹꽦??
		SeedGroup(std::string n, std::vector<int> idxs, glm::vec3 p)
			: name(n), indices(idxs), pivot(p) {}
	};
	
	class  ChainMail {
	public:
		// ChainMail ?대옒???대? (?먮뒗 ?ㅻ뜑)??異붽???蹂?섎뱾
		struct DeformTask {
			bool isRunning = false;
			int seedIdx = -1;
			glm::vec3 startPos;
			glm::vec3 targetPos;
			float progress = 0.0f;     // 0.0 ~ 1.0 (吏꾪뻾瑜?
			float speed = 2.0f;        // 1珥덉뿉 ?대룞??鍮꾩쑉 (2.0?대㈃ 0.5珥덈쭔???꾨떖)
		};

		DeformTask singleDeformTask; // 硫ㅻ쾭 蹂?섎줈 ?좎뼵
		bool FPS = false;
		bool loadGraph(
			ChainMail& cm,
			const std::vector<Pos>& cropped_pos,
			const std::vector<Edge>& cropped_edges,
			const std::vector<float>& cropped_opacity);
		void resetTime();
		void movePointPos(int* idx, const glm::vec3& dpos, std::vector<int>& activeSet);
		//void setPointPos(int idx, const glm::vec3& targetPos, std::vector<int>& activeSet);
		int findCheekPoint(FORWARD::ChainMail&);

		void propagate(std::vector<int>& activeSet); // propagated && moved 湲곗?
		void propagateStep(const std::vector<int>& currentFrontier, std::vector<int>& nextFrontier, std::vector<int>& totalActiveSet);
		void startWave(int seed, glm::vec3 delta, std::vector<int>& activeSet);
		void startWaveMultiple(const std::vector<int>& seeds, glm::vec3 delta, std::vector<int>& activeSet);
		void startWavingMultiple(
			const std::vector<int>& seeds,
			glm::vec3 delta,
			std::vector<int>& activeSet);
		std::vector<int> collectSeedsBFS(int startNode, int targetCount);
		void applyWaveOffset(const glm::vec3& delta, std::vector<int>& activeSet);

		void B_relax(const std::vector<int>& activeSet);
		void Stabilize(const std::vector<int>& activeSet);
		void relax(const std::vector<int>& activeSet);
		void relax2(const std::vector<int>& activeSet);

		size_t numElements() const;
		const Element& getElement(int i) const;
		Element& getElement(int i);
		const Neighbor& getNeighbor(int i) const;
		Neighbor& getNeighbor(int i);
		CMConstraint getConstraint(float);
		const std::vector<cEdge>& getEdges() const;
		std::vector<Edge> cropped_edges;
		float d_thresholdP = 0.0f;
		bool isWaveRunning() const { return waveRunning; }
	

		std::vector<int> getSeeds() const { return seeds; }
		bool getRunning() const { return waveRunning; }

		void setSeeds(const std::vector<int>& s) {
			seeds = s;
		}
		void setRunning(bool wR) {
			waveRunning = wR;
		}
		float _deformingPoint1;
		const std::vector<SeedGroup>& getSeedGroups() const { return seedGroups; }

		// [?섏젙] 洹몃９ 異붽? ?⑥닔 (洹 1媛?異붽????뚮쭏???몄텧)
		void addSeedGroup(const std::string& name, const std::vector<int>& indices, glm::vec3 pivot) {
			seedGroups.emplace_back(name, indices, pivot);
		}

		// [?섏젙] 珥덇린??
		void clearSeedGroups() {
			seedGroups.clear();
		}

	private:
		float propagationTime(const Element& e, const Element& n);
		void shiftElementPoint(Element& elem, const Element& n, float targDist, bool& moved);
		// ?곗씠??
		std::vector<Element> elements;
		std::vector<Neighbor> neighbors;
		std::vector<cEdge> cedges;
		std::vector<int> waveActive;
		std::vector<int> seeds;
		std::vector<SeedGroup> seedGroups; // <-- ?닿구濡??泥?
		bool waveRunning = false;


	};


	extern __constant__ float mytime[1];
	// Perform initial steps for each Gaussian prior to rasterization.

	void setChainmailActiveMapEnabled(bool enabled);
	float getChainmailActiveRatio();
	int getChainmailActiveCount();
	void setGpuChainmailMode(int mode); // 0: cpu, 1: gpu
	int getGpuChainmailMode();
	void setPickingParams(int hops);
	void getPickingParams(int* hops);
	bool copyCurrentDeformedPositions(float* outXYZ, int pointCount);
	bool copyCurrentScreenProjection(
		float* outXY,
		int* outRadii,
		int pointCount,
		int* outRenderW,
		int* outRenderH);

	void setChainmailParams(
		int propIters,
		int relaxIters,
		float propStrength,
		float stiffness,
		float damping);
	void getChainmailParams(
		int* propIters,
		int* relaxIters,
		float* propStrength,
		float* stiffness,
		float* damping);
	void setChainmailMaterialParams(
		float constraintGlobalScale,
		float airScale,
		float skinScale,
		float boneScale,
		bool useEdgeStiffness,
		float edgeStiffnessInfluence);
	void getChainmailMaterialParams(
		float* constraintGlobalScale,
		float* airScale,
		float* skinScale,
		float* boneScale,
		bool* useEdgeStiffness,
		float* edgeStiffnessInfluence);
	void setChainmailDynamicsParams(
		float inertiaGain,
		float velocityRetention,
		float velocityClamp);
	void getChainmailDynamicsParams(
		float* inertiaGain,
		float* velocityRetention,
		float* velocityClamp);
	void preprocess(
		ChainMail& cm, std::vector<int>& activeSet,

		int P, int D, int M,
		 float* orig_points,
		int nbr_K,

		const glm::vec3* scales,
		const float scale_modifier,



		const float _rotatingModifier_COV3D_Matrix_x,
		const float _rotatingModifier_COV3D_Matrix_y,
		const float _rotatingModifier_COV3D_Matrix_z,
		const float _rotatingModifier_COV2D_Matrix_x,
		const float _rotatingModifier_COV2D_Matrix_y,
		const float _rotatingModifier_COV2D_Matrix_z,


		const float _pivotRotX,
		const float _pivotRotY,
		const float _pivotRotZ,


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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		int2* rects,
		float3 boxmin,
		float3 boxmax,
		bool antialiasing,
		float t,
		bool _wave,
		bool _twist,
		bool _bubble);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		int* id_buffer);

	
	// ?몃??먯꽌 ?묎렐???꾩뿭 ChainMail ?몄뒪?댁뒪
	/*extern ChainMail g_chainmail;
	extern bool      g_chainmail_ready;*/
}


#endif


