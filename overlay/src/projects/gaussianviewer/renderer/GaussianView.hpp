/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */
#pragma once

# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Camera.hpp>
# include <core/graphics/Window.hpp>
# include <core/view/InteractiveCameraHandler.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <functional>
# include "GaussianSurfaceRenderer.hpp"
#include "nanoflann.hpp"
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <cmath>
#include "forward.h"
#include <string>
#include <fstream>
#include <iostream>
namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr { 
	// 외부에서 접근할 전역 ChainMail 인스턴스
	

	class BufferCopyRenderer;
	class BufferCopyRenderer2;
	class APGGraphBuilder;
	// 엣지 구조체
	struct Edge {
		int m_vert[2];
		float st;
		float rl;
		Edge(int v0, int v1, float restlen, float stiff = 1.0f)
			: st(stiff), rl(restlen) {
			m_vert[0] = v0; m_vert[1] = v1;
		}
	};
	typedef sibr::Vector3f Pos;
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



	////Graph parameter////


	struct APGGraphConfig {
		// ── 유사도 가중치 (기존 하드코딩 값이 default) ──────
		float w_dist = 2.9f;   // 공간 거리
		float w_ori = 0.5f;   // 방향 유사도
		float w_shape = 0.4f;   // Aspect Ratio (형태)
		float w_sh = 0.5f;   // Spherical Harmonics ← 핵심

		// ── 그래프 구조 파라미터 ────────────────────────────
		float d_thresholdP = 85.0f;  // 퍼센타일 임계값
		int   default_k = 6;

		// ── 새로 추가: 데이터셋 & 크롭박스 ──────────────────
		std::string dataset_path = "";
		std::string ply_path = "";
		std::string model_path = "";
		float crop_min[3] = { -3.097f, -0.816f, 1.129f };
		float crop_max[3] = { 0.009f,  3.390f, 4.445f };


		// ── 실험 메타데이터 (결과 파일명 자동 생성용) ────────
		std::string experiment_name = "baseline";
		std::string dataset_name = "unknown";

		// ── JSON 로드 ────────────────────────────────────────
		static APGGraphConfig fromJSON(const std::string& path) {
			APGGraphConfig cfg;
			std::ifstream f(path);
			if (!f.is_open()) {
				std::cerr << "[Config] JSON 없음, 기본값 사용: " << path << std::endl;
				return cfg;
			}

			// ── content로 전체 읽기 (getline 방식 제거) ──────────────────
			std::string content((std::istreambuf_iterator<char>(f)),
				std::istreambuf_iterator<char>());
			f.close();

			// ── 파서 람다 (content 기반으로 통일) ────────────────────────
			auto parseFloat = [&](const std::string& key, float& out) {
				auto pos = content.find("\"" + key + "\"");
				if (pos == std::string::npos) return;
				auto colon = content.find(':', pos);
				if (colon == std::string::npos) return;
				try { out = std::stof(content.substr(colon + 1)); }
				catch (...) {}
			};
			auto parseInt = [&](const std::string& key, int& out) {
				auto pos = content.find("\"" + key + "\"");
				if (pos == std::string::npos) return;
				auto colon = content.find(':', pos);
				if (colon == std::string::npos) return;
				try { out = std::stoi(content.substr(colon + 1)); }
				catch (...) {}
			};
			auto parseStr = [&](const std::string& key, std::string& out) {
				auto pos = content.find("\"" + key + "\"");
				if (pos == std::string::npos) return;
				auto colon = content.find(':', pos);
				auto q1 = content.find('"', colon + 1);
				auto q2 = content.find('"', q1 + 1);
				if (q1 == std::string::npos || q2 == std::string::npos) return;
				out = content.substr(q1 + 1, q2 - q1 - 1);
			};
			// [x, y, z] 배열 파싱
			auto parseVec3 = [&](const std::string& key, float out[3]) {
				auto pos = content.find("\"" + key + "\"");
				if (pos == std::string::npos) return;
				auto bracket = content.find('[', pos);
				auto end = content.find(']', bracket);
				if (bracket == std::string::npos || end == std::string::npos) return;
				std::string arr = content.substr(bracket + 1, end - bracket - 1);
				std::istringstream ss(arr);
				std::string token;
				int idx = 0;
				while (std::getline(ss, token, ',') && idx < 3) {
					try { out[idx++] = std::stof(token); }
					catch (...) {}
				}
			};

			// ── 실제 파싱 (content 전체에서) ─────────────────────────────
			parseFloat("w_dist", cfg.w_dist);
			parseFloat("w_ori", cfg.w_ori);
			parseFloat("w_shape", cfg.w_shape);
			parseFloat("w_sh", cfg.w_sh);
			parseFloat("d_thresholdP", cfg.d_thresholdP);
			parseInt("default_k", cfg.default_k);
			parseStr("experiment_name", cfg.experiment_name);
			parseStr("dataset_name", cfg.dataset_name);
			parseStr("model_path", cfg.model_path);    // ← 추가
			parseStr("dataset_path", cfg.dataset_path);  // ← 추가
			parseVec3("crop_min", cfg.crop_min);      // ← 추가
			parseVec3("crop_max", cfg.crop_max);      // ← 추가

			std::cout << "[Config] 로드 완료: " << path
				<< " | w_dist=" << cfg.w_dist
				<< " w_sh=" << cfg.w_sh
				<< " | crop_min=["
				<< cfg.crop_min[0] << "," << cfg.crop_min[1] << "," << cfg.crop_min[2]
				<< "]" << std::endl;
			return cfg;
		}

		void print() const {
			printf("[APGGraphConfig] exp=%s | w_dist=%.2f w_ori=%.2f "
				"w_shape=%.2f w_sh=%.2f | k=%d threshold=%.1f%%\n",
				experiment_name.c_str(),
				w_dist, w_ori, w_shape, w_sh, default_k, d_thresholdP);
			printf("[APGGraphConfig] crop_min=[%.3f, %.3f, %.3f] crop_max=[%.3f, %.3f, %.3f]\n",
				crop_min[0], crop_min[1], crop_min[2],
				crop_max[0], crop_max[1], crop_max[2]);
		}
	};


	////Graph parameter////



	/**
	 * \class RemotePointView
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 */
		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* message_read, int sh_degree, const sibr::Window& window, bool white_bg = false, bool useInterop = true, int device = 0,
			const APGGraphConfig& cfg = APGGraphConfig());
		void buildConstraintGraph(int k  ); // 제약 그래프 생성 함수

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr & newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input) override;
		void onUpdate(Input& input, const Viewport& vp) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;
		void setCameraHandler(const sibr::InteractiveCameraHandler::Ptr& handler);
		void rebuildGraphDebugEdges();
		void updateGraphDebugPositions();
		void drawGraphDebugWindow();

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

		virtual ~GaussianView() override;

		bool* _dontshow;

	
	protected:
		sibr::Window* _window; // 윈도우 객체를 가리킬 포인터
		bool _vsyncEnabled;    // GUI 체크박스 상태 저장용
		std::string currMode = "Splats";
		bool _antialiasing = false;
		bool _cropping = false;

		bool _wave  = false;
		bool _twist  = false;
		bool _bubble = false;

		sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
		char _buff[512] = "cropped.ply";

		bool _fastCulling = true;
		int _device = 0;
		int _sh_degree = 3;

		int count;
		std::vector<Pos> pos;
		std::vector<Rot> rot;
		std::vector<Scale> scale;
		std::vector<float> opacity;
		std::vector<SHs<3>> shs;
		std::vector<Pos> cropped_pos;
		std::vector<Rot> cropped_rot;
		std::vector<Scale> cropped_scale;
		std::vector<SHs<3>> cropped_shs;
		std::vector<float> cropped_opacity;
		float _radius;
		glm::vec3  _pivot;
		std::vector<int> idx_map; // 원본 인덱스 → 크롭 인덱스
		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;
		int* rect_cuda;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;
		int* _idBufferCuda = nullptr;//picking 상태 
		size_t _idBufferCount = 0;//picking 드래그 상태용 변수

		float _scalingModifier = 1.0f;
		float _rotatingModifier = 1.0f;
		float _rotatingModifier_COV3D_Matrix_x = 1.0f;
		float _rotatingModifier_COV3D_Matrix_y = 1.0f;
		float _rotatingModifier_COV3D_Matrix_z = 1.0f;
		float _rotatingModifier_COV2D_Matrix_x = 1.0f;
		float _rotatingModifier_COV2D_Matrix_y = 1.0f;
		float _rotatingModifier_COV2D_Matrix_z = 1.0f;
		int _deformPoint1 = 44986;//data1 44986;
		int _deformseeds = 5000;//data1 44986;

		int _deformPoint2 = 180063;
		int _potinSize;
		float _pivotRotX = 1.0f;
		float _pivotRotY = 1.0f;
		float _pivotRotZ = 1.0f;
		GaussianData* gData;
		float d_thresholdP = 0.0f;
		int default_k = 0;
		// 기존 멤버 변수
		std::vector<sibr::Edge> cropped_edges;
		std::vector<FORWARD::Edge> FORWARD_cropped_edges;

		// ====== Element 분리 ======
		std::vector<glm::vec3> elem_pos;
		std::vector<float> elem_density;
		std::vector<float> elem_time;
		std::vector<int> elem_offset;
		std::vector<int> elem_neighborCnt;
		// ====== Neighbor 분리 ======
		std::vector<int> neighbor_idx;
		std::vector<float> neighbor_dist;
		std::unique_ptr<APGGraphBuilder> _graphBuilder;

		std::vector<Edge> edges; // CPU 측 제약 리스트
		Edge* d_edges;           // GPU 측 제약 리스트 포인터
		int num_edges;           // 엣지 개수

		FORWARD::ChainMail _cm;        // ChainMail 객체
		std::vector<int> _activeSet;   // ChainMail 활성 요소 목록
		bool _showGraphDebugWindow = true;
		// 헤더에 추가
		float _graphDebugViewX = 0.0f;    // 화면 왼쪽에서 픽셀 오프셋
		float _graphDebugViewY = 0.0f;    // 화면 위에서 픽셀 오프셋  
		float _graphDebugViewW = 800.0f;  // 가시화 영역 너비
		float _graphDebugViewH = 600.0f;  // 가시화 영역 높이
		float _graphDebugPointRadius = 1.5f;
		float _graphDebugZoom = 1.0f;
		float _graphDebugEdgeAlpha = 0.001f;
		float _graphDebugLineThickness = 1.0f;
		float _graphDebugMinEdgePixel = 0.0f;
		float _graphDebugUpdateIntervalSec = 0.0f;
		double _graphDebugLastUpdateSec = -1.0;
		std::vector<float> _graphDebugXYZ;
		std::vector<float> _graphDebugScreenXY;
		std::vector<int> _graphDebugRadii;
		int _graphDebugRenderW = 0;
		int _graphDebugRenderH = 0;
		std::vector<sibr::Vector3f> _graphDebugPositions;
		std::vector<std::pair<int, int>> _graphDebugEdges;

		// Graph Debug OpenGL resources
		GLuint _graphDebugVAO = 0;
		GLuint _graphDebugVBO = 0;       // 노드 위치 (2D screen xy)
		GLuint _graphDebugEBO = 0;       // 엣지 인덱스
		GLuint _graphDebugShader = 0;
		bool   _graphDebugGLReady = false;

		// 컴파일된 2D projected positions (float x,y per node)
		std::vector<float> _graphDebugProjXY;  // size = N*2
		void initGraphDebugGL();
		void renderGraphDebugGL();
		// Graph debug canvas 좌표 (onGUI → onRenderIBR 전달용)
		float _graphDebugDrawX = 0.f;
		float _graphDebugDrawY = 0.f;
		float _graphDebugDrawW = 0.f;
		float _graphDebugDrawH = 0.f;
		bool  _graphDebugCanvasReady = false;

		// 새 함수 선언
		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;
		bool _useActiveMap = false;
		int _cpugpuMode = 0;

		int _cmPropIters = 20;
		int _cmRelaxIters = 5;
		float _cmPropStrength = 0.5f;
		float _cmStiffness = 0.99f;
		float _cmDamping = 0.1f;
		float _cmConstraintGlobalScale = 1.0f;
		float _cmConstraintAirScale = 0.3f;
		float _cmConstraintSkinScale = 0.06f;
		float _cmConstraintBoneScale = 0.04f;
		bool _cmUseEdgeStiffness = true;
		float _cmEdgeStiffnessInfluence = 0.5f;
		float _cmInertiaGain = 0.0f;
		float _cmVelocityRetention = 0.85f;
		float _cmVelocityClamp = 0.05f;




		//////////////////////////picking//////////////////////
		bool _pickingMode = false;
		bool _pickDragActive = false;
		bool _pendingPick = false;
		bool _dragStarted = false;
		bool _hasLastEye = false;
		bool _pickedWorldPosValid = false;
		int _pickedId = -1;
		float _pickDragScale = 1.0f;
		int _pickbfsHopsScale = 2;
		bool _keepCameraMotionWhilePicking = true;
		sibr::Vector2i _pendingPickPixel = { 0, 0 };
		sibr::Vector2i _lastPickMouse = { 0, 0 };
		sibr::Vector2i _pickInputViewportSize = { 1, 1 };
		sibr::Vector3f _pickedWorldPos = { 0.f, 0.f, 0.f };
		sibr::Camera _lastEye;
		sibr::Matrix4f _graphDebugProjMat = sibr::Matrix4f::Identity();
		bool _graphDebugProjValid = false;
		sibr::InteractiveCameraHandler::Ptr _cameraHandler;
		//////////////////////////picking//////////////////////


		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;
	};


	
} /*namespace sibr*/ 
//// ===== 자료구조 =====
//constexpr float AIR = 0.13f;
//constexpr float SKIN = 0.35f;
//constexpr float BONE = 0.55f;
//struct Vec3 {
//	float x, y, z;
//	Vec3() : x(0), y(0), z(0) {}
//	Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
//
//	Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
//	Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
//	Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
//
//	float length() const { return std::sqrt(x * x + y * y + z * z); }
//	Vec3 normalized() const {
//		float l = length();
//		if (l < 1e-6f) return Vec3(0, 0, 0);
//		return Vec3(x / l, y / l, z / l);
//	}
//};
//
//struct CMConstraint {
//	float dx, dy, dz;
//	float xShearY, xShearZ;       // X방향 위치에 대한 Y, Z축 전단
//	float yShearX, yShearZ;       // Y방향 위치에 대한 X, Z축 전단
//	float zShearX, zShearY;       // Z방향 위치에 대한 X, Y축 전단
//	CMConstraint()
//		: dx(0.01f), dy(0.01f), dz(0.01f),
//		xShearY(0.01f), xShearZ(0.01f),
//		yShearX(0.01f), yShearZ(0.01f),
//		zShearX(0.01f), zShearY(0.01f)
//	{}
//	CMConstraint(float dx, float dy, float dz,
//		float xSy, float xSz,
//		float ySx, float ySz,
//		float zSx, float zSy)
//		: dx(dx), dy(dy), dz(dz),
//		xShearY(xSy), xShearZ(xSz),
//		yShearX(ySx), yShearZ(ySz),
//		zShearX(zSx), zShearY(zSy) {}
//};
//
//struct Neighbor {
//	int idx;    // neighbor 정점 인덱스
//	float dist; // 거리
//	Neighbor() : idx(-1), dist(0) {}
//	Neighbor(int idx, float dist) : idx(idx), dist(dist) {}
//};
//
//struct Element {
//	glm::vec3 pos;
//	float density;
//	float time;
//	int offset;      // neighbor 배열 시작 인덱스
//	int neighborCnt;
//	// 추가 파라미터들 필요시 여기에
//	Element()
//		: pos(), density(0), time(1e9f), offset(0), neighborCnt(0) {}
//};
//struct cEdge {
//	int v1, v2;
//	float dist;
//};
//class  ChainMail {
//public:
//	bool loadGraph(
//		ChainMail& cm,
//		const std::vector<sibr::Pos>& cropped_pos,
//		const std::vector<sibr::Edge>& cropped_edges,
//		const std::vector<float>& cropped_opacity);
//	bool saveGraph(const std::string& filename);
//	void resetTime();
//	void movePointPos(int idx, const glm::vec3& dpos);
//	void propagate(std::vector<int>& activeSet); // propagated && moved 기준
//	void relax(const std::vector<int>& activeSet);
//	size_t numElements() const;
//	const Element& getElement(int i) const;
//	Element& getElement(int i);
//	const std::vector<cEdge>& getEdges() const;
//
//	std::vector<Element> elements;
//	std::vector<Neighbor> neighbors;
//	std::vector<cEdge> cedges;
//
//	
//private:
//	float propagationTime(const Element& e, const Element& n);
//	void shiftElementPoint(Element& elem, const Element& n, float targDist, bool& moved);
//	// 데이터
//	
//};
//extern ChainMail g_chainmail;
//extern bool      g_chainmail_ready;
//void SetChainMailGraphFromVectors(
//	const std::vector<sibr::Pos>& cropped_pos,
//	const std::vector<sibr::Edge>& cropped_edges,
//	const std::vector<float>& cropped_opacity
//);
