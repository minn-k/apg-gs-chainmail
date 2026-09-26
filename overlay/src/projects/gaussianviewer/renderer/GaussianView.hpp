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
		float w_dist = 2.9f;
		float w_ori = 0.5f;
		float w_shape = 0.4f;
		float w_sh = 0.5f;
		float edge_percentile = 85.0f;
		int candidate_neighbors = 6;

		static APGGraphConfig fromJSON(const std::string& path) {
			APGGraphConfig cfg;
			std::ifstream file(path);
			if (!file.is_open()) {
				std::cerr << "[APGGraphConfig] Could not open " << path << "; using defaults.\n";
				return cfg;
			}

			const std::string content((std::istreambuf_iterator<char>(file)),
				std::istreambuf_iterator<char>());
			auto parseFloat = [&](const std::string& key, float& value) {
				const auto keyPos = content.find("\"" + key + "\"");
				if (keyPos == std::string::npos) return;
				const auto colon = content.find(':', keyPos);
				if (colon == std::string::npos) return;
				try { value = std::stof(content.substr(colon + 1)); }
				catch (...) {}
			};
			auto parseInt = [&](const std::string& key, int& value) {
				const auto keyPos = content.find("\"" + key + "\"");
				if (keyPos == std::string::npos) return;
				const auto colon = content.find(':', keyPos);
				if (colon == std::string::npos) return;
				try { value = std::stoi(content.substr(colon + 1)); }
				catch (...) {}
			};

			parseFloat("w_dist", cfg.w_dist);
			parseFloat("w_ori", cfg.w_ori);
			parseFloat("w_shape", cfg.w_shape);
			parseFloat("w_sh", cfg.w_sh);
			parseFloat("edge_percentile", cfg.edge_percentile);
			parseInt("candidate_neighbors", cfg.candidate_neighbors);
			std::cout << "[APGGraphConfig] Loaded " << path << "\n";
			return cfg;
		}

		void print() const {
			std::cout << "[APGGraphConfig] w_dist=" << w_dist
				<< " w_ori=" << w_ori
				<< " w_shape=" << w_shape
				<< " w_sh=" << w_sh
				<< " candidates=" << candidate_neighbors
				<< " percentile=" << edge_percentile << "%\n";
		}
	};




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

		bool _deformationAwareCovariance = true;

		sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
		char _buff[512] = "deformed_gaussians.ply";

		bool _fastCulling = true;
		int _device = 0;
		int _sh_degree = 3;

		int count;
		std::vector<Pos> pos;
		std::vector<Rot> rot;
		std::vector<Scale> scale;
		std::vector<float> opacity;
		std::vector<SHs<3>> shs;
		float* pos_cuda = nullptr;
		float* rot_cuda = nullptr;
		float* scale_cuda = nullptr;
		float* opacity_cuda = nullptr;
		float* shs_cuda = nullptr;
		int* rect_cuda = nullptr;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda = nullptr;
		float* proj_cuda = nullptr;
		float* cam_pos_cuda = nullptr;
		float* background_cuda = nullptr;
		int* _idBufferCuda = nullptr;//picking 상태
		size_t _idBufferCount = 0;//picking 드래그 상태용 변수

		float _scalingModifier = 1.0f;
		float _rotatingModifier_COV2D_Matrix_x = 1.0f;
		float _rotatingModifier_COV2D_Matrix_y = 1.0f;
		float _rotatingModifier_COV2D_Matrix_z = 1.0f;
		int _pointCount = 0;
		int _neighborCapacity = 6;
		GaussianData* gData = nullptr;
		std::unique_ptr<APGGraphBuilder> _graphBuilder;
		FORWARD::ChainMail _cm;
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



} // namespace sibr
