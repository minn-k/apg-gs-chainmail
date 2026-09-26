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

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <core/view/InteractiveCameraHandler.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>
#include "APGGraphBuilder.hpp"
#include <glm/gtc/quaternion.hpp>

#include <vector>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>
 // Define the types and sizes that make up the contents of each Gaussian 
 // in the trained model.


template<int D>
struct RichPoint
{
	sibr::Pos pos;
	float n[3];
	sibr::SHs<D> shs;
	float opacity;
	sibr::Scale scale;
	sibr::Rot rot;
};
float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<sibr::Pos>& pos,
	std::vector<sibr::SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<sibr::Scale>& scales,
	std::vector<sibr::Rot>& rot,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);
		
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}
	return count;
}

void savePly(const char* filename,
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::SHs<3>>& shs,
	const std::vector<float>& opacities,
	const std::vector<sibr::Scale>& scales,
	const std::vector<sibr::Rot>& rot,
	const sibr::Vector3f& minn,
	const sibr::Vector3f& maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		count++;
	}
	std::vector<RichPoint<3>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2" };
	std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}



namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			// Keep shader-side indexing aligned with the current render size.
			_width.get() = width;
			_height.get() = height;

			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader;
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, const sibr::Window& window, bool white_bg, bool useInterop, int device, const APGGraphConfig& cfg) :
	_scene(ibrScene),
	_dontshow(messageRead),
	_sh_degree(sh_degree),
	_graphBuilder(std::make_unique<APGGraphBuilder>()), _window(const_cast<sibr::Window*>(&window)),
	sibr::ViewBase(render_w, render_h)
{
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}
	_vsyncEnabled = _window->isVsynced(); // 현재 윈도우의 VSync 상태 가져오기
	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;
	_cpugpuMode = FORWARD::getGpuChainmailMode();
	FORWARD::getChainmailParams(
		&_cmPropIters,
		&_cmRelaxIters,
		&_cmPropStrength,
		&_cmStiffness,
		&_cmDamping);
	FORWARD::getChainmailMaterialParams(
		&_cmConstraintGlobalScale,
		&_cmConstraintAirScale,
		&_cmConstraintSkinScale,
		&_cmConstraintBoneScale,
		&_cmUseEdgeStiffness,
		&_cmEdgeStiffnessInfluence);
	FORWARD::getChainmailDynamicsParams(
		&_cmInertiaGain,
		&_cmVelocityRetention,
		&_cmVelocityClamp);
	std::vector<uint> imgs_ulr;
	const auto& cams = ibrScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	// Load the PLY data (AoS) to the GPU (SoA)
	
	if (sh_degree == 0)
	{
		count = loadPly<0>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 1)
	{
		count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 2)
	{
		count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 3)
	{
		count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}

	_boxmin = _scenemin;
	_boxmax = _scenemax;

	int P = count;
	cfg.print();

	std::vector<FORWARD::Pos> graphPositions;
	graphPositions.reserve(pos.size());
	for (const auto& point : pos) {
		graphPositions.emplace_back(point.x(), point.y(), point.z());
	}

	std::vector<FORWARD::Edge> graphEdges;
	_graphBuilder->build(pos, rot, scale, graphEdges, shs, cfg, true);
	_pointCount = static_cast<int>(pos.size());
	_neighborCapacity = std::max(1, cfg.candidate_neighbors);
	_cm.loadGraph(_cm, graphPositions, graphEdges, opacity);
	rebuildGraphDebugEdges();

	P = _pointCount;
	count = P;

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(P,
		(float*)pos.data(),
		(float*)rot.data(),
		(float*)scale.data(),
		opacity.data(),
		(float*)shs.data());

	_gaussianRenderer = new GaussianSurfaceRenderer();

	//chainmail




	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop)
	{
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
}



void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr& newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto& cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::setCameraHandler(const sibr::InteractiveCameraHandler::Ptr& handler)
{
	_cameraHandler = handler;
	if (_cameraHandler)
	{
		_cameraHandler->setInputEnabled(!_pickingMode || _keepCameraMotionWhilePicking);
	}
}
//__constant__ float mytime[1];
//#include "forward.h" // extern 선언 포함
static float t = 0.0f;

void sibr::GaussianView::rebuildGraphDebugEdges()
{
	_graphDebugEdges.clear();
	const auto& cmEdges = _cm.getEdges();
	if (cmEdges.empty()) {
		return;
	}
	const int N = static_cast<int>(_cm.numElements());
	_graphDebugEdges.reserve(cmEdges.size());
	for (const auto& e : cmEdges) {
		const int a = e.v1;
		const int b = e.v2;
		if (a < 0 || b < 0 || a >= N || b >= N || a == b) continue;
		_graphDebugEdges.emplace_back(a, b);
	}
}

void sibr::GaussianView::updateGraphDebugPositions()
{
	if (count <= 0) {
		return;
	}

	const auto now = std::chrono::steady_clock::now();
	const double nowSec = std::chrono::duration<double>(now.time_since_epoch()).count();
	if (_graphDebugLastUpdateSec >= 0.0 &&
		(nowSec - _graphDebugLastUpdateSec) < static_cast<double>(_graphDebugUpdateIntervalSec)) {
		return;
	}
	_graphDebugLastUpdateSec = nowSec;

	_graphDebugScreenXY.resize(static_cast<size_t>(count) * 2ull);
	_graphDebugRadii.resize(static_cast<size_t>(count));
	if (!FORWARD::copyCurrentScreenProjection(
		_graphDebugScreenXY.data(),
		_graphDebugRadii.data(),
		count,
		&_graphDebugRenderW,
		&_graphDebugRenderH)) {
		return;
	}
}
static GLuint compileGraphDebugShader()
{
	const char* vsrc = R"(
        #version 330 core
        layout(location = 0) in vec2 pos;
        uniform float uPointSize;
        void main() {
            // pos는 이미 NDC (-1~1)로 변환된 값
            gl_Position = vec4(pos, 0.0, 1.0);
            gl_PointSize = uPointSize;
        }
    )";
	const char* fsrcEdge = R"(
        #version 330 core
        out vec4 fragColor;
        uniform vec4 uColor;
        void main() { fragColor = uColor; }
    )";

	auto compile = [](GLenum type, const char* src) -> GLuint {
		GLuint s = glCreateShader(type);
		glShaderSource(s, 1, &src, nullptr);
		glCompileShader(s);
		return s;
	};

	GLuint vs = compile(GL_VERTEX_SHADER, vsrc);
	GLuint fs = compile(GL_FRAGMENT_SHADER, fsrcEdge);
	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	glDeleteShader(vs);
	glDeleteShader(fs);
	return prog;
}

void sibr::GaussianView::initGraphDebugGL()
{
	if (_graphDebugGLReady) return;

	_graphDebugShader = compileGraphDebugShader();

	glGenVertexArrays(1, &_graphDebugVAO);
	glGenBuffers(1, &_graphDebugVBO);
	glGenBuffers(1, &_graphDebugEBO);

	glBindVertexArray(_graphDebugVAO);
	glBindBuffer(GL_ARRAY_BUFFER, _graphDebugVBO);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	_graphDebugGLReady = true;
}


void sibr::GaussianView::drawGraphDebugWindow()
{
	if (!_showGraphDebugWindow) return;

	ImGuiIO& io = ImGui::GetIO();
	const float screenW = io.DisplaySize.x;
	const float screenH = io.DisplaySize.y;

	ImGui::SetNextWindowSize(ImVec2(300.0f, 280.0f), ImGuiCond_Always);
	if (!ImGui::Begin("Graph Debug", &_showGraphDebugWindow, ImGuiWindowFlags_NoResize)) {
		ImGui::End();
		return;
	}

	ImGui::Text("Nodes: %d, Edges: %d", count, static_cast<int>(_graphDebugEdges.size()));
	ImGui::SliderFloat("Line Alpha", &_graphDebugEdgeAlpha, 0.005f, 1.0f, "%.3f");
	ImGui::SliderFloat("Point Size", &_graphDebugPointRadius, 0.5f, 5.0f, "%.2f");
	ImGui::SliderFloat("Line Size", &_graphDebugLineThickness, 0.5f, 5.0f, "%.2f");

	ImGui::SliderFloat("Zoom", &_graphDebugZoom, 0.5f, 4.0f, "%.2f");
	ImGui::Separator();
	ImGui::Text("Graph Viewport (screen pixels)");
	ImGui::SliderFloat("X", &_graphDebugViewX, 0.0f, screenW - 100.0f, "%.0f");
	ImGui::SliderFloat("Y", &_graphDebugViewY, 0.0f, screenH - 100.0f, "%.0f");
	ImGui::SliderFloat("Width", &_graphDebugViewW, 100.0f, screenW, "%.0f");
	ImGui::SliderFloat("Height", &_graphDebugViewH, 100.0f, screenH, "%.0f");
	ImGui::End();

	if (_hasLastEye) {
		updateGraphDebugPositions();
	}

	initGraphDebugGL();

	const int N = count;
	if (N <= 0) return;
	if (_graphDebugScreenXY.empty() || _graphDebugRenderW <= 0 || _graphDebugRenderH <= 0) return;
	if (screenW <= 0.0f || screenH <= 0.0f) return;

	// ── 1. 좌표 변환 ─────────────────────────────────────────────
	const float zoom = std::max(0.1f, _graphDebugZoom);
	_graphDebugProjXY.assign(static_cast<size_t>(N) * 2, 0.0f);
	std::vector<unsigned char> visible(static_cast<size_t>(N), 0);

	for (int i = 0; i < N; ++i) {
		if (!_graphDebugRadii.empty() && _graphDebugRadii[static_cast<size_t>(i)] <= 0)
			continue;
		const size_t base = static_cast<size_t>(i) * 2ull;
		const float px = _graphDebugScreenXY[base + 0];
		const float py = _graphDebugScreenXY[base + 1];
		if (!std::isfinite(px) || !std::isfinite(py)) continue;
		if (px < 0.0f || py < 0.0f ||
			px >= float(_graphDebugRenderW) || py >= float(_graphDebugRenderH)) continue;

		float ndcX = (px / float(_graphDebugRenderW)) * 2.0f - 1.0f;
		float ndcY = 1.0f - (py / float(_graphDebugRenderH)) * 2.0f;
		if (!std::isfinite(ndcX) || !std::isfinite(ndcY)) continue;

		_graphDebugProjXY[base + 0] = ndcX * zoom;
		_graphDebugProjXY[base + 1] = ndcY * zoom;
		visible[static_cast<size_t>(i)] = 1;
	}

	// ── 2. 엣지 인덱스 빌드 ──────────────────────────────────────
	const int totalEdges = static_cast<int>(_graphDebugEdges.size());
	std::vector<unsigned int> edgeIdx;
	edgeIdx.reserve(static_cast<size_t>(totalEdges) * 2);
	for (int e = 0; e < totalEdges; ++e) {
		const int a = _graphDebugEdges[static_cast<size_t>(e)].first;
		const int b = _graphDebugEdges[static_cast<size_t>(e)].second;
		if (a < 0 || b < 0 || a >= N || b >= N) continue;
		if (!visible[static_cast<size_t>(a)] || !visible[static_cast<size_t>(b)]) continue;
		edgeIdx.push_back(static_cast<unsigned int>(a));
		edgeIdx.push_back(static_cast<unsigned int>(b));
	}

	std::vector<unsigned int> nodeIdx;
	nodeIdx.reserve(static_cast<size_t>(N));
	for (int i = 0; i < N; ++i)
		if (visible[static_cast<size_t>(i)])
			nodeIdx.push_back(static_cast<unsigned int>(i));

	// ── 3. VBO/EBO 업로드 ────────────────────────────────────────
	glBindVertexArray(_graphDebugVAO);

	glBindBuffer(GL_ARRAY_BUFFER, _graphDebugVBO);
	glBufferData(GL_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(_graphDebugProjXY.size() * sizeof(float)),
		_graphDebugProjXY.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _graphDebugEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(edgeIdx.size() * sizeof(unsigned int)),
		edgeIdx.data(), GL_DYNAMIC_DRAW);

	// ── 4. 상태 저장 ─────────────────────────────────────────────
	GLint     prevViewport[4], prevScissorBox[4];
	GLboolean prevScissorEnabled = glIsEnabled(GL_SCISSOR_TEST);
	GLboolean prevBlendEnabled = glIsEnabled(GL_BLEND);
	GLint     prevBlendSrc, prevBlendDst;
	GLfloat   prevClearColor[4];                         

	glGetIntegerv(GL_VIEWPORT, prevViewport);
	glGetIntegerv(GL_SCISSOR_BOX, prevScissorBox);
	glGetIntegerv(GL_BLEND_SRC, &prevBlendSrc);
	glGetIntegerv(GL_BLEND_DST, &prevBlendDst);
	glGetFloatv(GL_COLOR_CLEAR_VALUE, prevClearColor);   

	// ── 5. Viewport 설정 (슬라이더로 조절 가능한 영역) ───────────
	const int vpX = static_cast<int>(_graphDebugViewX);
	const int vpY = static_cast<int>(screenH - _graphDebugViewY - _graphDebugViewH);
	const int vpW = static_cast<int>(_graphDebugViewW);
	const int vpH = static_cast<int>(_graphDebugViewH);

	glViewport(vpX, vpY, vpW, vpH);
	glEnable(GL_SCISSOR_TEST);
	glScissor(vpX, vpY, vpW, vpH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);               // 추가: 검정
	glClear(GL_COLOR_BUFFER_BIT);                        // 추가: 해당 영역만 클리어

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_PROGRAM_POINT_SIZE);

	glUseProgram(_graphDebugShader);
	const GLint colorLoc = glGetUniformLocation(_graphDebugShader, "uColor");
	const GLint pSizeLoc = glGetUniformLocation(_graphDebugShader, "uPointSize");

	// ── 6. 점 먼저 (작게, 반투명) ────────────────────────────────
	glUniform1f(pSizeLoc, std::max(1.0f, _graphDebugPointRadius));
	glUniform4f(colorLoc, 1.0f, 0.7f, 0.27f, 0.6f);  // 주황, 알파 낮춤

	GLuint tmpEBO;
	glGenBuffers(1, &tmpEBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tmpEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(nodeIdx.size() * sizeof(unsigned int)),
		nodeIdx.data(), GL_DYNAMIC_DRAW);
	glDrawElements(GL_POINTS,
		static_cast<GLsizei>(nodeIdx.size()), GL_UNSIGNED_INT, nullptr);
	glDeleteBuffers(1, &tmpEBO);

	// ── 7. 간선 나중에 (1px 고정, 알파로 밀도 표현) ──────────────
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _graphDebugEBO);
	glLineWidth(_graphDebugLineThickness);  // 드라이버 한계상 1px이 최솟값
	glUniform4f(colorLoc,
		220.f / 255.f, 100.f / 255.f, 155.f / 255.f,
		_graphDebugEdgeAlpha);  // 슬라이더로 조절
	glDrawElements(GL_LINES,
		static_cast<GLsizei>(edgeIdx.size()), GL_UNSIGNED_INT, nullptr);

	// ── 8. 상태 복구 ─────────────────────────────────────────────
	glBindVertexArray(0);
	glUseProgram(0);
	glLineWidth(1.0f);
	glDisable(GL_PROGRAM_POINT_SIZE);
	glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
	glScissor(prevScissorBox[0], prevScissorBox[1], prevScissorBox[2], prevScissorBox[3]);
	if (!prevScissorEnabled) glDisable(GL_SCISSOR_TEST);
	if (!prevBlendEnabled)   glDisable(GL_BLEND);
	else glBlendFunc(static_cast<GLenum>(prevBlendSrc), static_cast<GLenum>(prevBlendDst));
	glClearColor(prevClearColor[0], prevClearColor[1],
		prevClearColor[2], prevClearColor[3]);
}



//
//void sibr::GaussianView::drawGraphDebugWindow()
//{
//	if (!_showGraphDebugWindow) return;
//
//	// 1. 노드 projected 2D 좌표 → NDC 변환
//	//    ImGui 윈도우 픽셀 좌표 → OpenGL NDC (-1~1)
//	//    OpenGL viewport는 전체 화면 기준
//	ImGuiIO& io = ImGui::GetIO();
//	const float screenW = io.DisplaySize.x;
//	const float screenH = io.DisplaySize.y;
//	if (!_showGraphDebugWindow) return;
//
//	ImGui::SetNextWindowSize(ImVec2(260.0f, 180.0f), ImGuiCond_FirstUseEver);
//	if (!ImGui::Begin("Graph Debug", &_showGraphDebugWindow)) {
//		ImGui::End();
//		return;
//	}
//
//	ImGui::Text("Realtime graph topology + deformed node positions");
//	ImGui::SliderFloat("Update Interval (sec)", &_graphDebugUpdateIntervalSec, 0.0f, 0.25f, "%.3f");
//	ImGui::SliderFloat("Line Thickness", &_graphDebugLineThickness, 0.01f, 1.0f, "%.2f");
//	ImGui::SliderFloat("Point Radius", &_graphDebugPointRadius, 0.001f, 5.0f, "%.2f");
//	ImGui::Text("Nodes: %d, Edges: %d", count, static_cast<int>(_graphDebugEdges.size()));
//	ImGui::Text("Draw mode: OpenGL direct (no vertex limit)");
//	ImGui::SliderFloat("Edge Alpha", &_graphDebugEdgeAlpha, 0.01f, 1.0f, "%.3f");
//	ImGui::Separator();
//    ImGui::Text("Graph Viewport (screen pixels)");
//    ImGui::SliderFloat("X",      &_graphDebugViewX, 0.0f, screenW - 100.0f, "%.0f");
//    ImGui::SliderFloat("Y",      &_graphDebugViewY, 0.0f, screenH - 100.0f, "%.0f");
//    ImGui::SliderFloat("Width",  &_graphDebugViewW, 100.0f, screenW,        "%.0f");
//    ImGui::SliderFloat("Height", &_graphDebugViewH, 100.0f, screenH,        "%.0f");
//    ImGui::End();
//	if (_hasLastEye) {
//		updateGraphDebugPositions();
//	}
//
//
//	// ── OpenGL로 직접 그리기 ─────────────────────────────────────
//	initGraphDebugGL();
//
//	const int N = count;
//	if (N <= 0) return;
//
//	
//
//	_graphDebugProjXY.assign(static_cast<size_t>(N) * 2, 0.0f);
//	std::vector<unsigned char> visible(static_cast<size_t>(N), 0);
//
//	for (int i = 0; i < N; ++i) {
//		if (!_graphDebugRadii.empty() && _graphDebugRadii[static_cast<size_t>(i)] <= 0)
//			continue;
//		const size_t base = static_cast<size_t>(i) * 2ull;
//		const float px = _graphDebugScreenXY[base + 0];
//		const float py = _graphDebugScreenXY[base + 1];
//		if (!std::isfinite(px) || !std::isfinite(py)) continue;
//		if (px < 0.0f || py < 0.0f ||
//			px >= float(_graphDebugRenderW) || py >= float(_graphDebugRenderH)) continue;
//
//		// render-space → draw-area pixel
//		float ndcX = (px / float(_graphDebugRenderW)) * 2.0f - 1.0f;
//		float ndcY = 1.0f - (py / float(_graphDebugRenderH)) * 2.0f;
//		if (!std::isfinite(ndcX) || !std::isfinite(ndcY)) continue;
//		ndcX *= 1.5f;
//		ndcY *= 1.5f;
//		// pixel → NDC (OpenGL Y축 반전)
//		_graphDebugProjXY[base + 0] = ndcX;
//		_graphDebugProjXY[base + 1] = ndcY;
//		visible[static_cast<size_t>(i)] = 1;
//	}
//
//	// 2. 엣지 인덱스 배열 생성
//	const int totalEdges = static_cast<int>(_graphDebugEdges.size());
//	std::vector<unsigned int> edgeIdx;
//	edgeIdx.reserve(static_cast<size_t>(totalEdges) * 2);
//	for (int e = 0; e < totalEdges; ++e) {
//		const int a = _graphDebugEdges[static_cast<size_t>(e)].first;
//		const int b = _graphDebugEdges[static_cast<size_t>(e)].second;
//		if (a < 0 || b < 0 || a >= N || b >= N) continue;
//		if (!visible[static_cast<size_t>(a)] || !visible[static_cast<size_t>(b)]) continue;
//		edgeIdx.push_back(static_cast<unsigned int>(a));
//		edgeIdx.push_back(static_cast<unsigned int>(b));
//	}
//
//	// 3. VBO/EBO 업로드
//	glBindVertexArray(_graphDebugVAO);
//
//	glBindBuffer(GL_ARRAY_BUFFER, _graphDebugVBO);
//	glBufferData(GL_ARRAY_BUFFER,
//		static_cast<GLsizeiptr>(_graphDebugProjXY.size() * sizeof(float)),
//		_graphDebugProjXY.data(), GL_DYNAMIC_DRAW);
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _graphDebugEBO);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
//		static_cast<GLsizeiptr>(edgeIdx.size() * sizeof(unsigned int)),
//		edgeIdx.data(), GL_DYNAMIC_DRAW);
//
//	// 4. OpenGL 상태 설정
//	glEnable(GL_BLEND);
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//	glEnable(GL_PROGRAM_POINT_SIZE);
//	glLineWidth(_graphDebugLineThickness);
//
//	glUseProgram(_graphDebugShader);
//	const GLint colorLoc = glGetUniformLocation(_graphDebugShader, "uColor");
//	const GLint pSizeLoc = glGetUniformLocation(_graphDebugShader, "uPointSize");
//
//	
//
//	// 6. 노드(점) 그리기
//	//    PointSize를 픽셀 반경 기준으로 설정
//	const char* psSrc = nullptr; // gl_PointSize는 vsrc에서 고정값 사용 중
//	// → PointRadius를 uniform으로 넘기려면 셰이더 수정 필요.
//	//   여기서는 glPointSize로 근사
//	glPointSize(std::max(1.0f, _graphDebugPointRadius));
//	glUniform4f(colorLoc, 255.f / 255.f, 180.f / 255.f, 70.f / 255.f, 0.6f);
//
//	
//
//	// visible 노드만 그리기 위해 별도 인덱스 배열
//	std::vector<unsigned int> nodeIdx;
//	nodeIdx.reserve(static_cast<size_t>(N));
//	for (int i = 0; i < N; ++i)
//		if (visible[static_cast<size_t>(i)])
//			nodeIdx.push_back(static_cast<unsigned int>(i));
//
//	GLuint tmpEBO;
//	glGenBuffers(1, &tmpEBO);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tmpEBO);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
//		static_cast<GLsizeiptr>(nodeIdx.size() * sizeof(unsigned int)),
//		nodeIdx.data(), GL_DYNAMIC_DRAW);
//	glDrawElements(GL_POINTS,
//		static_cast<GLsizei>(nodeIdx.size()),
//		GL_UNSIGNED_INT, nullptr);
//	glDeleteBuffers(1, &tmpEBO);
//
//	// 5. 엣지 그리기
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _graphDebugEBO);
//	glLineWidth(std::max(0.0001f, _graphDebugLineThickness));
//	glUniform4f(colorLoc,
//		120.f / 255.f, 200.f / 255.f, 255.f / 255.f,
//		_graphDebugEdgeAlpha);
//	glDrawElements(GL_LINES,
//		static_cast<GLsizei>(edgeIdx.size()),
//		GL_UNSIGNED_INT, nullptr);
//
//	// 7. 정리
//	glBindVertexArray(0);
//	glUseProgram(0);
//	glDisable(GL_PROGRAM_POINT_SIZE);
//	glLineWidth(std::max(0.0001f, _graphDebugLineThickness));
//}
//


void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
{
	static int i = 0;

	_lastEye = eye;
	_hasLastEye = true;

	// Sync internal render resolution to the actual destination RT size.
	// This is required for correct picking/index mapping after window resize
	// or when launching without an explicit --rendering-size.
	const int targetW = std::max(1, static_cast<int>(dst.w()));
	const int targetH = std::max(1, static_cast<int>(dst.h()));
	if (_resolution.x() != targetW || _resolution.y() != targetH)
	{
		setResolution(sibr::Vector2i(targetW, targetH));
		_copyRenderer->width() = targetW;
		_copyRenderer->height() = targetH;

		const size_t rgbBytes =
			static_cast<size_t>(targetW) * static_cast<size_t>(targetH) * 3ull * sizeof(float);

		// Recreate GL output buffer at the new size.
		if (!_interop_failed && imageBufferCuda)
		{
			cudaGraphicsUnregisterResource(imageBufferCuda);
			imageBufferCuda = nullptr;
		}
		if (imageBuffer)
		{
			glDeleteBuffers(1, &imageBuffer);
			imageBuffer = 0;
		}
		glCreateBuffers(1, &imageBuffer);
		glNamedBufferStorage(imageBuffer, rgbBytes, nullptr, GL_DYNAMIC_STORAGE_BIT);

		if (!_interop_failed)
		{
			cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
			if (cudaGetLastError() != cudaSuccess)
			{
				// Fallback path when interop cannot be re-registered.
				_interop_failed = true;
				imageBufferCuda = nullptr;
			}
		}

		if (_interop_failed)
		{
			if (fallbackBufferCuda)
			{
				cudaFree(fallbackBufferCuda);
				fallbackBufferCuda = nullptr;
			}
			fallback_bytes.resize(rgbBytes);
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc(&fallbackBufferCuda, fallback_bytes.size()));
		}

		// Force picking ID buffer reallocation for the new resolution.
		if (_idBufferCuda)
		{
			cudaFree(_idBufferCuda);
			_idBufferCuda = nullptr;
		}
		_idBufferCount = 0;
	}
	
	t += 0.01;
	
	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;
		_graphDebugProjMat = proj_mat;
		_graphDebugProjValid = true;

		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		// Rasterize
		int* rects = _fastCulling ? rect_cuda : nullptr;
		float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
		float* boxmax = _cropping ? (float*)&_boxmax : nullptr;

		int* idBufferForRender = nullptr;
		
		//////////////////////////picking//////////////////////
		if (_pickingMode)
		{//해상도 크기만큼_idBufferCuda id 저장버퍼 준비
			//이미 renderCUDA는 원래 픽셀마다 계산을 매 프레임마다 하고있기때문에 int*에 값을 쓰는비용은거의0에가까움
			//따라서 해상도크기만큼 픽셀별로 저장해주는게 최선의 방법임
			const size_t idCount = static_cast<size_t>(_resolution.x()) * static_cast<size_t>(_resolution.y());
			if (_idBufferCount != idCount)
			{
				if (_idBufferCuda)
				{
					cudaFree(_idBufferCuda);
					_idBufferCuda = nullptr;
				}
				if (idCount > 0)
				{
					CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&_idBufferCuda, idCount * sizeof(int)));
				}
				_idBufferCount = idCount;
			}
			idBufferForRender = _idBufferCuda;
		}
		//////////////////////////picking////////////////////////

		CudaRasterizer::Rasterizer::forward(


			_cm, _activeSet,


			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			count, _sh_degree, 16,
			background_cuda,
			_resolution.x(), _resolution.y(),
			pos_cuda,
			_neighborCapacity,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			
			_rotatingModifier_COV2D_Matrix_x,
			_rotatingModifier_COV2D_Matrix_y,
			_rotatingModifier_COV2D_Matrix_z,

			t,
			_deformationAwareCovariance,

			rot_cuda,
			nullptr,
			view_cuda,
			proj_cuda,
			cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			_antialiasing,
			idBufferForRender,//renderCUDA.까지 들어가서 해상도크기만큼 즉 픽셀개수만큼 매 프레임 픽셀별로 가장 앞에있는 가우시안을 저장
			nullptr,
			rects,
			boxmin,
			boxmax
			
		);

		if (_pickingMode && _pendingPick && _idBufferCuda && _idBufferCount > 0)
		{
			const int w = _resolution.x();
			const int h = _resolution.y();
			//마우스로 클릭한 픽셀위치 px py
			int px = std::max(0, std::min(_pendingPickPixel.x(), w - 1));
			int py = std::max(0, std::min(_pendingPickPixel.y(), h - 1));
			// gpu  메모리는 1차원 배열,따라서 내가 마우스로 선택한 픽셀 (px, py)를 1차원 인덱스(pixIndex)로 바꿈
			const size_t pixIndex = static_cast<size_t>(py) * static_cast<size_t>(w) + static_cast<size_t>(px);
			int picked = -1;
			//각픽셀에서 저장한 가우시안들배열 _idBufferCuda 에서 pixIndex번째 가우시안 을 선택_idBufferCuda + pixIndex(시작주소)  
			CUDA_SAFE_CALL(cudaMemcpy(&picked, _idBufferCuda + pixIndex, sizeof(int), cudaMemcpyDeviceToHost));
			_pickedId = picked;
			_pickedWorldPosValid = false;
			if (_pickedId >= 0 && _pickedId < count)
			{
				float p[3] = { 0.f, 0.f, 0.f };
				//선택된 가우시안의 x,y,z가 시작되는 주소
				CUDA_SAFE_CALL(cudaMemcpy(p, pos_cuda + 3 * _pickedId, 3 * sizeof(float), cudaMemcpyDeviceToHost));
				_pickedWorldPos = sibr::Vector3f(p[0], p[1], p[2]);
				_pickedWorldPosValid = true;
			}
			_pendingPick = false;
			_dragStarted = false;
		}

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}

}


//////////////////////////picking//////////////////////////
void sibr::GaussianView::onUpdate(Input& input)//ViewBase 의 onUpdate 를 재정의함 MultiViewManager.cpp 에서 GaussianView 의 Ptr 이 여기서 정의한onUpdate 를 호출
{
	if (!_pickingMode || !_hasLastEye || _resolution.x() <= 0 || _resolution.y() <= 0)
	{
		return;
	}

	const auto mapMouseToRenderPixel = [&](const sibr::Vector2i& mouse) -> sibr::Vector2i
	{
		const int rw = std::max(1, _resolution.x());
		const int rh = std::max(1, _resolution.y());
		const int vw = std::max(1, (_pickInputViewportSize.x() > 1) ? _pickInputViewportSize.x() : _resolution.x());
		const int vh = std::max(1, (_pickInputViewportSize.y() > 1) ? _pickInputViewportSize.y() : _resolution.y());

		int px = static_cast<int>(std::floor(float(mouse.x()) * float(rw) / float(vw)));
		int py = static_cast<int>(std::floor(float(mouse.y()) * float(rh) / float(vh)));
		px = std::max(0, std::min(px, rw - 1));
		py = std::max(0, std::min(py, rh - 1));
		return sibr::Vector2i(px, py);
	};

	if (input.mouseButton().isPressed(sibr::Mouse::Left))
	{
		_pendingPick = true;
		_pendingPickPixel = mapMouseToRenderPixel(input.mousePosition());
		_lastPickMouse = _pendingPickPixel;
		_pickDragActive = true;
		_pickedId = -1;
		_pickedWorldPosValid = false;
		_dragStarted = false;
	}

	if (input.mouseButton().isReleased(sibr::Mouse::Left))
	{
		_pickDragActive = false;
	}

	if (_pickDragActive && _pickedId >= 0 && _pickedWorldPosValid && input.mouseButton().isActivated(sibr::Mouse::Left))
	{
		const sibr::Vector2i cur = mapMouseToRenderPixel(input.mousePosition());
		const sibr::Vector2i delta = cur - _lastPickMouse;
		_lastPickMouse = cur;
		// Keep issuing pick commands while dragging, including zero cursor delta frames.
		{
			const float depth = (_lastEye.position() - _pickedWorldPos).norm();
			const float tan_fovy = tan(_lastEye.fovy() * 0.5f);
			const float tan_fovx = tan_fovy * _lastEye.aspect();
			const float world_per_px_x = (2.0f * depth * tan_fovx) / float(_resolution.x());
			const float world_per_px_y = (2.0f * depth * tan_fovy) / float(_resolution.y());
			sibr::Vector3f move = _lastEye.right() * (delta.x() * world_per_px_x)
				- _lastEye.up() * (delta.y() * world_per_px_y);
			move *= _pickDragScale;// 드래그 그레디언트 조절
			if (!_dragStarted)
			{
				_cm.FPS = true;
				_cm.resetTime();
				_activeSet.clear();
				_dragStarted = true;
			}
			int idx = _pickedId;
			//_cm.setPointPos(idx, glm::vec3(_pickedWorldPos.x(), _pickedWorldPos.y(), _pickedWorldPos.z()), _activeSet);
			_cm.movePointPos(&idx, glm::vec3(move.x(), move.y(), move.z()), _activeSet);
			_pickedWorldPos += move;
		}
	}
}
//////////////////////////picking//////////////////////////

void sibr::GaussianView::onUpdate(Input& input, const Viewport& vp)
{
	// Mouse coordinates come in viewport space (local). Cache viewport size so
	// we can remap them to render-pixel space used by CUDA picking buffers.
	_pickInputViewportSize = sibr::Vector2i(
		std::max(1, static_cast<int>(std::round(vp.finalWidth()))),
		std::max(1, static_cast<int>(std::round(vp.finalHeight()))));
	onUpdate(input);
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str()))
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 10.0f);
		ImGui::Checkbox("Update covariance from local deformation", &_deformationAwareCovariance);
	}
	if (ImGui::Checkbox("Enable VSync (Limit 75Hz)", &_vsyncEnabled)) {
		// 체크박스를 누를 때마다 실시간으로 윈도우 설정 변경!
		_window->setVsynced(_vsyncEnabled);
	}

	if (!_vsyncEnabled) {
		ImGui::SameLine();
		ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), " (Performance Mode)");
	}
	ImGui::Checkbox("Fast culling", &_fastCulling);
	ImGui::Checkbox("Antialiasing", &_antialiasing);
	ImGui::Checkbox("Show Graph Debug Window", &_showGraphDebugWindow);

	ImGui::Checkbox("Crop Box", &_cropping);

	if (ImGui::Checkbox("Picking Mode", &_pickingMode))
	{
		if (_cameraHandler)
		{
			_cameraHandler->setInputEnabled(!_pickingMode || _keepCameraMotionWhilePicking);
		}
		_pickDragActive = false;
		_pendingPick = false;
		_dragStarted = false;
		_pickedId = -1;
		_pickedWorldPosValid = false;
	}
	if (ImGui::Checkbox("Keep Camera Motion While Picking", &_keepCameraMotionWhilePicking))
	{
		if (_cameraHandler)
		{
			_cameraHandler->setInputEnabled(!_pickingMode || _keepCameraMotionWhilePicking);
		}
	}
	ImGui::SliderFloat("Pick Drag Scale", &_pickDragScale, 0.01f, 10.0f);
	ImGui::SliderInt("Pick bfsHops Scale", &_pickbfsHopsScale, 1, 100);
	FORWARD::setPickingParams(_pickbfsHopsScale);
	ImGui::Text("Picked ID: %d", _pickedId);
	const char* chainmailModes[] = { "CPU", "GPU" };
	int selectedChainmailMode = _cpugpuMode;
	if (ImGui::Combo("ChainMail execution", &selectedChainmailMode, chainmailModes, IM_ARRAYSIZE(chainmailModes)))
	{
		_cpugpuMode = selectedChainmailMode;
		FORWARD::setGpuChainmailMode(_cpugpuMode);
	}
	if (_cpugpuMode == 1 && ImGui::Checkbox("Active Map (GPU)", &_useActiveMap))
	{
		FORWARD::setChainmailActiveMapEnabled(_useActiveMap);
	}

	ImGui::Separator();
	ImGui::Text("ChainMail parameters");
	ImGui::SliderInt("Propagation iterations", &_cmPropIters, 0, 100);
	ImGui::SliderInt("Relaxation iterations", &_cmRelaxIters, 0, 50);
	ImGui::SliderFloat("Propagation strength", &_cmPropStrength, 0.0f, 2.0f);
	ImGui::SliderFloat("Stiffness", &_cmStiffness, 0.0f, 2.0f);
	ImGui::SliderFloat("Damping", &_cmDamping, 0.0f, 2.0f);
	ImGui::Separator();
	ImGui::Text("Material response");
	ImGui::Text("Constraint scale: < 1.0 is stiffer; > 1.0 is softer");
	ImGui::SliderFloat("Global constraint", &_cmConstraintGlobalScale, 0.001f, 50.0f, "%.4f");
	ImGui::SliderFloat("Air constraint", &_cmConstraintAirScale, 0.001f, 1.0f, "%.3f");
	ImGui::SliderFloat("Skin constraint", &_cmConstraintSkinScale, 0.001f, 1.0f, "%.3f");
	ImGui::SliderFloat("Bone constraint", &_cmConstraintBoneScale, 0.001f, 1.0f, "%.3f");
	ImGui::Checkbox("Use edge stiffness", &_cmUseEdgeStiffness);
	ImGui::SliderFloat("Edge stiffness influence", &_cmEdgeStiffnessInfluence, 0.0f, 10.0f, "%.3f");
	ImGui::Separator();
	ImGui::Text("Motion response");
	ImGui::SliderFloat("Inertia gain", &_cmInertiaGain, 0.0f, 1.0f, "%.3f");
	ImGui::SliderFloat("Velocity retention", &_cmVelocityRetention, 0.0f, 1.0f, "%.3f");
	ImGui::SliderFloat("Velocity clamp", &_cmVelocityClamp, 0.0f, 50.0f, "%.3f");

	_cmConstraintGlobalScale = std::max(_cmConstraintGlobalScale, 1e-4f);
	_cmConstraintAirScale = std::max(_cmConstraintAirScale, 1e-4f);
	_cmConstraintSkinScale = std::max(_cmConstraintSkinScale, 1e-4f);
	_cmConstraintBoneScale = std::max(_cmConstraintBoneScale, 1e-4f);
	_cmEdgeStiffnessInfluence = std::max(_cmEdgeStiffnessInfluence, 0.0f);
	_cmInertiaGain = std::max(_cmInertiaGain, 0.0f);
	_cmVelocityRetention = std::clamp(_cmVelocityRetention, 0.0f, 1.0f);
	_cmVelocityClamp = std::max(_cmVelocityClamp, 0.0f);
	FORWARD::setChainmailParams(_cmPropIters, _cmRelaxIters, _cmPropStrength, _cmStiffness, _cmDamping);
	FORWARD::setChainmailMaterialParams(_cmConstraintGlobalScale, _cmConstraintAirScale, _cmConstraintSkinScale, _cmConstraintBoneScale, _cmUseEdgeStiffness, _cmEdgeStiffnessInfluence);
	FORWARD::setChainmailDynamicsParams(_cmInertiaGain, _cmVelocityRetention, _cmVelocityClamp);

	const int activeCount = FORWARD::getChainmailActiveCount();
	const float ratio = (_pointCount > 0) ? (100.0f * float(activeCount) / float(_pointCount)) : 0.0f;
	ImGui::Text("Active ratio: %.2f%% (%d/%d)", ratio, activeCount, _pointCount);

	if (_cropping)
	{
		
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());


		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save"))
		{
			std::vector<Pos> pos(count);
			std::vector<Rot> rot(count);
			std::vector<float> opacity(count);
			std::vector<SHs<3>> shs(count);
			std::vector<Scale> scale(count);
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
			savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax);
		}
	}

	ImGui::End();
	drawGraphDebugWindow();

	if (!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	// Cleanup
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);
	if (_idBufferCuda)
	{
		cudaFree(_idBufferCuda);
		_idBufferCuda = nullptr;
		_idBufferCount = 0;
	}

	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}

