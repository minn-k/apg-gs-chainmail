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


//chainmail

////bool ChainMail::saveGraph(const std::string& filename) {
////	std::ofstream out(filename);
////	if (!out) return false;
////	out << elements.size() << " " << neighbors.size() / 2 << "\n";
////	for (const auto& e : elements)
////		out << e.pos.x << " " << e.pos.y << " " << e.pos.z << " " << e.density << "\n";
////	// 간선 정보 저장(중복주의: 양방향으로 저장되어 있을 가능성)
////	std::unordered_set<uint64_t> written;
////	for (int i = 0; i < elements.size(); ++i) {
////		const Element& e = elements[i];
////		for (int j = 0; j < e.neighborCnt; ++j) {
////			const Neighbor& n = neighbors[e.offset + j];
////			int a = std::min(i, n.idx), b = std::max(i, n.idx);
////			uint64_t key = (static_cast<uint64_t>(a) << 32) | (b);
////			if (written.count(key)) continue;
////			out << i << " " << n.idx << " " << n.dist << "\n";
////			written.insert(key);
////		}
////	}
////	return true;
////}
//
//void FORWARD::ChainMail::resetTime() {
//	for (auto& e : elements)
//		e.time = 1e9f;
//}
//


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
std::vector<sibr::Pos> _pos;
std::vector<sibr::Rot> _rot;
std::vector<sibr::Scale> _scale;
std::vector<float> _opacity;
std::vector<sibr::SHs<3>> _shs;
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
		
		// Activate alpha
		glm::vec3 pivot(-2.0f, 1.0f, 1.0f);
		float radius_sq = 6 * 6;
		float dx = pos[k].x() - pivot.x;
		float dy = pos[k].y() - pivot.y;
		float dz = pos[k].z() - pivot.z;
		float dist_sq = dx * dx + dy * dy + dz * dz;
		opacities[k] = sigmoid(points[i].opacity);
		float min_opacity_threshold = 0.0899;
		float max_scale_threshold = 0.085f; // 예시값, 데이터에 따라 0.5~2.0 사이에서 실험적으로 조정
		// 스케일(크기) 필터링 추가
		float max_scale = std::max({ scales[k].scale[0], scales[k].scale[1], scales[k].scale[2] });
		/*if (opacities[k] < min_opacity_threshold || dist_sq > radius_sq|| max_scale > max_scale_threshold) {
			opacities[k] = 0;
			scales[k].scale[0] = 0;
			scales[k].scale[1] = 0;
			scales[k].scale[2] = 0;
			}*/
		if (opacities[k] < min_opacity_threshold ) {
			opacities[k] = 0;
		}
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

void saveARDiffsToFile(const std::vector<float>& AR_diffs, const std::string& filename) {
	std::ofstream out(filename);
	if (!out.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}
	for (float val : AR_diffs) {
		out << val << "\n";
	}
	out.close();
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

// KD-Tree 어댑터 정의
template <typename Derived>
struct KDTreeAdaptor {
	const Derived& obj;
	KDTreeAdaptor(const Derived& obj_) : obj(obj_) {}

	inline size_t kdtree_get_point_count() const { return obj.rows(); }

	inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
		return obj.coeff(idx, dim);
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};


using namespace nanoflann;

void saveFilteredGaussianEdgesToFile(
	const std::vector<sibr::Pos>& pos,
	std::vector<FORWARD::Edge>& edges,
	std::vector<float>& opacity,
	const std::string& filename
) {
	std::set<int> valid_indices;
	for (size_t i = 0; i < pos.size(); ++i) {
			valid_indices.insert(i);
		
	}

	std::vector<sibr::Pos> filtered_pos;
	std::vector<float> filtered_opacity;

	std::map<int, int> old_to_new_idx;
	int new_idx = 0;
	for (size_t i = 0; i < pos.size(); ++i) {
		if (valid_indices.count(i)) {
			filtered_pos.push_back(pos[i]);
			filtered_opacity.push_back(opacity[i]);
			old_to_new_idx[i] = new_idx++;
		}
	}

	std::vector<FORWARD::Edge> filtered_edges;
	for (const auto& e : edges) {
		int v0 = e.m_vert[0], v1 = e.m_vert[1];
		if (valid_indices.count(v0) && valid_indices.count(v1)) {
			filtered_edges.emplace_back(
				old_to_new_idx[v0], old_to_new_idx[v1], e.rl, e.st
			);
		}
	}

	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "[ERROR] Failed to open file: " << filename << std::endl;
		return;
	}
	file << filtered_pos.size() << " " << filtered_edges.size() << "\n";
	for (size_t i = 0; i < filtered_pos.size(); ++i) {
		const auto& p = filtered_pos[i];
		file << p.x() << " " << p.y() << " " << p.z() << " " << filtered_opacity[i] << "\n";
	}
	for (const auto& e : filtered_edges)
		file << e.m_vert[0] << " " << e.m_vert[1] << " " << e.rl << "\n";
	std::cout << "Saved filtered gaussian edges생성된 간선 수: " << filtered_edges.size() << std::endl;

	file.close();
	std::cout << "[INFO] Saved filtered gaussian edges to " << filename << std::endl;
}

// 거리 가중치 함수
#if 0
float computeDistanceWeight(float dist, float max_dist) {
	return std::exp(-2.0f * dist / (max_dist + 1e-6f));
}

// 방향성 가중치 함수 (쿼터니언 사용, glm 필요)
float computeOrientationWeight(const sibr::Rot& rot1, const sibr::Rot& rot2, const sibr::Pos& p1, const sibr::Pos& p2) {
	glm::quat q1(rot1[0], rot1[1], rot1[2], rot1[3]);
	glm::quat q2(rot2[0], rot2[1], rot2[2], rot2[3]);
	glm::vec3 dir1 = q1 * glm::vec3(1, 0, 0);
	glm::vec3 dir2 = q2 * glm::vec3(1, 0, 0);
	glm::vec3 conn = glm::normalize(glm::vec3(p2.x() - p1.x(), p2.y() - p1.y(), p2.z() - p1.z()));
	float align1 = std::abs(glm::dot(dir1, conn));
	float align2 = std::abs(glm::dot(dir2, conn));
	return 0.5f * (align1 + align2);
}


// 최적화된 진행률 리포터 클래스
class OptimizedProgressReporter {
private:
	std::atomic<size_t> processed{ 0 };
	size_t total;
	std::chrono::high_resolution_clock::time_point start_time;
	std::mutex print_mutex;

public:
	OptimizedProgressReporter(size_t total_work) : total(total_work) {
		start_time = std::chrono::high_resolution_clock::now();
	}

	void increment(size_t count = 1) {
		processed += count;
	}

	void printProgress() {
		std::lock_guard<std::mutex> lock(print_mutex);

		auto current_time = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);

		size_t current_processed = processed.load();
		int percentage = (int)((current_processed * 100) / total);

		// ETA 계산
		double eta_seconds = 0.0;
		if (current_processed > 0) {
			eta_seconds = (elapsed.count() * (total - current_processed)) / (double)current_processed;
		}

		std::cout << "\r[";
		int filled_width = (percentage * 50) / 100;
		for (int i = 0; i < 50; ++i) {
			std::cout << (i < filled_width ? "=" : " ");
		}
		std::cout << "] " << percentage << "% (" << current_processed << "/" << total
			<< ") - ETA: " << (int)eta_seconds << "s";
		std::cout.flush();

		if (current_processed >= total) {
			std::cout << std::endl;
		}
	}
};
// 빠른 방향성 유사도 계산 (간소화된 버전)
inline float fastOrientationSimilarity(const sibr::Rot& rot1, const sibr::Rot& rot2) {
	// 쿼터니언 내적으로 빠른 유사도 계산
	float dot = rot1.rot[0] * rot2.rot[0] + rot1.rot[1] * rot2.rot[1] +
		rot1.rot[2] * rot2.rot[2] + rot1.rot[3] * rot2.rot[3];
	return std::abs(dot); // 0 ~ 1 범위
}

// 최적화된 밀집도 계산 (근사치 사용)
inline float approximateLocalDensity(const std::vector<sibr::Pos>& pos, int index,
	const std::vector<size_t>& neighbor_indices) {
	// KNN 결과를 재활용하여 밀집도 근사 계산
	return static_cast<float>(neighbor_indices.size()) / (4.0f / 3.0f * M_PI * 0.1f * 0.1f * 0.1f);
}

// 스케일 비율 계산 최적화
inline float getAspectRatio(const sibr::Scale& scale) {
	float max_scale = std::max({ scale.scale[0], scale.scale[1], scale.scale[2] });
	float min_scale = std::min({ scale.scale[0], scale.scale[1], scale.scale[2] });
	return max_scale / std::max(min_scale, 1e-6f); // 0으로 나누기 방지
}

// 빠른 거리 제곱 계산
inline float fastDistanceSquared(const sibr::Pos& p1, const sibr::Pos& p2) {
	float dx = p1.x() - p2.x();
	float dy = p1.y() - p2.y();
	float dz = p1.z() - p2.z();
	return dx * dx + dy * dy + dz * dz;
}




// Disjoint?Set (Union?Find) 자료구조
struct DisjointSet {
	std::vector<int> parent, rank;
	DisjointSet(int n) : parent(n), rank(n, 0) {
		for (int i = 0; i < n; ++i) parent[i] = i;
	}
	int find(int x) {
		return parent[x] == x ? x : parent[x] = find(parent[x]);
	}
	bool unite(int a, int b) {
		a = find(a); b = find(b);
		if (a == b) return false;
		if (rank[a] < rank[b]) std::swap(a, b);
		parent[b] = a;
		if (rank[a] == rank[b]) ++rank[a];
		return true;
	}
};
// 기본 특성 유틸리티 (예시)
float euclidDist(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	return (a - b).norm();
}
float orientationSim(const sibr::Rot& ri, const sibr::Rot& rj) {
	// 회전 코사인 유사도 구현 예시
	Eigen::Quaternionf 
		qi(ri.rot[0], ri.rot[1], ri.rot[2], ri.rot[3]),
		qj(rj.rot[0], rj.rot[1], rj.rot[2], rj.rot[3]);
	return std::abs(qi.dot(qj));
}
float aspectRatio(const sibr::Scale& s) {
	float mx = std::max({ s.scale[0], s.scale[1], s.scale[2] });
	float mn = std::min({ s.scale[0], s.scale[1], s.scale[2] });
	return mx / mn;
}
struct ProgressReporter {
	std::atomic<size_t> count;
	size_t total;
	ProgressReporter(size_t t) : count(0), total(t) {}
	void increment(size_t n = 1) { count += n; }
	void print(const std::string& prefix) {
		size_t c = count.load();
		int pct = int(100.0 * c / total);
		std::cout << "\r" << prefix
			<< " " << c << " / " << total
			<< " (" << pct << "%)" << std::flush;
	}
};

/******************************************************
 *  Enhanced buildfunctionEdgeGraph for Chain-mail
 *  - 지역 밀도 기반 적응적 임계값
 *  - 멀티뷰 일관성 검증
 *  - 그래프 정규화 및 영역별 연결성 강화
 *  - 실시간 품질 평가 및 매개변수 조정
 *****************************************************/

#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <numeric>

 // 추가 헬퍼 구조체들
struct GraphQualityMetrics {
	float largest_component_ratio = 0.0f;
	int num_components = 0;
	float avg_component_size = 0.0f;
	float connectivity_score = 0.0f;

	bool isAcceptableQuality() const {
		return largest_component_ratio > 0.8f && num_components < 50;
	}
};

struct RegionInfo {
	std::vector<int> vertices;
	sibr::Pos centroid;
	float avg_density;
	bool is_primary_region;
	int region_id;
};

// 지역 밀도 계산 함수
float computeLocalDensity(int vertex_idx,
	const std::vector<sibr::Pos>& pos,
	const nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<float, KDTreeAdaptor<Eigen::MatrixXf>>,
	KDTreeAdaptor<Eigen::MatrixXf>, 3>& kdtree,
	float radius = 0.1f,
	int min_pts = 5) {
	float q[3] = { pos[vertex_idx].x(), pos[vertex_idx].y(), pos[vertex_idx].z() };
	std::vector<size_t> neighbors(min_pts + 10);
	std::vector<float> distances(min_pts + 10);

	// k-distance 계산 (HDBSCAN 방식)
	size_t found = kdtree.knnSearch(q, min_pts + 5, neighbors.data(), distances.data());
	if (found < min_pts) return 0.0f;

	float k_distance = std::sqrt(distances[min_pts - 1]);
	float effective_radius = std::max(radius, k_distance);

	// reachability distance 기반 밀도 계산
	float volume = (4.0f / 3.0f) * M_PI * pow(effective_radius, 3);
	return static_cast<float>(found) / volume;
}

// 적응적 임계값 계산
float adaptiveThreshold(int i, int j,
	const std::vector<float>& densities,
	float base_threshold,
	float density_factor = 0.3f)
{
	float avg_density = (densities[i] + densities[j]) * 0.5f;
	float density_weight = 1.0f + density_factor * std::exp(-avg_density * 0.01f);
	return base_threshold * density_weight;
}

// 멀티뷰 일관성 점수 (간소화된 버전)
float computeMultiViewConsistency(int i, int j,
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot) {
	// 거리 기반 일관성
	float distance = (pos[i] - pos[j]).norm();

	// 방향 일관성 (기존 orientationSim 활용)
	float orientation_consistency = orientationSim(rot[i], rot[j]);

	// 간단한 멀티뷰 일관성 근사
	float depth_consistency = std::exp(-distance * distance / 0.01f);

	return 0.6f * orientation_consistency + 0.4f * depth_consistency;
}

// Region Growing 기반 영역 분할
std::vector<RegionInfo> segmentByRegionGrowing(
	const std::vector<sibr::Pos>& pos,
	const std::vector<float>& densities,
	const  nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<float, KDTreeAdaptor<Eigen::MatrixXf>>,
	KDTreeAdaptor<Eigen::MatrixXf>, 3>& kdtree,
	float similarity_threshold = 0.05f,
	int min_region_size = 100) {

	const int N = static_cast<int>(pos.size());
	std::vector<bool> visited(N, false);
	std::vector<RegionInfo> regions;

	// 밀도 기준으로 정렬된 인덱스
	std::vector<int> sorted_indices(N);
	std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
	std::sort(sorted_indices.begin(), sorted_indices.end(),
		[&densities](int a, int b) { return densities[a] > densities[b]; });

	for (int seed_idx : sorted_indices) {
		if (visited[seed_idx]) continue;

		RegionInfo region;
		region.region_id = static_cast<int>(regions.size());
		region.avg_density = 0.0f;

		std::queue<int> queue;
		queue.push(seed_idx);
		visited[seed_idx] = true;

		sibr::Pos centroid_sum(0, 0, 0);

		while (!queue.empty()) {
			int current = queue.front();
			queue.pop();

			region.vertices.push_back(current);
			region.avg_density += densities[current];
			centroid_sum = centroid_sum + pos[current];

			// 이웃 탐색
			float q[3] = { pos[current].x(), pos[current].y(), pos[current].z() };
			std::vector<size_t> neighbors(20);
			std::vector<float> distances(20);

			size_t found = kdtree.knnSearch(q, 20, neighbors.data(), distances.data());

			for (size_t k = 1; k < found; ++k) {
				int neighbor = static_cast<int>(neighbors[k]);
				if (visited[neighbor]) continue;

				float distance = std::sqrt(distances[k]);
				float density_diff = std::abs(densities[current] - densities[neighbor]);

				if (distance < similarity_threshold && density_diff < 0.1f) {
					visited[neighbor] = true;
					queue.push(neighbor);
				}
			}
		}

		if (region.vertices.size() >= min_region_size) {
			region.avg_density /= region.vertices.size();
			region.centroid = centroid_sum / static_cast<float>(region.vertices.size());
			region.is_primary_region = region.avg_density > 5.0f; // 임계값 조정 필요
			regions.push_back(region);
		}
	}

	return regions;
}

// 연결성 품질 평가
GraphQualityMetrics evaluateGraphQuality(const std::set<std::pair<int, int>>& edgeSet,
	size_t num_vertices) {
	GraphQualityMetrics metrics;

	// 간단한 연결 요소 계산 (BFS 기반)
	std::vector<std::vector<int>> adj(num_vertices);
	for (const auto& edge : edgeSet) {
		adj[edge.first].push_back(edge.second);
		adj[edge.second].push_back(edge.first);
	}

	std::vector<bool> visited(num_vertices, false);
	std::vector<int> component_sizes;

	for (size_t i = 0; i < num_vertices; ++i) {
		if (visited[i]) continue;

		int size = 0;
		std::queue<int> queue;
		queue.push(static_cast<int>(i));
		visited[i] = true;

		while (!queue.empty()) {
			int v = queue.front();
			queue.pop();
			size++;

			for (int neighbor : adj[v]) {
				if (!visited[neighbor]) {
					visited[neighbor] = true;
					queue.push(neighbor);
				}
			}
		}
		component_sizes.push_back(size);
	}

	metrics.num_components = static_cast<int>(component_sizes.size());
	if (!component_sizes.empty()) {
		int largest = *std::max_element(component_sizes.begin(), component_sizes.end());
		metrics.largest_component_ratio = static_cast<float>(largest) / static_cast<float>(num_vertices);
		metrics.avg_component_size = std::accumulate(component_sizes.begin(), component_sizes.end(), 0.0f) / component_sizes.size();
	}

	metrics.connectivity_score = metrics.largest_component_ratio * (1.0f - std::log(metrics.num_components + 1) / 10.0f);

	return metrics;
}
//// MST 기반 백본 구조 확보
//std::set<std::pair<int, int>> extractMSTBackbone(
//	const std::vector<sibr::Pos>& pos,
//	const std::set<std::pair<int, int>>& candidate_edges) {
//
//	// Union-Find 자료구조로 MST 구성
//	std::vector<int> parent(pos.size());
//	std::iota(parent.begin(), parent.end(), 0);
//
//	std::function<int(int)> find = [&](int x) {
//		return parent[x] == x ? x : parent[x] = find(parent[x]);
//	};
//
//	// 간선을 거리 기준으로 정렬
//	std::vector<std::tuple<float, int, int>> sorted_edges;
//	for (const auto& edge : candidate_edges) {
//		float dist = (pos[edge.first] - pos[edge.second]).norm();
//		sorted_edges.emplace_back(dist, edge.first, edge.second);
//	}
//	std::sort(sorted_edges.begin(), sorted_edges.end());
//
//	std::set<std::pair<int, int>> mst_edges;
//	for (const auto& [dist, u, v] : sorted_edges) {
//		int pu = find(u), pv = find(v);
//		if (pu != pv) {
//			parent[pu] = pv;
//			mst_edges.emplace(std::min(u, v), std::max(u, v));
//		}
//	}
//
//	return mst_edges;
//}
//// K-최근접 기반 로컬 스파시피케이션
//void applyKNeighborSparsification(std::set<std::pair<int, int>>& edgeSet,
//	const std::vector<sibr::Pos>& pos,
//	int max_neighbors_per_vertex = 8) {
//	std::map<int, std::vector<std::pair<float, int>>> vertex_neighbors;
//
//	// 각 정점별 이웃 수집 및 거리 기준 정렬
//	for (const auto& edge : edgeSet) {
//		float dist = (pos[edge.first] - pos[edge.second]).norm();
//		vertex_neighbors[edge.first].emplace_back(dist, edge.second);
//		vertex_neighbors[edge.second].emplace_back(dist, edge.first);
//	}
//
//	std::set<std::pair<int, int>> sparsified_edges;
//	for (auto& [vertex, neighbors] : vertex_neighbors) {
//		std::sort(neighbors.begin(), neighbors.end());
//
//		// 최대 K개의 최근접 이웃만 유지
//		int keep_count = std::min(max_neighbors_per_vertex, static_cast<int>(neighbors.size()));
//		for (int i = 0; i < keep_count; ++i) {
//			int neighbor = neighbors[i].second;
//			int a = std::min(vertex, neighbor);
//			int b = std::max(vertex, neighbor);
//			sparsified_edges.emplace(a, b);
//		}
//	}
//
//	edgeSet = sparsified_edges;
//}
// D는 SH 차수 (예: D=2, D=3 등)
template<int D>
float shAlignmentSim(const sibr::SHs<D>& sh1, const sibr::SHs<D>& sh2)
{
	// SHs<D>::shs는 float 배열, 크기는 (D+1)*(D+1)*3
	constexpr int N = (D + 1) * (D + 1) * 3;
	float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
	for (int i = 0; i < N; ++i) {
		dot += sh1.shs[i] * sh2.shs[i];
		norm1 += sh1.shs[i] * sh1.shs[i];
		norm2 += sh2.shs[i] * sh2.shs[i];
	}
	if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
	return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}
template<int D>
//float shAlignmentSimLuminanceWeighted(const sibr::SHs<D>& sh1, const sibr::SHs<D>& sh2) {
//	constexpr int SH_PER_CHANNEL = (D + 1) * (D + 1);
//	constexpr int N = SH_PER_CHANNEL * 3;
//	const float luminance[3] = { 0.3f, 0.59f, 0.11f };
//
//	float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
//	for (int l = 0; l <= D; ++l) {
//		for (int m = -l; m <= l; ++m) {
//			int base = l * l + l + m;
//			for (int ch = 0; ch < 3; ++ch) {
//				int idx = ch * SH_PER_CHANNEL + base;
//				float a = luminance[ch] * sh1.shs[idx];
//				float b = luminance[ch] * sh2.shs[idx];
//				dot += a * b;
//				norm1 += a * a;
//				norm2 += b * b;
//			}
//		}
//	}
//	if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
//	// 코사인 유사도
//	float cosine_sim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
//	// 정규화
//	return 0.5f * (cosine_sim + 1.0f);
//}
float shAlignmentSimLuminanceWeighted(const sibr::SHs<D>& sh1, const sibr::SHs<D>& sh2) {
	constexpr int SH_PER_CHANNEL = (D + 1) * (D + 1);
	const float w[3] = { 0.3f, 0.59f, 0.11f }; // R, G, B weights

	float sim = 0.0f;

	for (int ch = 0; ch < 3; ++ch) {
		float dot = 0.0f;
		float norm1_sq = 0.0f; // norm squared
		float norm2_sq = 0.0f; // norm squared

		int offset = ch * SH_PER_CHANNEL;

		for (int i = 0; i < SH_PER_CHANNEL; ++i) {
			float a = sh1.shs[offset + i];
			float b = sh2.shs[offset + i];
			dot += a * b;
			norm1_sq += a * a;
			norm2_sq += b * b;
		}

		float channel_sim = 0.0f;
		// 0으로 나누기 방지 및 논문 로직(0-norm일 때 유사도 0) 적용
		if (norm1_sq > 1e-6f && norm2_sq > 1e-6f) {
			float raw_cos = dot / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));

			// [-1, 1] → [0, 1] 정규화는 유효한 경우에만 수행
			channel_sim = 0.5f * (raw_cos + 1.0f);
		}
		// else: channel_sim remains 0.0f (as per paper)

		sim += w[ch] * channel_sim;
	}

	return sim;
}
float normalize(float x, float min_val, float max_val) {
	if (max_val - min_val < 1e-6f) return 0.0f;
	return (x - min_val) / (max_val - min_val);
}

// --- 1. 유사도 거리 계산 함수 분리 ---

inline float computeSimilarityDistance(
	float dist, float oSim, float AR_diff, float shSim,
	float sigma, float sigma_AR, float sigma_sh, float sigma_ori,
	float w_dist, float w_ori, float w_shape, float w_sh) {
	sigma *= 5;
	//sigma_AR /= 8;
		
	return

		//w_dist * dist;
		//w_ori * (1.0f - oSim)+
		//w_shape * AR_diff+
		//w_sh * (1 - shSim);

		//w_dist* exp(dist*dist) +
		//w_ori * (1.0f - oSim) * (1.0f - oSim) +
		//w_shape * AR_diff * AR_diff +
		//w_sh * (1 - shSim) * (1 - shSim);
		
		//w_dist* exp(dist*dist) +
		//w_ori * exp((1.0f - oSim) * (1.0f - oSim)) +
		//w_shape * exp(AR_diff * AR_diff) +
		//w_sh * exp((1 - shSim) * (1 - shSim));
	
		//w_dist* (1.0 - exp(-dist * dist)) +
		//w_ori * (1.0 - exp(-(1.0f - oSim) * (1.0f - oSim))) +
		//w_shape * (1.0 - exp(-(AR_diff * AR_diff))) +
		//w_sh * (1.0 - exp(-((1 - shSim) * (1 - shSim))));


		/*----------------------------------------------------------------------------------------------------------------------------*/
		w_dist* (1.0 - exp(-dist * dist / (sigma * sigma))) +
		w_ori * (1.0 - exp(-(1.0f - oSim) * (1.0f - oSim) / (sigma_ori * sigma_ori))) +
		w_shape * AR_diff * AR_diff +
		////w_shape * (1.0 - exp(-((AR_diff * AR_diff) / (sigma_AR * sigma_AR)))) +
		w_sh * (1.0 - exp(-((1 - shSim) * (1 - shSim)) / (sigma_sh * sigma_sh)));
		/*----------------------------------------------------------------------------------------------------------------------------*/

		//w_dist * (1.0f - exp(-pow(dist, 2) / (sigma * sigma))) +
		//w_ori * pow((1.0f - oSim), 2) +
		//w_shape * (1.0f - exp(-pow(AR_diff, 2) / (sigma_AR * sigma_AR))) +
		//w_sh * pow((1.0f - shSim), 2);
}

inline uint64_t makeKey(int a, int b) {
	return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

// 메인 함수 - 개선된 버전
template<int D>
void buildfunctionEdgeGraph2(
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot,
	const std::vector<sibr::Scale>& scales,
	std::vector<FORWARD::Edge>& edges,
	std::vector<sibr::SHs<D>>& shs,      // ?? 추가!
	const sibr::APGGraphConfig& cfg,
	bool enable_debug = true)
{
	const int N = static_cast<int>(pos.size());
	float d_threshold = 0.0f;
	auto start_time = std::chrono::high_resolution_clock::now();
	if (enable_debug) {
		std::cout << "\n=== 체인메일 그래프 구축 ===" << std::endl;
		std::cout << "입력 정점 수: " << N << std::endl;
		cfg.print();  // 실험 파라미터 콘솔 출력 추가

	}

	// 1) KD-Tree 준비
	Eigen::MatrixXf pts(N, 3);
	for (int i = 0; i < N; ++i)
		pts.row(i) = Eigen::Vector3f(pos[i].x(), pos[i].y(), pos[i].z());

	using KDAdaptor = KDTreeAdaptor<Eigen::MatrixXf>;
	using KDType = nanoflann::KDTreeSingleIndexAdaptor<
		nanoflann::L2_Simple_Adaptor<float, KDAdaptor>,
		KDAdaptor, 3>;
	KDAdaptor adaptor(pts);
	KDType kdtree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	kdtree.buildIndex();

	// 2) 보수적인 초기 간선 수집 (기존보다 더 엄격한 기준)
	// 파라미터 초기화

	float sigma = 1.0f;
	float sigma_AR = 1.0f;
	float sigma_SH = 1.0f; 
	float sigma_ORI = 1.0f;

	// 내부 파라미터를 cfg에서 받아옴 (기존 하드코딩 라인만 교체)
	const int   max_k = cfg.default_k + 1;
	const float d_thresholdP = cfg.d_thresholdP;

	float w_dist = cfg.w_dist;   // ← 기존: float w_dist  = 2.9f;
	float w_ori = cfg.w_ori;    // ← 기존: float w_ori   = 0.5f;
	float w_shape = cfg.w_shape;  // ← 기존: float w_shape = 0.4f;
	float w_sh = cfg.w_sh;     // ← 기존: float w_sh    = 0.5f;
	

	float sum_shSim = 0.0f;
	float min_shSim = 1.0f;
	float max_shSim = 0.0f;
	int shSim_count = 0;
	int highshSim_count = 0;

	float sim;
	std::vector<float> dist_values, ori_diffs, sh_diffs, AR_diffs, d_samples;

	float dist_sum = 0.0f, dist_min = FLT_MAX, dist_max = -FLT_MAX;
	float oSim_sum = 0.0f, oSim_min = FLT_MAX, oSim_max = -FLT_MAX;
	float shSim_sum = 0.0f, shSim_min = FLT_MAX, shSim_max = -FLT_MAX;
	float AR_diff_sum = 0.0f, AR_diff_min = FLT_MAX, AR_diff_max = -FLT_MAX;

	int count = 0;

	for (int i = 0; i < N; ++i) {
		float AR_i = getAspectRatio(scales[i]);
		float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
		std::vector<size_t> neigh(max_k + 1);
		std::vector<float> d2(max_k + 1);
		size_t found = kdtree.knnSearch(q, max_k + 1, neigh.data(), d2.data());
		for (size_t k = 1; k < found; ++k) {
			int j = static_cast<int>(neigh[k]);
			float dist = std::sqrt(d2[k]);
			float AR_j = getAspectRatio(scales[j]);
			// 통계 누적
			float shSim = shAlignmentSimLuminanceWeighted<D>(shs[i], shs[j]);
			float oSim = orientationSim(rot[i], rot[j]);

			//float AR_diff = std::abs(std::log(AR_i + 1e-6f) - std::log(AR_j + 1e-6f));
			float AR_diff = std::abs(std::log((AR_i + 1e-6f) / (AR_j + 1e-6f)));
			//float AR_diff = std::max(AR_i, AR_j) / std::max(std::min(AR_i, AR_j), 1e-6f) - 1.0f;

			// 누적
			dist_sum += dist;
			dist_min = std::min(dist_min, dist);
			dist_max = std::max(dist_max, dist);

			oSim_sum += oSim;
			oSim_min = std::min(oSim_min, oSim);
			oSim_max = std::max(oSim_max, oSim);

			shSim_sum += shSim;
			shSim_min = std::min(shSim_min, shSim);
			shSim_max = std::max(shSim_max, shSim);

			AR_diff_sum += AR_diff;
			AR_diff_min = std::min(AR_diff_min, AR_diff);
			AR_diff_max = std::max(AR_diff_max, AR_diff);

			count++;
			dist_values.push_back(dist);
			AR_diffs.push_back(std::abs(AR_diff));
			sh_diffs.push_back(std::abs(shSim));
			ori_diffs.push_back(std::abs(oSim));

		}
	}
	// 평균 계산
	float dist_avg = dist_sum / count;
	float oSim_avg = oSim_sum / count;
	float shSim_avg = shSim_sum / count;
	float AR_diff_avg = AR_diff_sum / count;

	// 출력
	std::cout << "Distance: min = " << dist_min << ", max = " << dist_max << ", avg = " << dist_avg << "\n";
	std::cout << "OrientationSim: min = " << oSim_min << ", max = " << oSim_max << ", avg = " << oSim_avg << "\n";
	std::cout << "SHSim: min = " << shSim_min << ", max = " << shSim_max << ", avg = " << shSim_avg << "\n";
	std::cout << "AR_diff: min = " << AR_diff_min << ", max = " << AR_diff_max << ", avg = " << AR_diff_avg << "\n";

	auto get_median = [](std::vector<float>& v) {
		std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
		return v[v.size() / 2];
	};
	// 새로운 robust sigma 추정 방식
	auto get_iqr = [](std::vector<float>& v) {
		std::sort(v.begin(), v.end());
		return v[v.size() * 3 / 4] - v[v.size() / 4];
	};
	auto minmax_norm = [](float x, float x_min, float x_max) -> float {
		if (x_max - x_min < 1e-6f) return 0.0f;
		return (x - x_min) / (x_max - x_min);
	};

	// 정규화된 feature 벡터 생성
	std::vector<float> dist_norm_values, AR_diff_norm_values;
	for (float d : dist_values)
		dist_norm_values.push_back(minmax_norm(d, dist_min, dist_max));
	for (float ad : AR_diffs)
		AR_diff_norm_values.push_back(minmax_norm(ad, AR_diff_min, AR_diff_max));
	sigma = std::max(0.1f, get_iqr(dist_norm_values) / 1.45f);
	sigma_AR = std::max(0.1f, get_iqr(AR_diff_norm_values) / 1.45f);
	sigma_SH = std::max(0.1f, get_iqr(sh_diffs) / 1.45f);
	sigma_ORI = std::max(0.1f, get_iqr(ori_diffs) / 1.45f);

	// 1. d값 분포 샘플링
	for (int i = 0; i < N; ++i) {
		float AR_i = getAspectRatio(scales[i]);
		float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
		std::vector<size_t> neigh(max_k + 1);
		std::vector<float> d2(max_k + 1);
		size_t found = kdtree.knnSearch(q, max_k + 1, neigh.data(), d2.data());
		for (size_t k = 1; k < found; ++k) {
			int j = static_cast<int>(neigh[k]);
			float dist = std::sqrt(d2[k]);
			float oSim = orientationSim(rot[i], rot[j]);
			float AR_j = getAspectRatio(scales[j]);
			float shSim = shAlignmentSimLuminanceWeighted<D>(shs[i], shs[j]);
			// 통계 누적

			//float AR_diff = std::abs(std::log(AR_i + 1e-6f) - std::log(AR_j + 1e-6f));
			float AR_diff = std::abs(std::log((AR_i + 1e-6f) / (AR_j + 1e-6f)));
			//float AR_diff = std::max(AR_i, AR_j) / std::max(std::min(AR_i, AR_j), 1e-6f) - 1.0f;
			float dist_norm = minmax_norm(dist, dist_min, dist_max);
			float AR_diff_norm = minmax_norm(AR_diff, AR_diff_min, AR_diff_max);

			float d = computeSimilarityDistance(
				dist_norm, oSim, AR_diff_norm, shSim,
				sigma, sigma_AR, sigma_SH, sigma_ORI,
				w_dist, w_ori, w_shape, w_sh);

			d_samples.push_back(d);
		}
	}
	// 2. d값 정렬 및 10% 위치 임계값 산출
	std::sort(d_samples.begin(), d_samples.end());
	size_t idx_10pct = static_cast<size_t>(d_samples.size() * d_thresholdP*0.01);
	 d_threshold = d_samples[idx_10pct];

	 if (enable_debug) {
		 std::cout << "sigma = " << sigma << std::endl;
		 std::cout << "sigma_AR = " << sigma_AR << std::endl;
		 std::cout << "sigma_ORI = " << sigma_ORI << std::endl;
		 std::cout << "sigma_SH = " << sigma_SH << std::endl;
		 std::cout << "[INFO] d값 샘플 수: " << d_samples.size() << std::endl;
		 std::cout << "[INFO] 자동 산출 d_threshold (" << d_thresholdP << "%): " << d_threshold << std::endl;
	 }
	
	//d_threshold = 56.0f;
	std::set<std::pair<int, int>> edgeSet;
	std::mutex edge_mtx;
	std::map<std::pair<int, int>, float> edgeScoreMap;

#pragma omp parallel
	{
		std::vector<size_t> neigh(max_k + 1);
		std::vector<float> d2(max_k + 1);
		std::set<std::pair<int, int>> localSet;
		std::map<std::pair<int, int>, float> localMap;

		// 진행률 리포터
		OptimizedProgressReporter progress(pos.size());
		std::atomic<int> progress_counter{ 0 };
#pragma omp for schedule(dynamic, 100)
		for (int i = 0; i < N; ++i) {
			float AR_i = getAspectRatio(scales[i]);
			float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
			size_t found = kdtree.knnSearch(q, max_k + 1, neigh.data(), d2.data());

			for (size_t k = 1; k < found; ++k) {
				int j = static_cast<int>(neigh[k]);
				if (j <= i) continue;
				
				float dist = std::sqrt(d2[k]);
				float oSim = orientationSim(rot[i], rot[j]);
				float AR_j = getAspectRatio(scales[j]);
				float shSim = shAlignmentSimLuminanceWeighted<D>(shs[i], shs[j]); // ?? 이렇게 사용
				float AR_diff = std::abs(std::log((AR_i + 1e-6f) / (AR_j + 1e-6f)));
				//float AR_diff = std::abs(std::log(AR_i + 1e-6f) - std::log(AR_j + 1e-6f));
				//float AR_diff = std::max(AR_i, AR_j) / std::max(std::min(AR_i, AR_j), 1e-6f) - 1.0f;
				float dist_norm = minmax_norm(dist, dist_min, dist_max);
				float AR_diff_norm = minmax_norm(AR_diff, AR_diff_min, AR_diff_max);

				// 엄격한 기준 적용 (기존보다 훨씬 보수적)
				float d = computeSimilarityDistance(
					dist_norm, oSim, AR_diff_norm, shSim,
					sigma, sigma_AR, sigma_SH, sigma_ORI,
					w_dist, w_ori, w_shape, w_sh);

				//	//w_sh * (1.0 - std::exp(-std::pow(1.0f - shSim,2) / (sigma_sh * sigma_sh)));
				//float d = 
				//	w_dist * dist +
				//	w_ori * (1.0f - oSim) +
				//	w_shape * abs(AR_i - AR_j) +
				//	w_sh * (1 - shSim);
				if (d <= d_threshold) { // 임계값을 20% 더 엄격하게
					//float d_norm = d / d_threshold;
					//d_norm = clamp(d_norm, 0.0f, 1.0f);

					int a = i, b = j;
					if (a > b) std::swap(a, b);
					auto key = std::make_pair(a, b);
					// localMap에 이미 있으면 더 좋은 값(예: 더 작은 d) 유지
					auto it = localMap.find(key);
					if (it == localMap.end()) localMap[key] = d;
					else it->second = std::min(it->second, d);
					localSet.emplace(a, b);
				}
			}
			
		}

		std::lock_guard<std::mutex> lk(edge_mtx);
		edgeSet.insert(localSet.begin(), localSet.end());
		for (auto& kv : localMap)
		{
			auto key = kv.first;
			float d = kv.second;

			auto it = edgeScoreMap.find(key);
			if (it == edgeScoreMap.end()) edgeScoreMap[key] = d;
			else it->second = std::min(it->second, d);
		}
	}

	if (enable_debug) {
		std::cout << "[1단계] 초기 간선 수집: " << edgeSet.size() << " 개" << std::endl;
		std::cout << "[1단계] 초기 간선 수집:" << edgeScoreMap.size() << " initial edges\n";

	}


	std::set<std::pair<int, int>> final_edges = edgeSet;

	GraphQualityMetrics quality = evaluateGraphQuality(final_edges, N);
	if (enable_debug) {
		std::cout << "[4단계] 품질 평가 - 최대 컴포넌트 비율: "
			<< quality.largest_component_ratio << std::endl;
	}

//	// 품질이 매우 부족한 경우에만 제한적 추가 (과연결 방지)
//	if (quality.largest_component_ratio < 0.5f && quality.num_components > 20) {
//		if (enable_debug) {
//			std::cout << "[5단계] 제한적 간선 추가 진행..." << std::endl;
//		}
//
//		// 매우 보수적인 추가 (기존 1.3배 증가 대신 1.1배만)
//		float conservative_threshold = 0.15f;
//		std::set<std::pair<int, int>> additional_edges;
//
//#pragma omp parallel
//		{
//			std::vector<size_t> neigh(10); // 더 작은 K값
//			std::vector<float> d2(10);
//			std::set<std::pair<int, int>> localSet;
//
//#pragma omp for
//			for (int i = 0; i < N; ++i) {
//				float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
//				size_t found = kdtree.knnSearch(q, 10, neigh.data(), d2.data());
//
//				for (size_t k = 1; k < found; ++k) {
//					int j = static_cast<int>(neigh[k]);
//					if (j <= i) continue;
//
//					std::pair<int, int> edge_key = { std::min(i, j), std::max(i, j) };
//					if (final_edges.count(edge_key)) continue;
//
//					float dist = std::sqrt(d2[k]);
//					if (dist < conservative_threshold) {
//						localSet.insert(edge_key);
//						if (localSet.size() > 5) break; // 정점당 최대 5개 추가만
//					}
//				}
//			}
//
//			std::lock_guard<std::mutex> lk(edge_mtx);
//			additional_edges.insert(localSet.begin(), localSet.end());
//		}
//
//		final_edges.insert(additional_edges.begin(), additional_edges.end());
//		if (enable_debug) {
//			std::cout << "  추가된 간선: " << additional_edges.size() << " 개" << std::endl;
//		}
//	}

	// 7) 최종 edges 벡터 생성
	//edges.clear();
	//edges.reserve(final_edges.size());

	//for (const auto& pr : final_edges) {
	//	float d = (pts.row(pr.first) - pts.row(pr.second)).norm();
	//	float stiff = std::exp(-(d * d) / (d_threshold * d_threshold));
	//	edges.emplace_back(pr.first, pr.second, d, stiff);
	//}
	// 6) 최종 edges 벡터 작성
	edges.clear();
	edges.reserve(edgeScoreMap.size());

	for (auto& kv : edgeScoreMap)
	{
		int a = kv.first.first;
		int b = kv.first.second;
		float score_d = kv.second;

		// rest length는 실제 거리
		float restLen = (pts.row(a) - pts.row(b)).norm();

		// stiffness는 score_d 기반으로 줄 수도 있고, restLen 기반으로 줄 수도 있음
		const float safeThreshold = std::max(d_threshold, 1e-6f);
		const float x = std::max(0.0f, score_d / safeThreshold);
		const float sim = std::exp(-(x * x));
		const float stiff = std::clamp(0.05f + 0.95f * sim, 0.05f, 1.0f);

		edges.emplace_back(a, b, restLen, stiff);
	}
	// 8) 최종 결과 리포트
	quality = evaluateGraphQuality(final_edges, N);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	if (enable_debug) {
		std::cout << "\n===  결과 ===" << std::endl;
		std::cout << "최종 간선 수: " << edges.size() << std::endl;
		std::cout << "간선 밀도: " << static_cast<float>(edges.size()) / (N * (N - 1) / 2) << std::endl;
		std::cout << "최대 컴포넌트 비율: " << quality.largest_component_ratio << std::endl;
		std::cout << "총 처리 시간: " << duration.count() << "ms" << std::endl;

		// 권장 밀도 범위 체크
		float edge_density = static_cast<float>(edges.size()) / N;
		if (edge_density > 20.0f) {
			std::cout << "[경고] 간선 밀도가 높습니다 (정점당 " << edge_density << " 개)" << std::endl;
		}
		else {
			std::cout << "[정상] 적절한 간선 밀도 (정점당 " << edge_density << " 개)" << std::endl;
		}
	}
}

void buildfunctionEdgeGraph(
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot,
	const std::vector<sibr::Scale>& scales,
	std::vector<sibr::Edge>& edges,
	int default_k,
	float d_threshold,
	bool enable_debug = true)
{
	const int N = static_cast<int>(pos.size());
	
	auto start_time = std::chrono::high_resolution_clock::now();
	if (enable_debug) {
		std::cout << "\n=== 통합 거리 기반 그래프 구축 시작 ===" << std::endl;
		std::cout << "입력 정점 수: " << pos.size() << std::endl;
		std::cout << "KNN 후보 수: " << default_k << std::endl;
		std::cout << "통합 임계값 d: " << d_threshold << std::endl;
	}
	// 1) KD-Tree 준비
	Eigen::MatrixXf pts(N, 3);
	for (int i = 0; i < N; ++i)
		pts.row(i) = Eigen::Vector3f(pos[i].x(), pos[i].y(), pos[i].z());

	using KDAdaptor = KDTreeAdaptor<Eigen::MatrixXf>;
	using KDType = nanoflann::KDTreeSingleIndexAdaptor<
		nanoflann::L2_Simple_Adaptor<float, KDAdaptor>,
		KDAdaptor, 3>;
	KDAdaptor adaptor(pts);
	KDType kdtree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	kdtree.buildIndex();

	const int max_k = std::min(20, default_k + 8);
	const float w_dist = 1.0f;
	const float w_ori = 0.5f;
	const float w_shape = 0.4f;
	// 진행률 리포터
	OptimizedProgressReporter progress(pos.size());
	std::atomic<int> progress_counter{ 0 };
	// 2) KNN 기반 초기 간선 수집 (OpenMP 병렬)
	std::set<std::pair<int, int>> edgeSet;
	std::vector<float>score;
	std::map<std::pair<int, int>, float> edgeScoreMap;

	std::mutex edge_mtx;
#pragma omp parallel
	{
		//neigh[k]: k번째 이웃의 입력 데이터 기준 인덱스 (0 ~ N-1) 를 의미
		std::vector<size_t> neigh(max_k + 1);
		std::vector<float>  d2(max_k + 1);
		std::map<std::pair<int, int>, float> localMap;

		std::set<std::pair<int, int>> localSet;
		std::vector<float>localscore;

#pragma omp for schedule(dynamic,100)
		for (int i = 0; i < N; ++i) {//N 은 정점개수
			float AR_i = aspectRatio(scales[i]);
			float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
			size_t found = kdtree.knnSearch(q, max_k + 1, neigh.data(), d2.data());
			//found 는 i 번째 정점에서 찾은 이웃들 모임
			for (size_t k = 1; k < found; ++k) {//한정점에서 찾은 이웃들 개수만큼 반복
				int j = static_cast<int>(neigh[k]);//k번째 이웃의 인덱스가 들어있다
				if (j <= i) continue;
				float dist = std::sqrt(d2[k]);
				float oSim = orientationSim(rot[i], rot[j]);
				float AR_j = aspectRatio(scales[j]);
				float d = w_dist * dist + 
					w_ori * (1.0f - oSim) +
					w_shape * std::abs(AR_i - AR_j);
				if (d <= d_threshold) {//d 값이 임계값보다 작으면 간선추가
					int a = i, b = j;
					if (a > b) std::swap(a, b);
					auto key = std::make_pair(a, b);

					// localMap에 이미 있으면 더 좋은 값(예: 더 작은 d) 유지
					auto it = localMap.find(key);
					if (it == localMap.end()) localMap[key] = d;
					else it->second = std::min(it->second, d);
					localSet.emplace(a, b);
				}
			}
		}

		std::lock_guard<std::mutex> lk(edge_mtx);
		edgeSet.insert(localSet.begin(), localSet.end());
		for (auto& kv : localMap)
		{
			auto key = kv.first;
			float d = kv.second;

			auto it = edgeScoreMap.find(key);
			if (it == edgeScoreMap.end()) edgeScoreMap[key] = d;
			else it->second = std::min(it->second, d);
		}

	}

	if (enable_debug) {
		std::cout << "[Step 2] Collected " << edgeSet.size() << " initial edges\n";
		std::cout << "[Step 2] Collected " << edgeScoreMap.size() << " initial edges\n";
	}

	// 6) 최종 edges 벡터 작성
	edges.clear();
	edges.reserve(edgeScoreMap.size());

	for (auto& kv : edgeScoreMap)
	{
		int a = kv.first.first;
		int b = kv.first.second;
		float score_d = kv.second;

		// rest length는 실제 거리
		float restLen = (pts.row(a) - pts.row(b)).norm();

		// stiffness는 score_d 기반으로 줄 수도 있고, restLen 기반으로 줄 수도 있음
		const float safeThreshold = std::max(d_threshold, 1e-6f);
		const float x = std::max(0.0f, score_d / safeThreshold);
		const float sim = std::exp(-(x * x));
		const float stiff = std::clamp(0.05f + 0.95f * sim, 0.05f, 1.0f);

		edges.emplace_back(a, b, restLen, stiff);
	}
	
	//edges.clear();
	//edges.reserve(edgeSet.size());
	//for (auto& pr : edgeSet) {
	//	float d = (pts.row(pr.first) - pts.row(pr.second)).norm();
	//	float stiff = std::exp(-(d * d) / (d_threshold * d_threshold));
	//	edges.emplace_back(pr.first, pr.second, d, stiff);
	//}
	if (enable_debug) {
		auto total_end = std::chrono::high_resolution_clock::now();
		auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - start_time);
		std::cout << "\n=== 통합 거리 기반 그래프 구축 완료 ===" << std::endl;
		std::cout << "총 처리 시간: " << total_duration.count() << "ms" << std::endl;
		std::cout << "생성된 간선 수: " << edges.size() << std::endl;
		std::cout << "==========================================\n" << std::endl;
	}
	if (enable_debug) {
		auto t1 = std::chrono::high_resolution_clock::now();
		
	}
}




void buildFullEdgeGraph(
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot,
	std::vector<sibr::Edge>& edges,
	int default_k,
	float max_distance_threshold,
	const glm::vec3& pivot,
	float radius,
	bool use_delaunay = false
) {
	static int call_count = 0;
	SIBR_LOG << "buildFullEdgeGraph called: " << ++call_count << " times\n";
	// 1. 반경 내 점만 추출 및 인덱스 맵핑
	/*std::vector<Pos> filtered_pos;
	std::vector<Rot> filtered_rot;
	std::map<int, int> old_to_new_idx;
	int new_idx = 0;
	float radius_sq = radius * radius;
	for (size_t i = 0; i < pos.size(); ++i) {
		float dx = pos[i].x() - pivot.x;
		float dy = pos[i].y() - pivot.y;
		float dz = pos[i].z() - pivot.z;
		float dist_sq = dx * dx + dy * dy + dz * dz;
		if (dist_sq < radius_sq) {
			filtered_pos.push_back(pos[i]);
			filtered_rot.push_back(rot[i]);
			old_to_new_idx[i] = new_idx++;
		}
	}*/
	// 1. Eigen 행렬 변환
	Eigen::MatrixXf points(pos.size(), 3);
	for (size_t i = 0; i < pos.size(); ++i)
		points.row(i) = pos[i];

	// 2. KD-Tree 생성
	typedef KDTreeAdaptor<Eigen::MatrixXf> MyKDTreeAdaptor;
	using MyKDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
		nanoflann::L2_Simple_Adaptor<float, MyKDTreeAdaptor>,
		MyKDTreeAdaptor, 3>;
	MyKDTreeAdaptor adaptor(points);
	MyKDTreeType index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	index.buildIndex();

	// 3. KNN 및 엣지 생성
	edges.clear();
	std::set<std::pair<int, int>> edge_set; // 중복 방지
	// 최대 adaptive_k(8) + 1 = 9 크기로 할당
	const int max_k = 8;
	std::vector<size_t> indices(max_k + 1);
	std::vector<float> dists(max_k + 1);
	float global_max_dist = 0.0f;

	for (size_t i = 0; i < pos.size(); ++i) {
		const float query[3] = { pos[i][0], pos[i][1], pos[i][2] };
		index.knnSearch(query, 6 + 1, &indices[0], &dists[0]);
		for (size_t j = 1; j < indices.size(); ++j) {
			int ni = static_cast<int>(indices[j]);
			if (ni == i) continue; // 자기 자신 제외
			if (ni < 0 || ni >= (int)pos.size()) continue; // 인덱스 범위 체크
			int v0 = std::min<int>(i, ni);
			int v1 = std::max<int>(i, ni);
			if (v0 == v1) continue; // 0-0 등 자기 자신 제외
			if (edge_set.count({ v0, v1 })) continue; // 중복 방지
			edge_set.insert({ v0, v1 });
			float restlen = std::sqrt(dists[j]);
			edges.emplace_back(v0, v1, restlen, 1.0f);
		}
	}

	////  적응적 k를 KD-Tree radius search로 최적화
	//const float search_radius = 5.0f;
	//const float search_radius_sq = search_radius * search_radius;

	//for (size_t i = 0; i < pos.size(); ++i) {
	//	// 진행상황 로그 (올바른 위치)
	//	if (i % 1000 == 0) {
	//		SIBR_LOG << "KNN progress: " << i << " / " << pos.size() << std::endl;
	//	}

	//	//  KD-Tree radius search로 적응적 k 계산
	//	std::vector<std::pair<size_t, float>> ret_matches;
	//	nanoflann::SearchParams params;
	//	float query_pt[3] = { points(i, 0), points(i, 1), points(i, 2) };
	//	size_t nMatches = index.radiusSearch(query_pt, search_radius_sq, ret_matches, params);

	//	int neighbors_in_radius = static_cast<int>(nMatches) - 1; // 자기 자신 제외
	//	int adaptive_k = (neighbors_in_radius > 20) ? 4 :
	//		(neighbors_in_radius > 10) ? 6 : 8;

	//	// adaptive_k가 max_k를 초과하지 않도록 제한
	//	adaptive_k = std::min(adaptive_k, max_k);

	//	// KNN 검색
	//	index.knnSearch(&points(i, 0), adaptive_k + 1, &indices[0], &dists[0]);

	//	for (int j = 1; j <= adaptive_k; ++j) {
	//		int ni = static_cast<int>(indices[j]);
	//		if (ni == i || ni < 0 || ni >= static_cast<int>(pos.size())) continue;

	//		int v0 = std::min(static_cast<int>(i), ni);
	//		int v1 = std::max(static_cast<int>(i), ni);
	//		if (v0 == v1 || edge_set.count({ v0, v1 })) continue;

	//		float restlen = std::sqrt(dists[j]);
	//		if (restlen > max_distance_threshold) continue;

	//		edge_set.insert({ v0, v1 });
	//		edges.emplace_back(v0, v1, restlen, 1.0f);
	//		global_max_dist = std::max(global_max_dist, restlen);
	//	}
	//}

	SIBR_LOG << "KNN completed. Total edges: " << edges.size() << std::endl;

	// 5. Delaunay edge 추가 (필요시)
	/*if (use_delaunay) {
		SIBR_LOG << "Adding Delaunay edges..." << std::endl;
		addDelaunayEdges(filtered_pos, edge_set, edges);
		SIBR_LOG << "Delaunay completed. Total edges: " << edges.size() << std::endl;
	}*/

	// 6. 가중치 할당
	SIBR_LOG << "Assigning weights..." << std::endl;


	//
	//// 1) 모든 점에 대해 edge 후보 생성 (적응적 k)
	//for (size_t i = 0; i < filtered_pos.size(); ++i) {
	//	// 적응적 k 계산
	//	int neighbors_in_radius = 0;
	//	const float search_radius = 5.0f;
	//	for (size_t j = 0; j < filtered_pos.size(); ++j) {
	//		if (i == j) continue;
	//		float dx = filtered_pos[i].x() - filtered_pos[j].x();
	//		float dy = filtered_pos[i].y() - filtered_pos[j].y();
	//		float dz = filtered_pos[i].z() - filtered_pos[j].z();
	//		float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
	//		if (dist < search_radius) neighbors_in_radius++;
	//	}
	//	int adaptive_k = (neighbors_in_radius > 20) ? 4 :
	//		(neighbors_in_radius > 10) ? 6 : 8;

	//	// KNN 검색
	//	index.knnSearch(&points(i, 0), adaptive_k + 1, &indices[0], &dists[0]);
	//	for (int j = 1; j < adaptive_k + 1; ++j) {
	//		int ni = static_cast<int>(indices[j]);
	//		if (ni == i || ni < 0 || ni >= static_cast<int>(filtered_pos.size())) continue;
	//		int v0 = std::min(static_cast<int>(i), ni);
	//		int v1 = std::max(static_cast<int>(i), ni);
	//		if (v0 == v1 || edge_set.count({ v0, v1 })) continue;

	//		float restlen = std::sqrt(dists[j]);
	//		if (restlen > max_distance_threshold) continue;

	//		edge_set.insert({ v0, v1 });
	//		edges.emplace_back(v0, v1, restlen, 1.0f);
	//		global_max_dist = std::max(global_max_dist, restlen);
	//		for (size_t i = 0; i < filtered_pos.size(); ++i) {
	//			if (i % 10000 == 0) {
	//				//SIBR_LOG << "KNN progress: " << i << " / " << filtered_pos.size() << std::endl;
	//			}
	//		}
	//	}
	//}

	//// 4. Delaunay edge 추가 (필요시)
	//if (use_delaunay) {
	//	addDelaunayEdges(filtered_pos, edge_set, edges); //filtered_pos 전달
	//}

	//// 5. 가중치 할당
	//for (auto& e : edges) {
	//	e.st = computeDistanceWeight(e.rl, global_max_dist);
	//	float orient_weight = computeOrientationWeight(
	//		filtered_rot[e.m_vert[0]], filtered_rot[e.m_vert[1]],
	//		filtered_pos[e.m_vert[0]], filtered_pos[e.m_vert[1]]
	//	);
	//	e.st *= orient_weight;
	//}



	SIBR_LOG << "buildFullEdgeGraph completed successfully!" << std::endl;

}


#endif
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


	//// 2. 크롭 범위 설정
	//Eigen::Vector3f crop_min1(-0.197f, -1.556f, 2.799f);//４３５５８　44704 ,43194 data1 44986
	//Eigen::Vector3f crop_max1(2.604f, 3.240f, 6.635f);
	//
	//Eigen::Vector3f crop_min2(-1.617f, -0.577f, 0.487f);
	//Eigen::Vector3f crop_max2(-0.402f, 1.981f, 1.766f);
	//
	//
	//Eigen::Vector3f crop_min3(-1.6f, 0.307f, 0.487f);
	//Eigen::Vector3f crop_max3(-0.613f, 1.916f, 1.766f);
	//
	//Eigen::Vector3f crop_min4(0.397f, -2.256f, 4.519f);
	//Eigen::Vector3f crop_max4(5.004f, 4.240f, 8.535f);
	//
	//Eigen::Vector3f crop_min5(-9.197f, -2.556f, 5.519f);
	//Eigen::Vector3f crop_max5(5.004f, 4.240f, 7.635f);
	//
	//
	//Eigen::Vector3f crop_min8(-2.197f, -0.816f, 0.589f);//21488 ,21824 
	//Eigen::Vector3f crop_max8(-0.009f, 1.590f, 3.039f);
	//
	//Eigen::Vector3f crop_min9(0.097f, -1.216f, 1.429f);//60628
	//Eigen::Vector3f crop_max9(2.79f, 2.390f, 4.45f);
	//
	//
	//////////////////////////////////////////////////////
	//Eigen::Vector3f crop_min10(-0.797f, -1.916f, 1.129f);//25245 , 25319 , 24502 , 17003 , 17449 , 16929
	//Eigen::Vector3f crop_max10(1.909f, 1.990f, 4.445f);//13513 눈썹실험
	//
	//Eigen::Vector3f crop_min11(-3.097f, -0.816f, 1.129f);//46078점빼기　예시３８２８１코
	//Eigen::Vector3f crop_max11(0.009f, 3.390f, 4.445f);//49329볼　늘리기예시２４６３１
	//
	//Eigen::Vector3f crop_min12(-2.197f, -1.616f, 3.429f);
	//Eigen::Vector3f crop_max12(1.179f, 1.390f, 5.45f);
	//
	//Eigen::Vector3f crop_min13(-2.197f, -2.916f, 3.129f);//17479오른쪽귀
	//Eigen::Vector3f crop_max13(2.210f, 1.390f, 5.45f);//7240왼쪽귀
	//
	//Eigen::Vector3f crop_min14(-1.597f, -1.616f, 1.829f);//２８５８３
	//Eigen::Vector3f crop_max14(1.179f, 1.390f, 4.345f);//소진30104 광대14393 눈썹3953코
	//
	//Eigen::Vector3f crop_min15(-0.297f, -1.516f, 1.529f);// 84398 , 83975 안경당기기   74888안경13892앞머리
	//Eigen::Vector3f crop_max15(3.179f, 2.590f, 5.452f);//66143안경　코56605　　６１７８９
	//
	//Eigen::Vector3f crop_min16(-3.097f, -1.816f, 1.129f);//여우 데이터20894 여우오른쪽귀 0왼쪽귀29441배
	//Eigen::Vector3f crop_max16(0.009f, 1.890f, 3.645f);//13484 13500　코 늘리기예시 17787 눈알인덱스 17722
	//
	//Eigen::Vector3f crop_min17(-5.097f, -1.099f, -2.129f);//여우 데이터
	//Eigen::Vector3f crop_max17(0.009f, 0.590f, 2.645f);//13484　코 늘리기예시 17787 눈알인덱스 17722
	//
	//Eigen::Vector3f crop_min18(-0.627f, 1.099f, 0.929f);//ｍｉｃ데이터
	//Eigen::Vector3f crop_max18(1.309f, 4.390f, 3.245f);//３７４０１　
	//
	//
	//Eigen::Vector3f crop_min19(-3.097f, -1.816f, 1.129f);
	//Eigen::Vector3f crop_max19(1.309f, 3.500f, 3.645f);
	//
	//
	//Eigen::Vector3f crop_min20(-3.097f, -0.716f, -3.529f);
	//Eigen::Vector3f crop_max20(6.309f, 0.500f, 2.345f);
	// 수정 (cfg에서 읽어옴)
	Eigen::Vector3f crop_min(cfg.crop_min[0], cfg.crop_min[1], cfg.crop_min[2]);
	Eigen::Vector3f crop_max(cfg.crop_max[0], cfg.crop_max[1], cfg.crop_max[2]);
	int min_idx = 1000000; // 충분히 큰 값으로 초기화
	int max_idx = -1;
	int count_match = 0;
	// 3. 크롭된 데이터 저장용
	std::vector<sibr::Pos> cropped_pos;
	std::vector<FORWARD::Pos> FORWARD_cropped_pos;
	std::vector<Rot> cropped_rot;
	std::vector<Scale> cropped_scale;
	std::vector<SHs<3>> cropped_shs;
	std::vector<float> cropped_opacity;
	// CPU 로딩부에서
	// 전역 또는 클래스 멤버 변수로 선언
	int roi_start_idx = 0;
	int roi_end_idx = 0;
	int actual_cropped_count = 0;
	float min_dist_to_target = 1e10f;
	int best_cheek_idx = -1;
	int best_cheek_relative_idx = -1; // 원본 인덱스가 아닌 '상대적 순번' 저장용

	// 로딩 루프
	actual_cropped_count = 0;
	int first_found = -1;
	int last_found = -1;
	std::vector<int> roi_indices; // ROI에 포함된 원본 인덱스들을 저장
	for (int i = 0; i < count; ++i) {
		const auto& p = pos[i];
		if (p.x() >= crop_min.x() && p.x() <= crop_max.x() &&
			p.y() >= crop_min.y() && p.y() <= crop_max.y() &&
			p.z() >= crop_min.z() && p.z() <= crop_max.z() &&
			opacity[i] != 0)
		{
			cropped_pos.push_back(pos[i]);
			FORWARD_cropped_pos.push_back(FORWARD::Pos(pos[i].x(), pos[i].y(), pos[i].z()));
			cropped_rot.push_back(rot[i]);
			cropped_scale.push_back(scale[i]);
			cropped_shs.push_back(shs[i]);
			cropped_opacity.push_back(opacity[i]);
		}
	}
	// 3. 할당 (이제 180063 대신 0~44709 사이의 안전한 값이 들어감)
	if (best_cheek_relative_idx != -1) {
		_deformPoint2 = best_cheek_relative_idx;
		printf("[Safe Target Found] Cropped 리스트 내 볼 중심 순번: %d\n", _deformPoint2);
	}
	
	
	cfg.print();  // 콘솔에 현재 파라미터 출력 (로그 남기기용)

	cropped_edges.clear();
	d_thresholdP = cfg.d_thresholdP;  // 기존 변수도 동기화
	default_k = cfg.default_k;

	
	_graphBuilder->build(
		cropped_pos,
		cropped_rot,
		cropped_scale,
		FORWARD_cropped_edges,
		cropped_shs,
		cfg,
		true   // enable_debug
		);
	_potinSize = cropped_pos.size();

	


	//SetChainMailGraphFromVectors(cropped_pos, cropped_edges, cropped_opacity);


	////ChainMail cm;
	_cm.loadGraph(_cm, FORWARD_cropped_pos, FORWARD_cropped_edges, cropped_opacity);
	rebuildGraphDebugEdges();

	//for(int i=0;i<20;i++)cm.relax(activeSet);

	

	/*for (auto& e : cm.elements) {
		elem_pos.push_back(e.pos);
		elem_density.push_back(e.density);
		elem_time.push_back(e.time);
		elem_offset.push_back(e.offset);
		elem_neighborCnt.push_back(e.neighborCnt);
	}

	
	for (auto& n : cm.neighbors) {
		neighbor_idx.push_back(n.idx);
		neighbor_dist.push_back(n.dist);
	}*/

	// 4. 변형 결과 반영
	/*for (int i = 0; i < cropped_pos.size(); ++i) {
		cropped_pos[i] = sibr::Vector3f(
			cm.getElement(i).pos.x,
			cm.getElement(i).pos.y,
			cm.getElement(i).pos.z
		);
	}*/
	// 5. GPU 업로드
	P = cropped_pos.size();
	count = P;

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, cropped_pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, cropped_rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, cropped_shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, cropped_opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, cropped_scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));



	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(P,
		(float*)cropped_pos.data(),
		(float*)cropped_rot.data(),
		(float*)cropped_scale.data(),
		cropped_opacity.data(),
		(float*)cropped_shs.data());

	_gaussianRenderer = new GaussianSurfaceRenderer();

	//chainmail


	glm::vec3 pivot(0.0f, 0.0f, 2.0f);
	float radius = 3.0f;



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
			default_k,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			
			_rotatingModifier_COV3D_Matrix_x,
			_rotatingModifier_COV3D_Matrix_y,
			_rotatingModifier_COV3D_Matrix_z,
			_rotatingModifier_COV2D_Matrix_x,
			_rotatingModifier_COV2D_Matrix_y,
			_rotatingModifier_COV2D_Matrix_z,

			_pivotRotX,
			_pivotRotY,
			_pivotRotZ,

			t,
			_wave,
			_twist,
			_bubble,

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
				_deformPoint1 = _pickedId;
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
		//ImGui::SliderFloat("Rotating Modifier X (COV3D Matrix R)", &_rotatingModifier_COV3D_Matrix_x, 0.001f, 50.0f);
		//ImGui::SliderFloat("Rotating Modifier Y (COV3D Matrix R)", &_rotatingModifier_COV3D_Matrix_y, 0.001f, 50.0f);
		//ImGui::SliderFloat("Rotating Modifier Z (COV3D Matrix R)", &_rotatingModifier_COV3D_Matrix_z, 0.001f, 50.0f);
		//ImGui::SliderFloat("Rotating Modifier X (COV2D Matrix W)", &_rotatingModifier_COV2D_Matrix_x, 0.001f, 50.0f);
		//ImGui::SliderFloat("Rotating Modifier Y (COV2D Matrix W)", &_rotatingModifier_COV2D_Matrix_y, 0.001f, 50.0f);
		//ImGui::SliderFloat("Rotating Modifier Z (COV2D Matrix W)", &_rotatingModifier_COV2D_Matrix_z, 0.001f, 50.0f);
		//ImGui::SliderFloat("Pivot Rot X", &_pivotRotX, -180.f, 180.f);
		//ImGui::SliderFloat("Pivot Rot Y", &_pivotRotY, -180.f, 180.f);
		//ImGui::SliderFloat("Pivot Rot Z", &_pivotRotZ, -180.f, 180.f);
		ImGui::SliderInt("_deformPoint ", &_deformPoint1, 0, _potinSize);
		ImGui::SliderInt("_deformPoint2 ", &_deformPoint2, 0, _potinSize);
		ImGui::SliderInt("_deformseeds ", &_deformseeds, 0, _potinSize);

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

	ImGui::Checkbox("wave", &_wave);
	ImGui::Checkbox("AdaptiveCovariance", &_twist);
	ImGui::Checkbox("bubble", &_bubble);

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
	const float ratio = (_potinSize > 0) ? (100.0f * float(activeCount) / float(_potinSize)) : 0.0f;
	ImGui::Text("Active ratio: %.2f%% (%d/%d)", ratio, activeCount, _potinSize);

	if (ImGui::Button("buildFullEdgeGraph(K-NN)"))
	{
	//	glm::vec3 pivot(-2.0f, 1.0f, 1.0f);
	//	float radius = 2.0f;


	//	buildFullEdgeGraph(
	//	pos, rot, edges, 6, 8.0f, pivot, radius
	//);
		//// 4. GPU 메모리 할당 및 복사
		////num_edges = edges.size();
		////CUDA_SAFE_CALL(cudaMalloc(&d_edges, num_edges * sizeof(Edge)));
		////CUDA_SAFE_CALL(cudaMemcpy(
		////	d_edges, edges.data(),
		////	num_edges * sizeof(Edge),
		////	cudaMemcpyHostToDevice
		////));
	

	}

	if (ImGui::Button("movePoints")) {

		_cm.FPS = true;
		_cm.resetTime();
		//activeSet.clear();//결과 배열
		//cm.startWave(_deformPoint1, glm::vec3(0.25, 0, 0), activeSet);
		_cm.movePointPos(&_deformPoint1, glm::vec3(0.25f, 0, 0.00f), _activeSet);
		//cm.propagate(activeSet);

		//cm.FPS = true;
		//cm.resetTime();
		//activeSet.clear();
		//
		//// 1. 초기화 및 상태 설정
		//int seed = _deformPoint1; // 선택된 점
		//glm::vec3 delta(0.25f, 0.0f, 0.0f);
		//
		//// 2. Task 설정 (시작점과 도착점 기록)
		//cm.singleDeformTask.isRunning = true;
		//cm.singleDeformTask.seedIdx = seed;
		//cm.singleDeformTask.startPos = cm.getElement(seed).pos;
		//cm.singleDeformTask.targetPos = cm.getElement(seed).pos + delta;
		//cm.singleDeformTask.progress = 0.0f;
		//
		//// startWave는 이제 즉시 이동이 아니라 초기화 용도로만 사용 (delta는 0으로)
		////cm.startWave(seed, glm::vec3(0.0f), activeSet);
		//cm.setRunning(true);
	}
	if (ImGui::Button("RmovePoints")) {
		_cm.FPS = true;
		_cm.resetTime();
		_activeSet.clear();//결과 배열
		_cm.startWave(_deformPoint2, glm::vec3(0, 0, 0.011), _activeSet);

		//cm.movePointPos(&_deformPoint1, glm::vec3(0.25f, 0.0f, 0.00f), activeSet);
		//cm.propagate(activeSet);



	}
	if (ImGui::Button("LmovePoints")) {
		_cm.FPS = true;
		_cm.resetTime();
		_activeSet.clear();//결과 배열
		_cm.startWave(_deformPoint1, glm::vec3(-0.25, 0, 0.0), _activeSet);

		//cm.movePointPos(&_deformPoint1, glm::vec3(0.25f, 0.0f, 0.00f), activeSet);
		//cm.propagate(activeSet);



	}
	if (ImGui::Button("moveMultiplePoints")) {
		_cm.FPS = true;

		_cm.resetTime();
		_activeSet.clear();

		auto seeds = _cm.collectSeedsBFS(_deformPoint1, _deformseeds); // 50개 영역 선택
		_cm.startWaveMultiple(seeds, glm::vec3(0, 0, 0.03), _activeSet);

	}
	if (ImGui::Button("RmoveMultiplePoints")) {
		_cm.FPS = true;

		_cm.resetTime();
		_activeSet.clear();

		auto seeds = _cm.collectSeedsBFS(_deformPoint2, _deformseeds); // 50개 영역 선택
		_cm.startWaveMultiple(seeds, glm::vec3(0, 0, 0.03), _activeSet);

	}	if (ImGui::Button("LmoveMultiplePoints")) {
		_cm.FPS = true;

		_cm.resetTime();
		_activeSet.clear();

		auto seeds = _cm.collectSeedsBFS(_deformPoint1, _deformseeds); // 50개 영역 선택
		_cm.startWaveMultiple(seeds, glm::vec3(0.001f, -0.0001, -0.1), _activeSet);

	}

	if (ImGui::Button("movebackMultiplePoints")) {
		_cm.resetTime();
		_activeSet.clear();

		auto seeds = _cm.collectSeedsBFS(_deformPoint2, _deformseeds); // 50개 영역 선택
		_cm.startWaveMultiple(seeds, glm::vec3(0.1f, -0.1f, 0.1f), _activeSet);

	}

	ImGui::Text("2. Register Body Parts");
	// [왼쪽 귀 등록 버튼]
	if (ImGui::Button("Add as Left Ear")) {
		// 1. 현재 슬라이더 값으로 영역(Seeds) 추출
		auto seeds = _cm.collectSeedsBFS(_deformPoint2, _deformseeds);

		// 2. 피벗(Pivot) 설정: 간단하게 '시작점(_deformPoint1)'의 위치를 회전축으로 사용
		// (이 점을 중심으로 귀가 회전하게 됨)
		glm::vec3 pivot = _cm.getElement(_deformPoint2).pos;

		// 3. 그룹 추가
		_cm.addSeedGroup("LeftEar", seeds, pivot);

		printf("Left Ear Added: %d vertices\n", (int)seeds.size());
	}

	ImGui::SameLine(); // 버튼 옆으로 나란히

	// [오른쪽 귀 등록 버튼]
	if (ImGui::Button("Add as Right Ear")) {
		// 사용자가 슬라이더를 오른쪽 귀 위치로 옮긴 상태라고 가정
		auto seeds = _cm.collectSeedsBFS(_deformPoint1, _deformseeds);
		glm::vec3 pivot = _cm.getElement(_deformPoint1).pos;

		_cm.addSeedGroup("RightEar", seeds, pivot);

		printf("Right Ear Added: %d vertices\n", (int)seeds.size());
	}
	// [오른쪽 귀 등록 버튼]
	if (ImGui::Button("Add as body")) {
		// 사용자가 슬라이더를 오른쪽 귀 위치로 옮긴 상태라고 가정
		auto seeds = _cm.collectSeedsBFS(_deformPoint1, _deformseeds);
		glm::vec3 pivot = _cm.getElement(_deformPoint1).pos;

		_cm.addSeedGroup("body", seeds, pivot);

		printf("body Added: %d vertices\n", (int)seeds.size());
	}

	// [초기화 버튼] - 잘못 등록했을 때 다 지우기
	if (ImGui::Button("Clear All Groups")) {
		_cm.clearSeedGroups();
		_cm.setRunning(false); // 실행 중이었다면 정지
	}
	ImGui::Separator();

	// -------------------------------------------------------------
	// 3. 실행 버튼 (애니메이션 시작)
	// -------------------------------------------------------------
	ImGui::Text("3. Animation Control");
	if (ImGui::Button("movingMultiplePoints")) {
		// 이미 위에서 등록을 마쳤으므로 여기서는 실행 플래그만 켭니다.
		if (_cm.getSeedGroups().empty()) {
			printf("Error: No groups registered! Add ears first.\n");
		}
		else {
			_cm.resetTime();
			_activeSet.clear();
			_cm.setRunning(true);
		}
		//cm.resetTime();
		//activeSet.clear();
		//
		//auto seeds = cm.collectSeedsBFS(_deformPoint1, _deformseeds);
		//// 50개 영역 선택
		//cm.setSeeds(seeds);   //  여기 중요
		////cm.startWaveMultiple(seeds, glm::vec3(0, 0, 0), activeSet);
		//cm.setRunning(true);


		//float d = amp * sin(t * speed);
		//cm.startWavingMultiple(seeds, glm::vec3(d, 0, 0), activeSet);
	}

	if (ImGui::Button("stopmovingMultiplePoints")) {
		
		_cm.setRunning(false);

		
	}
	if (ImGui::Button("stopFPS")) {

		_cm.FPS=false;


	}
	//if (ImGui::Button("frontMovePoints")) {
	//	cm.resetTime();
	//	activeSet.clear();//결과 배열


	//	cm.movePointPos(&_deformPoint1, glm::vec3(0.15f, 0.00f, -0.15f), activeSet);
	//	cm.propagate(activeSet);


	//}
	//if (ImGui::Button("downMovePoints")) {
	//	cm.resetTime();
	//	activeSet.clear();//결과 배열


	//	cm.movePointPos(&_deformPoint1, glm::vec3(0.00f, 0.00f, 0.25f), activeSet);
	//	cm.propagate(activeSet);


	//}
	//if (ImGui::Button("backmovePoints")) {
	//	cm.resetTime();
	//	activeSet.clear();//결과 배열
	//	

	//	cm.movePointPos(&_deformPoint1, glm::vec3(-0.25f, 0.0f, 0.00f), activeSet);
	//	cm.propagate(activeSet);


	//}
	//if (ImGui::Button("Loop Test")) {
	//	cm.resetTime();
	//	activeSet.clear();//결과 배열
	//	cm.movePointPos(&_deformPoint1, glm::vec3(0.55f, 0.0f, 0.00f), activeSet);
	//	cm.propagate(activeSet);

	//}
	static float prev_thresholdP = d_thresholdP;

	bool changed = ImGui::InputFloat("My Float", &d_thresholdP, 5.0f, 10.0f);
	ImGui::Text("Current value: %.3f", d_thresholdP);
	if ( d_thresholdP != prev_thresholdP)
	{

		if (ImGui::Button("buildfunctionEdgeGraph"))
		{


			//idx_map.clear();
			//cropped_pos.clear();
			//cropped_rot.clear();
			//cropped_scale.clear();
			//cropped_shs.clear();
			//cropped_edges.clear();  // 혹시 이전 edge도 재사용되면 이것도 clear!
			//cropped_opacity.clear();
			//for (int i = 0; i < pos.size(); ++i) {
			//	const auto& p = pos[i];
			//	if (p.x() >= _boxmin.x() && p.x() <= _boxmax.x() &&
			//		p.y() >= _boxmin.y() && p.y() <= _boxmax.y() &&
			//		p.z() >= _boxmin.z() && p.z() <= _boxmax.z()) {
			//		if (opacity[i] != 0) {
			//			idx_map.push_back(i);
			//			cropped_pos.push_back(pos[i]);
			//			cropped_rot.push_back(rot[i]);
			//			cropped_scale.push_back(scale[i]);
			//			cropped_shs.push_back(shs[i]);
			//			cropped_opacity.push_back(opacity[i]);
			//		}
			//	}
			//}

			//glm::vec3 pivot(1.0f, -0.3f, 4.5f);
			//float radius = 2.5f;
			//buildfunctionEdgeGraph2<3>(
			//	cropped_pos,
			//	cropped_rot,
			//	cropped_scale,
			//	cropped_edges,
			//	cropped_shs,
			//	8,      // default_k
			//	d_thresholdP,   // max_distance_threshold  
			//	true   // enable_debug
			//	);





			//saveFilteredGaussianEdgesToFile(cropped_pos, cropped_edges, cropped_opacity, filename, 20.0f, 0.35f, _pivot, _radius);

		}
	}
	
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

