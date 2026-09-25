#include "APGGraphBuilder.hpp"
#include "GaussianView.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// Keep the existing glm include style to avoid include path changes.
#include <glm/gtc/quaternion.hpp>
#include <glm/glm.hpp>

namespace {

float computeDistanceWeight(float dist, float max_dist) {
	return std::exp(-2.0f * dist / (max_dist + 1e-6f));
}

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

class OptimizedProgressReporter {
public:
	explicit OptimizedProgressReporter(size_t total_work) : total(total_work) {
		start_time = std::chrono::high_resolution_clock::now();
	}

	void increment(size_t count = 1) { processed += count; }

	void printProgress() {
		std::lock_guard<std::mutex> lock(print_mutex);

		auto current_time = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);

		size_t current_processed = processed.load();
		int percentage = static_cast<int>((current_processed * 100) / total);

		double eta_seconds = 0.0;
		if (current_processed > 0) {
			eta_seconds = (elapsed.count() * (total - current_processed)) / static_cast<double>(current_processed);
		}

		std::cout << "\r[";
		int filled_width = (percentage * 50) / 100;
		for (int i = 0; i < 50; ++i) {
			std::cout << (i < filled_width ? "=" : " ");
		}
		std::cout << "] " << percentage << "% (" << current_processed << "/" << total
			<< ") - ETA: " << static_cast<int>(eta_seconds) << "s";
		std::cout.flush();

		if (current_processed >= total) {
			std::cout << std::endl;
		}
	}

private:
	std::atomic<size_t> processed{ 0 };
	size_t total;
	std::chrono::high_resolution_clock::time_point start_time;
	std::mutex print_mutex;
};

inline float fastOrientationSimilarity(const sibr::Rot& rot1, const sibr::Rot& rot2) {
	float dot = rot1.rot[0] * rot2.rot[0] + rot1.rot[1] * rot2.rot[1] +
		rot1.rot[2] * rot2.rot[2] + rot1.rot[3] * rot2.rot[3];
	return std::abs(dot);
}

inline float approximateLocalDensity(const std::vector<sibr::Pos>& pos, int index,
	const std::vector<size_t>& neighbor_indices) {
	return static_cast<float>(neighbor_indices.size()) / (4.0f / 3.0f * M_PI * 0.1f * 0.1f * 0.1f);
}

inline float getAspectRatio(const sibr::Scale& scale) {
	float max_scale = std::max({ scale.scale[0], scale.scale[1], scale.scale[2] });
	float min_scale = std::min({ scale.scale[0], scale.scale[1], scale.scale[2] });
	return max_scale / std::max(min_scale, 1e-6f);
}

inline float fastDistanceSquared(const sibr::Pos& p1, const sibr::Pos& p2) {
	float dx = p1.x() - p2.x();
	float dy = p1.y() - p2.y();
	float dz = p1.z() - p2.z();
	return dx * dx + dy * dy + dz * dz;
}

struct DisjointSet {
	std::vector<int> parent, rank;
	explicit DisjointSet(int n) : parent(n), rank(n, 0) {
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

float euclidDist(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	return (a - b).norm();
}

float orientationSim(const sibr::Rot& ri, const sibr::Rot& rj) {
	Eigen::Quaternionf qi(ri.rot[0], ri.rot[1], ri.rot[2], ri.rot[3]),
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
	explicit ProgressReporter(size_t t) : count(0), total(t) {}
	void increment(size_t n = 1) { count += n; }
	void print(const std::string& prefix) {
		size_t c = count.load();
		int pct = static_cast<int>(100.0 * c / total);
		std::cout << "\r" << prefix
			<< " " << c << " / " << total
			<< " (" << pct << "%)" << std::flush;
	}
};

template <typename Derived>
struct KDTreeAdaptor {
	const Derived& obj;
	explicit KDTreeAdaptor(const Derived& obj_) : obj(obj_) {}

	inline size_t kdtree_get_point_count() const { return obj.rows(); }
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
		return obj.coeff(static_cast<int>(idx), static_cast<int>(dim));
	}
	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const { return false; }
};

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

float computeLocalDensity(int vertex_idx,
	const std::vector<sibr::Pos>& pos,
	const nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<float, KDTreeAdaptor<Eigen::MatrixXf>>,
	KDTreeAdaptor<Eigen::MatrixXf>, 3>& kdtree,
	float radius = 0.1f,
	int min_pts = 5) {
	float q[3] = { pos[vertex_idx].x(), pos[vertex_idx].y(), pos[vertex_idx].z() };
	std::vector<uint32_t> neighbors(min_pts + 10);
	std::vector<float> distances(min_pts + 10);

	size_t found = kdtree.knnSearch(q, min_pts + 5, neighbors.data(), distances.data());
	if (found < static_cast<size_t>(min_pts)) return 0.0f;

	float k_distance = std::sqrt(distances[min_pts - 1]);
	float effective_radius = std::max(radius, k_distance);

	float volume = (4.0f / 3.0f) * M_PI * std::pow(effective_radius, 3.0f);
	return static_cast<float>(found) / volume;
}

float adaptiveThreshold(int i, int j,
	const std::vector<float>& densities,
	float base_threshold,
	float density_factor = 0.3f) {
	float avg_density = (densities[i] + densities[j]) * 0.5f;
	float density_weight = 1.0f + density_factor * std::exp(-avg_density * 0.01f);
	return base_threshold * density_weight;
}

float computeMultiViewConsistency(int i, int j,
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot) {
	float distance = (pos[i] - pos[j]).norm();
	float orientation_consistency = orientationSim(rot[i], rot[j]);
	float depth_consistency = std::exp(-distance * distance / 0.01f);

	return 0.6f * orientation_consistency + 0.4f * depth_consistency;
}

std::vector<RegionInfo> segmentByRegionGrowing(
	const std::vector<sibr::Pos>& pos,
	const std::vector<float>& densities,
	const nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<float, KDTreeAdaptor<Eigen::MatrixXf>>,
	KDTreeAdaptor<Eigen::MatrixXf>, 3>& kdtree,
	float similarity_threshold = 0.05f,
	int min_region_size = 100) {

	const int N = static_cast<int>(pos.size());
	std::vector<bool> visited(N, false);
	std::vector<RegionInfo> regions;

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

			float q[3] = { pos[current].x(), pos[current].y(), pos[current].z() };
			std::vector<uint32_t> neighbors(20);
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

		if (region.vertices.size() >= static_cast<size_t>(min_region_size)) {
			region.avg_density /= static_cast<float>(region.vertices.size());
			region.centroid = centroid_sum / static_cast<float>(region.vertices.size());
			region.is_primary_region = region.avg_density > 5.0f;
			regions.push_back(region);
		}
	}

	return regions;
}

GraphQualityMetrics evaluateGraphQuality(const std::set<std::pair<int, int>>& edgeSet,
	size_t num_vertices) {
	GraphQualityMetrics metrics;

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
		metrics.avg_component_size = std::accumulate(component_sizes.begin(), component_sizes.end(), 0.0f) /
			static_cast<float>(component_sizes.size());
	}

	metrics.connectivity_score = metrics.largest_component_ratio *
		(1.0f - std::log(metrics.num_components + 1) / 10.0f);

	return metrics;
}

template<int D>
float shAlignmentSim(const sibr::SHs<D>& sh1, const sibr::SHs<D>& sh2) {
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
float shAlignmentSimLuminanceWeighted(const sibr::SHs<D>& sh1, const sibr::SHs<D>& sh2) {
	constexpr int SH_PER_CHANNEL = (D + 1) * (D + 1);
	const float w[3] = { 0.3f, 0.59f, 0.11f };

	float sim = 0.0f;

	for (int ch = 0; ch < 3; ++ch) {
		float dot = 0.0f;
		float norm1_sq = 0.0f;
		float norm2_sq = 0.0f;

		int offset = ch * SH_PER_CHANNEL;

		for (int i = 0; i < SH_PER_CHANNEL; ++i) {
			float a = sh1.shs[offset + i];
			float b = sh2.shs[offset + i];
			dot += a * b;
			norm1_sq += a * a;
			norm2_sq += b * b;
		}

		float channel_sim = 0.0f;
		if (norm1_sq > 1e-6f && norm2_sq > 1e-6f) {
			float raw_cos = dot / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
			channel_sim = 0.5f * (raw_cos + 1.0f);
		}

		sim += w[ch] * channel_sim;
	}

	return sim;
}

float normalize(float x, float min_val, float max_val) {
	if (max_val - min_val < 1e-6f) return 0.0f;
	return (x - min_val) / (max_val - min_val);
}

inline float computeSimilarityDistance(
	float dist, float oSim, float AR_diff, float shSim,
	float sigma, float sigma_AR, float sigma_sh, float sigma_ori,
	float w_dist, float w_ori, float w_shape, float w_sh) {
	sigma *= 5;
	return
		w_dist * (1.0f - std::exp(-dist * dist / (sigma * sigma))) +
		w_ori * (1.0f - std::exp(-(1.0f - oSim) * (1.0f - oSim) / (sigma_ori * sigma_ori))) +
		w_shape * AR_diff * AR_diff +
		w_sh * (1.0f - std::exp(-((1 - shSim) * (1 - shSim)) / (sigma_sh * sigma_sh)));

		//w_dist * dist +
		//w_ori * (1.0f - oSim) +
		//w_shape * AR_diff+
		//w_sh * (1 - shSim);
}

inline float mapScoreDistanceToStiffness(float score_d, float d_threshold) {
	// score_d is a distance-like score (smaller = more similar).
	// Map affinity smoothly to the ChainMail edge stiffness range.
	const float safeThreshold = std::max(d_threshold, 1e-6f);
	const float x = std::max(0.0f, score_d / safeThreshold);
	const float similarity = std::exp(-(x * x)); // in (0, 1], high for small score_d
	const float stiffMin = 0.05f;
	const float stiffMax = 1.00f;
	const float stiff = stiffMin + (stiffMax - stiffMin) * similarity;
	return std::clamp(stiff, stiffMin, stiffMax);
}

inline uint64_t makeKey(int a, int b) {
	return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

template<int D>
void buildfunctionEdgeGraph2Impl(
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot,
	const std::vector<sibr::Scale>& scales,
	std::vector<FORWARD::Edge>& edges,
	std::vector<sibr::SHs<D>>& shs,
	const sibr::APGGraphConfig& cfg,
	bool enable_debug = true) {
	const int N = static_cast<int>(pos.size());
	float d_threshold = 0.0f;
	auto start_time = std::chrono::high_resolution_clock::now();
	if (enable_debug) {
		std::cout << "\n=== ChainMail graph build ===" << std::endl;
		std::cout << "Input vertices: " << N << std::endl;
		cfg.print();
	}

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

	float sigma = 1.0f;
	float sigma_AR = 1.0f;
	float sigma_SH = 1.0f;
	float sigma_ORI = 1.0f;

	const int max_k = cfg.default_k + 1;
	const float d_thresholdP = cfg.d_thresholdP;

	float w_dist = cfg.w_dist;
	float w_ori = cfg.w_ori;
	float w_shape = cfg.w_shape;
	float w_sh = cfg.w_sh;

	std::vector<float> dist_values, ori_diffs, sh_diffs, AR_diffs, d_samples;

	float dist_sum = 0.0f, dist_min = FLT_MAX, dist_max = -FLT_MAX;
	float oSim_sum = 0.0f, oSim_min = FLT_MAX, oSim_max = -FLT_MAX;
	float shSim_sum = 0.0f, shSim_min = FLT_MAX, shSim_max = -FLT_MAX;
	float AR_diff_sum = 0.0f, AR_diff_min = FLT_MAX, AR_diff_max = -FLT_MAX;

	int count = 0;

	for (int i = 0; i < N; ++i) {
		float AR_i = getAspectRatio(scales[i]);
		float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
		std::vector<uint32_t> neigh(max_k + 1);
		std::vector<float> d2(max_k + 1);
		size_t found = kdtree.knnSearch(q, max_k + 1, neigh.data(), d2.data());
		for (size_t k = 1; k < found; ++k) {
			int j = static_cast<int>(neigh[k]);
			float dist = std::sqrt(d2[k]);
			float AR_j = getAspectRatio(scales[j]);
			float shSim = shAlignmentSimLuminanceWeighted<D>(shs[i], shs[j]);
			float oSim = orientationSim(rot[i], rot[j]);

			float AR_diff = std::abs(std::log((AR_i + 1e-6f) / (AR_j + 1e-6f)));

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

	float dist_avg = dist_sum / count;
	float oSim_avg = oSim_sum / count;
	float shSim_avg = shSim_sum / count;
	float AR_diff_avg = AR_diff_sum / count;

	std::cout << "Distance: min = " << dist_min << ", max = " << dist_max << ", avg = " << dist_avg << "\n";
	std::cout << "OrientationSim: min = " << oSim_min << ", max = " << oSim_max << ", avg = " << oSim_avg << "\n";
	std::cout << "SHSim: min = " << shSim_min << ", max = " << shSim_max << ", avg = " << shSim_avg << "\n";
	std::cout << "AR_diff: min = " << AR_diff_min << ", max = " << AR_diff_max << ", avg = " << AR_diff_avg << "\n";

	auto get_iqr = [](std::vector<float>& v) {
		std::sort(v.begin(), v.end());
		return v[v.size() * 3 / 4] - v[v.size() / 4];
	};
	auto minmax_norm = [](float x, float x_min, float x_max) -> float {
		if (x_max - x_min < 1e-6f) return 0.0f;
		return (x - x_min) / (x_max - x_min);
	};

	std::vector<float> dist_norm_values, AR_diff_norm_values;
	for (float d : dist_values)
		dist_norm_values.push_back(minmax_norm(d, dist_min, dist_max));
	for (float ad : AR_diffs)
		AR_diff_norm_values.push_back(minmax_norm(ad, AR_diff_min, AR_diff_max));
	sigma = std::max(0.1f, get_iqr(dist_norm_values) / 1.45f);
	sigma_AR = std::max(0.1f, get_iqr(AR_diff_norm_values) / 1.45f);
	sigma_SH = std::max(0.1f, get_iqr(sh_diffs) / 1.45f);
	sigma_ORI = std::max(0.1f, get_iqr(ori_diffs) / 1.45f);

	for (int i = 0; i < N; ++i) {
		float AR_i = getAspectRatio(scales[i]);
		float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
		std::vector<uint32_t> neigh(max_k + 1);
		std::vector<float> d2(max_k + 1);
		size_t found = kdtree.knnSearch(q, max_k + 1, neigh.data(), d2.data());
		for (size_t k = 1; k < found; ++k) {
			int j = static_cast<int>(neigh[k]);
			float dist = std::sqrt(d2[k]);
			float oSim = orientationSim(rot[i], rot[j]);
			float AR_j = getAspectRatio(scales[j]);
			float shSim = shAlignmentSimLuminanceWeighted<D>(shs[i], shs[j]);
			float AR_diff = std::abs(std::log((AR_i + 1e-6f) / (AR_j + 1e-6f)));
			float dist_norm = minmax_norm(dist, dist_min, dist_max);
			float AR_diff_norm = minmax_norm(AR_diff, AR_diff_min, AR_diff_max);

			float d = computeSimilarityDistance(
				dist_norm, oSim, AR_diff_norm, shSim,
				sigma, sigma_AR, sigma_SH, sigma_ORI,
				w_dist, w_ori, w_shape, w_sh);

			d_samples.push_back(d);
		}
	}

	std::sort(d_samples.begin(), d_samples.end());
	size_t idx_10pct = static_cast<size_t>(d_samples.size() * d_thresholdP * 0.01f);
	// 수정 후 (안전한 인덱싱)
	if (d_samples.empty()) {
		d_threshold = 0.0f;
	}
	else {
		size_t safe_idx = std::min(idx_10pct, d_samples.size() - 1);
		d_threshold = d_samples[safe_idx];
	}
	if (enable_debug) {
		std::cout << "sigma = " << sigma << std::endl;
		std::cout << "sigma_AR = " << sigma_AR << std::endl;
		std::cout << "sigma_ORI = " << sigma_ORI << std::endl;
		std::cout << "sigma_SH = " << sigma_SH << std::endl;
		std::cout << "[INFO] d samples: " << d_samples.size() << std::endl;
		std::cout << "[INFO] d_threshold (" << d_thresholdP << "%): " << d_threshold << std::endl;
	}

	std::set<std::pair<int, int>> edgeSet;
	std::mutex edge_mtx;
	std::map<std::pair<int, int>, float> edgeScoreMap;

#pragma omp parallel
	{
		std::vector<uint32_t> neigh(max_k + 1);
		std::vector<float> d2(max_k + 1);
		std::set<std::pair<int, int>> localSet;
		std::map<std::pair<int, int>, float> localMap;

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
				float shSim = shAlignmentSimLuminanceWeighted<D>(shs[i], shs[j]);
				float AR_diff = std::abs(std::log((AR_i + 1e-6f) / (AR_j + 1e-6f)));
				float dist_norm = minmax_norm(dist, dist_min, dist_max);
				float AR_diff_norm = minmax_norm(AR_diff, AR_diff_min, AR_diff_max);

				float d = computeSimilarityDistance(
					dist_norm, oSim, AR_diff_norm, shSim,
					sigma, sigma_AR, sigma_SH, sigma_ORI,
					w_dist, w_ori, w_shape, w_sh);

				if (d <= d_threshold) {
					int a = i, b = j;
					if (a > b) std::swap(a, b);
					auto key = std::make_pair(a, b);
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
		std::cout << "[Step 1] Initial edges: " << edgeSet.size() << "\n";
		std::cout << "[Step 1] Initial edges (score map): " << edgeScoreMap.size() << "\n";
	}

	std::set<std::pair<int, int>> final_edges = edgeSet;
	GraphQualityMetrics quality = evaluateGraphQuality(final_edges, N);
	if (enable_debug) {
		std::cout << "[Step 2] Largest component ratio: "
			<< quality.largest_component_ratio << std::endl;
	}

	edges.clear();
	edges.reserve(edgeScoreMap.size());
	float stiff_min = FLT_MAX;
	float stiff_max = -FLT_MAX;
	double stiff_sum = 0.0;

	for (auto& kv : edgeScoreMap)
	{
		int a = kv.first.first;
		int b = kv.first.second;
		float score_d = kv.second;

		float restLen = (pts.row(a) - pts.row(b)).norm();
		float stiff = mapScoreDistanceToStiffness(score_d, d_threshold);
		stiff_min = std::min(stiff_min, stiff);
		stiff_max = std::max(stiff_max, stiff);
		stiff_sum += stiff;

		edges.emplace_back(a, b, restLen, stiff);
	}

	quality = evaluateGraphQuality(final_edges, N);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	if (enable_debug) {
		std::cout << "\n=== Result ===" << std::endl;
		std::cout << "Final edges: " << edges.size() << std::endl;
		std::cout << "Edge density: " << static_cast<float>(edges.size()) / (N * (N - 1) / 2) << std::endl;
		std::cout << "Largest component ratio: " << quality.largest_component_ratio << std::endl;
		std::cout << "Total time: " << duration.count() << "ms" << std::endl;
		if (!edges.empty()) {
			const double stiff_avg = stiff_sum / static_cast<double>(edges.size());
			std::cout << "Stiffness range: [" << stiff_min << ", " << stiff_max
				<< "], avg=" << stiff_avg << std::endl;
		}

		float edge_density = static_cast<float>(edges.size()) / N;
		if (edge_density > 20.0f) {
			std::cout << "[WARN] Edge density high (" << edge_density << " per vertex)" << std::endl;
		}
		else {
			std::cout << "[OK] Edge density normal (" << edge_density << " per vertex)" << std::endl;
		}
	}
}

} // namespace

namespace sibr {

void APGGraphBuilder::build(
	const std::vector<sibr::Pos>& pos,
	const std::vector<sibr::Rot>& rot,
	const std::vector<sibr::Scale>& scales,
	std::vector<FORWARD::Edge>& edges,
	std::vector<sibr::SHs<3>>& shs,
	const sibr::APGGraphConfig& cfg,
	bool enable_debug) {
	buildfunctionEdgeGraph2Impl<3>(pos, rot, scales, edges, shs, cfg, enable_debug);
}

} // namespace sibr
