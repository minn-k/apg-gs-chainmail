#include "APGGraphBuilder.hpp"
#include "GaussianView.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>

#include <Eigen/Dense>

// Keep the existing glm include style to avoid include path changes.
#include <glm/gtc/quaternion.hpp>
#include <glm/glm.hpp>

namespace {

inline float getAspectRatio(const sibr::Scale& scale) {
	const float maxScale = std::max({ scale.scale[0], scale.scale[1], scale.scale[2] });
	const float minScale = std::min({ scale.scale[0], scale.scale[1], scale.scale[2] });
	return maxScale / std::max(minScale, 1e-6f);
}

float orientationSim(const sibr::Rot& first, const sibr::Rot& second) {
	const Eigen::Quaternionf q1(first.rot[0], first.rot[1], first.rot[2], first.rot[3]);
	const Eigen::Quaternionf q2(second.rot[0], second.rot[1], second.rot[2], second.rot[3]);
	return std::abs(q1.dot(q2));
}

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


inline float computeSimilarityDistance(
	float dist, float oSim, float AR_diff, float shSim,
	float sigma, float sigma_sh, float sigma_ori,
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
	if (N < 2) {
		edges.clear();
		if (enable_debug) std::cout << "[APG] Need at least two Gaussians to build a graph.\n";
		return;
	}
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
	float sigma_SH = 1.0f;
	float sigma_ORI = 1.0f;

	const size_t queryCount = std::min(static_cast<size_t>(N),
		static_cast<size_t>(std::max(2, cfg.candidate_neighbors + 1)));
	const float edgePercentile = std::clamp(cfg.edge_percentile, 0.0f, 100.0f);

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
		std::vector<uint32_t> neigh(queryCount);
		std::vector<float> d2(queryCount);
		size_t found = kdtree.knnSearch(q, queryCount, neigh.data(), d2.data());
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

	if (count == 0) {
		edges.clear();
		if (enable_debug) std::cout << "[APG] No neighbor pairs found.\n";
		return;
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

	std::vector<float> dist_norm_values;
	for (float d : dist_values)
		dist_norm_values.push_back(minmax_norm(d, dist_min, dist_max));
	sigma = std::max(0.1f, get_iqr(dist_norm_values) / 1.45f);
	sigma_SH = std::max(0.1f, get_iqr(sh_diffs) / 1.45f);
	sigma_ORI = std::max(0.1f, get_iqr(ori_diffs) / 1.45f);

	for (int i = 0; i < N; ++i) {
		float AR_i = getAspectRatio(scales[i]);
		float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
		std::vector<uint32_t> neigh(queryCount);
		std::vector<float> d2(queryCount);
		size_t found = kdtree.knnSearch(q, queryCount, neigh.data(), d2.data());
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
				sigma, sigma_SH, sigma_ORI,
				w_dist, w_ori, w_shape, w_sh);

			d_samples.push_back(d);
		}
	}

	std::sort(d_samples.begin(), d_samples.end());
	size_t idx_10pct = static_cast<size_t>(d_samples.size() * edgePercentile * 0.01f);
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
		std::cout << "sigma_ORI = " << sigma_ORI << std::endl;
		std::cout << "sigma_SH = " << sigma_SH << std::endl;
		std::cout << "[INFO] d samples: " << d_samples.size() << std::endl;
		std::cout << "[INFO] score threshold (" << edgePercentile << "%): " << d_threshold << std::endl;
	}

	std::set<std::pair<int, int>> edgeSet;
	std::mutex edge_mtx;
	std::map<std::pair<int, int>, float> edgeScoreMap;

#pragma omp parallel
	{
		std::vector<uint32_t> neigh(queryCount);
		std::vector<float> d2(queryCount);
		std::set<std::pair<int, int>> localSet;
		std::map<std::pair<int, int>, float> localMap;

#pragma omp for schedule(dynamic, 100)
		for (int i = 0; i < N; ++i) {
			float AR_i = getAspectRatio(scales[i]);
			float q[3] = { pos[i].x(), pos[i].y(), pos[i].z() };
			size_t found = kdtree.knnSearch(q, queryCount, neigh.data(), d2.data());

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
					sigma, sigma_SH, sigma_ORI,
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
		std::cout << "Edge density: " << static_cast<float>(edges.size()) / static_cast<float>(N * (N - 1) / 2) << std::endl;
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
