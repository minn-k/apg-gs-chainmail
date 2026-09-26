#pragma once

#include <vector>
#include "GaussianView.hpp"

namespace sibr {

struct APGGraphConfig;

class APGGraphBuilder {
public:
	APGGraphBuilder() = default;

	void build(
		const std::vector<sibr::Pos>& pos,
		const std::vector<sibr::Rot>& rot,
		const std::vector<sibr::Scale>& scales,
		std::vector<FORWARD::Edge>& edges,
		std::vector<sibr::SHs<3>>& shs,
		const sibr::APGGraphConfig& cfg,
		bool enable_debug = true);
};

} // namespace sibr
