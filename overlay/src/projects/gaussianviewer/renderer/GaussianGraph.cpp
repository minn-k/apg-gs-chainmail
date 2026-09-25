#include "GaussianGraph.hpp"

void GaussianGraphBuilder::buildGraph(
    const std::vector<sibr::Pos>& pos,
    const std::vector<sibr::Scale>& scale,
    const std::vector<float>& opacity,
    float maxConnectionRadius,
    int maxNeighbors,
    bool useAdaptiveStiffness
) {
    // 초기화
    vertices = pos;
    edges.clear();
    vertexStiffness.resize(pos.size(), 1.0f);
    vertexOpacity = opacity;

    std::cout << "가우시안 그래프 생성 시작 - 점 개수: " << pos.size() << std::endl;

    // 1. KD-Tree 구축
    std::vector<int> indices(pos.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto kdTree = buildKDTree(pos, indices);

    // 2. 각 점에 대해 최근접 이웃 찾기
    for (int i = 0; i < pos.size(); ++i) {
        std::priority_queue<std::pair<float, int>> neighbors;
        kNearestNeighbors(kdTree, pos[i], maxNeighbors + 1, neighbors);

        // 자기 자신 제외한 이웃들과 연결
        std::vector<std::pair<float, int>> validNeighbors;
        while (!neighbors.empty()) {
            auto [dist, idx] = neighbors.top();
            neighbors.pop();

            if (idx != i && dist <= maxConnectionRadius) {
                validNeighbors.push_back({ dist, idx });
            }
        }

        // 거리 순으로 정렬
        std::sort(validNeighbors.begin(), validNeighbors.end());

        // 최대 maxNeighbors 개의 이웃과 연결
        for (int j = 0; j < std::min(maxNeighbors, (int)validNeighbors.size()); ++j) {
            float dist = validNeighbors[j].first;
            int neighborIdx = validNeighbors[j].second;

            // 중복 간선 방지
            if (i < neighborIdx) {
                float stiffness = 1.0f;
                if (useAdaptiveStiffness) {
                    stiffness = calculateAdaptiveStiffness(
                        pos[i], pos[neighborIdx],
                        opacity[i], opacity[neighborIdx],
                        scale[i], scale[neighborIdx]
                    );
                }

                edges.emplace_back(i, neighborIdx, dist, stiffness);
            }
        }
    }

    std::cout << "기본 연결 완료 - 간선 개수: " << edges.size() << std::endl;

    // 3. 연결성 보장
    ensureConnectivity();

    std::cout << "연결성 보장 완료 - 최종 간선 개수: " << edges.size() << std::endl;
}

void GaussianGraphBuilder::buildChainmailGraph(
    const std::vector<sibr::Pos>& pos,
    const std::vector<sibr::Scale>& scale,
    float baseRadius,
    float rigidStiffness,
    float flexibleStiffness
) {
    vertices = pos;
    edges.clear();
    vertexStiffness.resize(pos.size());

    std::cout << "체인메일 그래프 생성 시작" << std::endl;

    // 다중 스케일 연결 생성
    std::vector<float> radiusLevels = { baseRadius, baseRadius * 1.5f, baseRadius * 2.0f };
    std::vector<float> stiffnessLevels = { rigidStiffness, rigidStiffness * 0.7f, flexibleStiffness };

    for (int level = 0; level < radiusLevels.size(); ++level) {
        float radius = radiusLevels[level];
        float stiffness = stiffnessLevels[level];

        for (int i = 0; i < pos.size(); ++i) {
            for (int j = i + 1; j < pos.size(); ++j) {
                float dist = distance(pos[i], pos[j]);

                if (dist <= radius) {
                    // 스케일 기반 강성 조정
                    float scaleAvg = (scale[i].scale[0] + scale[i].scale[1] + scale[i].scale[2] +
                        scale[j].scale[0] + scale[j].scale[1] + scale[j].scale[2]) / 6.0f;
                    float adjustedStiffness = stiffness * (1.0f + scaleAvg * 0.5f);

                    // 중복 간선 체크
                    bool exists = false;
                    for (const auto& edge : edges) {
                        if ((edge.m_vert[0] == i && edge.m_vert[1] == j) ||
                            (edge.m_vert[0] == j && edge.m_vert[1] == i)) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        edges.emplace_back(i, j, dist, adjustedStiffness);
                    }
                }
            }
        }
    }

    // 연결성 보장
    ensureConnectivity();

    std::cout << "체인메일 그래프 생성 완료 - 간선 개수: " << edges.size() << std::endl;
}

std::shared_ptr<KDNode> GaussianGraphBuilder::buildKDTree(
    const std::vector<sibr::Pos>& points,
    const std::vector<int>& indices,
    int depth
) {
    if (indices.empty()) return nullptr;

    if (indices.size() == 1) {
        return std::make_shared<KDNode>(points[indices[0]], indices[0], depth);
    }

    int axis = depth % 3;
    std::vector<int> sortedIndices = indices;

    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&](int a, int b) {
            switch (axis) {
            case 0: return points[a].x() < points[b].x();
            case 1: return points[a].y() < points[b].y();
            case 2: return points[a].z() < points[b].z();
            default: return false;
            }
        });

    int medianIdx = sortedIndices.size() / 2;
    auto node = std::make_shared<KDNode>(
        points[sortedIndices[medianIdx]],
        sortedIndices[medianIdx],
        depth
        );

    std::vector<int> leftIndices(sortedIndices.begin(), sortedIndices.begin() + medianIdx);
    std::vector<int> rightIndices(sortedIndices.begin() + medianIdx + 1, sortedIndices.end());

    node->left = buildKDTree(points, leftIndices, depth + 1);
    node->right = buildKDTree(points, rightIndices, depth + 1);

    return node;
}

void GaussianGraphBuilder::kNearestNeighbors(
    const std::shared_ptr<KDNode>& node,
    const sibr::Pos& target,
    int k,
    std::priority_queue<std::pair<float, int>>& heap
) const {
    if (!node) return;

    float dist = distance(node->point, target);

    if (heap.size() < k) {
        heap.push({ dist, node->index });
    }
    else if (dist < heap.top().first) {
        heap.pop();
        heap.push({ dist, node->index });
    }

    int axis = node->depth % 3;
    float axisDistance = 0.0f;

    switch (axis) {
    case 0: axisDistance = target.x() - node->point.x(); break;
    case 1: axisDistance = target.y() - node->point.y(); break;
    case 2: axisDistance = target.z() - node->point.z(); break;
    }

    auto firstChild = (axisDistance < 0) ? node->left : node->right;
    auto secondChild = (axisDistance < 0) ? node->right : node->left;

    kNearestNeighbors(firstChild, target, k, heap);

    if (heap.size() < k || std::abs(axisDistance) < heap.top().first) {
        kNearestNeighbors(secondChild, target, k, heap);
    }
}

void GaussianGraphBuilder::connectComponentsFast(
    const std::unordered_map<int, std::vector<int>>& componentMap
) {
    std::vector<int> roots;
    roots.reserve(componentMap.size());
    for (auto& kv : componentMap) roots.push_back(kv.first);

    for (int i = 0; i + 1 < (int)roots.size(); ++i) {
        auto& c1 = componentMap.at(roots[i]);
        auto& c2 = componentMap.at(roots[i + 1]);
        float minD = std::numeric_limits<float>::max();
        int b1 = -1, b2 = -1;
        for (int v1 : c1) for (int v2 : c2) {
            float dx = vertices[v1].x() - vertices[v2].x();
            float dy = vertices[v1].y() - vertices[v2].y();
            float dz = vertices[v1].z() - vertices[v2].z();
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < minD) { minD = d2; b1 = v1; b2 = v2; }
        }
        float restLen = std::sqrt(minD);
        edges.emplace_back(b1, b2, restLen, 0.5f);
    }
}

void GaussianGraphBuilder::ensureConnectivity() {
    if (vertices.empty()) return;

    std::cout << "고속 연결성 보장 시작..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 최적화된 Union-Find 사용
    OptimizedUnionFind uf(vertices.size());

    // 기존 간선으로 Union 수행
    for (const auto& edge : edges) {
        uf.unite(edge.m_vert[0], edge.m_vert[1]);
    }

    // 연결 컴포넌트 찾기 (빠른 방법)
    std::unordered_map<int, std::vector<int>> componentMap;
    for (int i = 0; i < vertices.size(); ++i) {
        int root = uf.find(i);
        componentMap[root].push_back(i);
    }

    std::cout << "연결 컴포넌트 수: " << componentMap.size() << std::endl;

    // 컴포넌트가 2개 이상이면 연결
    if (componentMap.size() <= 1) return;
    connectComponentsFast(componentMap);
    

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "연결성 보장 완료: " << duration.count() << "ms" << std::endl;
}

float GaussianGraphBuilder::distance(const sibr::Pos& a, const sibr::Pos& b) const {
    float dx = a.x() - b.x();
    float dy = a.y() - b.y();
    float dz = a.z() - b.z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float GaussianGraphBuilder::calculateAdaptiveStiffness(
    const sibr::Pos& pos1, const sibr::Pos& pos2,
    float opacity1, float opacity2,
    const sibr::Scale& scale1, const sibr::Scale& scale2
) const {
    // 거리 기반 강성
    float dist = distance(pos1, pos2);
    float distanceFactor = std::exp(-dist * 10.0f);

    // 불투명도 기반 강성
    float opacityFactor = (opacity1 + opacity2) / 2.0f;

    // 스케일 기반 강성
    float avgScale1 = (scale1.scale[0] + scale1.scale[1] + scale1.scale[2]) / 3.0f;
    float avgScale2 = (scale2.scale[0] + scale2.scale[1] + scale2.scale[2]) / 3.0f;
    float scaleFactor = std::min(avgScale1, avgScale2);

    // 위치 기반 강성 (예: 얼굴 중심부는 강하게)
    float centerDist = std::sqrt(pos1.x() * pos1.x() + pos1.y() * pos1.y() + pos1.z()* pos1.z());
    float positionFactor = std::exp(-centerDist * 0.5f);

    return 0.5f + 1.5f * (distanceFactor * opacityFactor * scaleFactor * positionFactor);
}

int GaussianGraphBuilder::find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

void GaussianGraphBuilder::unite(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX != rootY) {
        parent[rootX] = rootY;
    }
}
bool GaussianGraphBuilder::saveToPLY(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "PLY 파일 생성 실패: " << filename << std::endl;
        return false;
    }

    std::cout << "PLY 파일 저장 시작: " << filename << std::endl;

    // PLY 헤더 작성
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "comment Generated by Gaussian Chainmail Graph Builder\n";
    file << "comment Contains vertices and edges for deformation simulation\n";

    // 정점 정보
    file << "element vertex " << vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float stiffness\n";
    if (!vertexOpacity.empty()) {
        file << "property float opacity\n";
    }
    file << "property int vertex_id\n";

    // 간선 정보
    file << "element edge " << edges.size() << "\n";
    file << "property int vertex1\n";
    file << "property int vertex2\n";
    file << "property float rest_length\n";
    file << "property float stiffness\n";
    file << "property float weight\n";

    // 그래프 통계 정보
    file << "element graph_info 1\n";
    file << "property int total_vertices\n";
    file << "property int total_edges\n";
    file << "property float avg_edge_length\n";
    file << "property float max_edge_length\n";
    file << "property float min_edge_length\n";

    file << "end_header\n";

    // 정점 데이터 작성
    for (int i = 0; i < vertices.size(); ++i) {
        file << vertices[i].x() << " " << vertices[i].y() << " " << vertices[i].z() << " ";
        file << (i < vertexStiffness.size() ? vertexStiffness[i] : 1.0f) << " ";
        if (!vertexOpacity.empty()) {
            file << vertexOpacity[i] << " ";
        }
        file << i << "\n";
    }

    // 간선 데이터 작성
    float totalLength = 0.0f;
    float minLength = std::numeric_limits<float>::max();
    float maxLength = 0.0f;

    for (const auto& edge : edges) {
        float length = edge.rl;
        totalLength += length;
        minLength = std::min(minLength, length);
        maxLength = std::max(maxLength, length);

        // 간선 가중치 계산 (강성과 길이의 조합)
        float weight = edge.st / (1.0f + length);

        file << edge.m_vert[0] << " " << edge.m_vert[1] << " ";
        file << edge.rl << " " << edge.st << " " << weight << "\n";
    }

    // 그래프 통계 정보 작성
    float avgLength = edges.empty() ? 0.0f : totalLength / edges.size();
    file << vertices.size() << " " << edges.size() << " ";
    file << avgLength << " " << maxLength << " " << minLength << "\n";

    file.close();

    std::cout << "PLY 파일 저장 완료!" << std::endl;
    std::cout << "- 정점 수: " << vertices.size() << std::endl;
    std::cout << "- 간선 수: " << edges.size() << std::endl;
    std::cout << "- 평균 간선 길이: " << avgLength << std::endl;

    return true;
}
void GaussianGraphBuilder::printStatistics() const {
    std::cout << "\n=== 그래프 통계 ===" << std::endl;
    std::cout << "정점 수: " << vertices.size() << std::endl;
    std::cout << "간선 수: " << edges.size() << std::endl;

    if (!edges.empty()) {
        float totalLength = 0.0f;
        float totalStiffness = 0.0f;
        float minLength = std::numeric_limits<float>::max();
        float maxLength = 0.0f;

        for (const auto& edge : edges) {
            totalLength += edge.rl;
            totalStiffness += edge.st;
            minLength = std::min(minLength, edge.rl);
            maxLength = std::max(maxLength, edge.rl);
        }

        std::cout << "평균 간선 길이: " << totalLength / edges.size() << std::endl;
        std::cout << "최소 간선 길이: " << minLength << std::endl;
        std::cout << "최대 간선 길이: " << maxLength << std::endl;
        std::cout << "평균 강성: " << totalStiffness / edges.size() << std::endl;
        std::cout << "평균 연결도: " << (2.0f * edges.size()) / vertices.size() << std::endl;
    }
}

bool GaussianGraphBuilder::verifyConnectivity() const {
    if (vertices.empty()) return true;
    int n = vertices.size();
    std::vector<bool> visited(n, false);
    std::vector<int> stack;
    stack.reserve(n);
    stack.push_back(0);
    visited[0] = true;
    int count = 0;
    // DFS를 사용한 연결성 검사
    std::vector<std::vector<int>> adjacencyList(vertices.size());

    // 인접 리스트 구성
    for (const auto& edge : edges) {
        adjacencyList[edge.m_vert[0]].push_back(edge.m_vert[1]);
        adjacencyList[edge.m_vert[1]].push_back(edge.m_vert[0]);
    }

    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        ++count;
        for (int u : adjacencyList[v]) {
            if (!visited[u]) {
                visited[u] = true;
                stack.push_back(u);
            }
        }
    }

    // 모든 정점이 방문되었는지 확인
    for (bool v : visited) {
        if (!v) return false;
    }

    return true;
}
