#pragma once

#include <vector>
#include <memory>
#include <queue>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <numeric>
#include <functional>
#include <limits>
#include <GaussianView.hpp>
// SIBR 타입 정의



// 간선 구조체
struct Edge {
    int m_vert[2];
    float st;  // stiffness
    float rl;  // rest length

    Edge(int v0, int v1, float restlen, float stiff = 1.0f)
        : st(stiff), rl(restlen) {
        m_vert[0] = v0;
        m_vert[1] = v1;
    }
};

// KD-Tree 노드 구조
struct KDNode {
    sibr::Pos point;
    int index;
    int depth;
    std::shared_ptr<KDNode> left;
    std::shared_ptr<KDNode> right;

    KDNode(const sibr::Pos& p, int idx, int d)
        : point(p), index(idx), depth(d), left(nullptr), right(nullptr) {}
};
class OptimizedUnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;

public:
    OptimizedUnionFind(int n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    // Path Compression으로 최적화된 Find
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // 경로 압축
        }
        return parent[x];
    }

    // Union by Rank로 최적화된 Union
    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) return false;

        // 작은 트리를 큰 트리에 붙임
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        }
        else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        }
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        return true;
    }
};

// 가우시안 그래프 생성 클래스
class GaussianGraphBuilder {
public:
    // 메인 그래프 생성 함수
    void buildGraph(
        const std::vector<sibr::Pos>& pos,
        const std::vector<sibr::Scale>& scale,
        const std::vector<float>& opacity,
        float maxConnectionRadius = 0.1f,
        int maxNeighbors = 6,
        bool useAdaptiveStiffness = true
    );

    // 특화된 체인메일 그래프 생성
    void buildChainmailGraph(
        const std::vector<sibr::Pos>& pos,
        const std::vector<sibr::Scale>& scale,
        float baseRadius = 0.08f,
        float rigidStiffness = 2.0f,
        float flexibleStiffness = 0.5f
    );
    void connectComponentsFast(const std::unordered_map<int, std::vector<int>>& componentMap
        );
    // PLY 파일 저장
    bool saveToPLY(const std::string& filename) const;

    // 그래프 정보 접근
    const std::vector<Edge>& getEdges() const { return edges; }
    const std::vector<sibr::Pos>& getVertices() const { return vertices; }

    // 통계 정보
    void printStatistics() const;
    bool verifyConnectivity() const;

private:
    std::vector<sibr::Pos> vertices;
    std::vector<Edge> edges;
    std::vector<float> vertexStiffness;
    std::vector<float> vertexOpacity;

    // KD-Tree 구축
    std::shared_ptr<KDNode> buildKDTree(
        const std::vector<sibr::Pos>& points,
        const std::vector<int>& indices,
        int depth = 0
    );

    // 최근접 이웃 검색
    void kNearestNeighbors(
        const std::shared_ptr<KDNode>& node,
        const sibr::Pos& target,
        int k,
        std::priority_queue<std::pair<float, int>>& heap
    ) const;

    // 연결성 보장을 위한 최소 신장 트리
    void ensureConnectivity();

    // 유틸리티 함수들
    float distance(const sibr::Pos& a, const sibr::Pos& b) const;
    float calculateAdaptiveStiffness(
        const sibr::Pos& pos1,
        const sibr::Pos& pos2,
        float opacity1,
        float opacity2,
        const sibr::Scale& scale1,
        const sibr::Scale& scale2
    ) const;

    // Union-Find 구조 (연결성 검사용)
    std::vector<int> parent;
    int find(int x);
    void unite(int x, int y);
};
