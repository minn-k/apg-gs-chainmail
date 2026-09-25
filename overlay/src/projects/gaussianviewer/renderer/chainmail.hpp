#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <cmath>

// ===== 자료구조 =====
constexpr float AIR = 0.13f;
constexpr float SKIN = 0.35f;
constexpr float BONE = 0.55f;
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
    float xShearY, xShearZ;       // X방향 위치에 대한 Y, Z축 전단
    float yShearX, yShearZ;       // Y방향 위치에 대한 X, Z축 전단
    float zShearX, zShearY;       // Z방향 위치에 대한 X, Y축 전단
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

struct Neighbor {
    int idx;    // neighbor 정점 인덱스
    float dist; // 거리
    Neighbor() : idx(-1), dist(0) {}
    Neighbor(int idx, float dist) : idx(idx), dist(dist) {}
};

struct Element {
    Vec3 pos;
    float density;
    float time;
    int offset;      // neighbor 배열 시작 인덱스
    int neighborCnt;
    // 추가 파라미터들 필요시 여기에
    Element()
        : pos(), density(0), time(1e9f), offset(0), neighborCnt(0) {}
};
struct Edge {
    int v1, v2;
    float dist;
};
// ===== 체인메일 메인 클래스 =====

class ChainMail {
public:
    ChainMail();
    ~ChainMail();
    const std::vector<Edge>& getEdges() const { return edges; }

    // 파일 로드/저장
    bool loadGraph(const std::string& filename);
    bool saveGraph(const std::string& filename);

    // propagation 알고리즘
    void resetTime();
    void movePointPos(int idx, const Vec3& dpos);
    void propagate( std::vector<int>& activeSet);
    // relaxation (필요시)
    void relax(const std::vector<int>& activeSet);

    // 기타
    size_t numElements() const { return elements.size(); }
    Element& getElement(int i) { return elements[i]; }
    const Element& getElement(int i) const { return elements[i]; }

private:
    // propagation 내부
    float propagationTime(const Element& e, const Element& neighbor);
    void shiftElementPoint(Element& elem, const Element& n, float targDist, bool& moved);

    // 데이터
    std::vector<Element> elements;
    std::vector<Neighbor> neighbors;
    std::vector<Edge> edges; // 추가

};
