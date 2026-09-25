#include "chainmail.h"

// 콘스트레인트 하드코딩(단순 버전)
// AIR
CMConstraint h_AIR(
    0.1f, 0.1f, 0.1f,   // dx, dy, dz
    0.1f, 0.1f,         // xShearY, xShearZ
    0.1f, 0.1f,         // yShearX, yShearZ
    0.1f, 0.1f          // zShearX, zShearY
);

// SKIN
CMConstraint h_SKIN(
    0.01f, 0.01f, 0.01f,
    0.01f, 0.01f,
    0.01f, 0.01f,
    0.01f, 0.01f
);

// BONE
CMConstraint h_BONE(
    0.0001f, 0.0001f, 0.0001f,
    0.0001f, 0.0001f,
    0.0001f, 0.0001f,
    0.0001f, 0.0001f
);

ChainMail::ChainMail() {}
ChainMail::~ChainMail() {}

bool ChainMail::loadGraph(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cout << "Failed to open: " << filename << "\n";
        return false;
    }

    int numVertices, numEdges;
    in >> numVertices >> numEdges;
    elements.resize(numVertices);

    std::vector<std::vector<Neighbor>> tempNeigh(numVertices);

    for (int i = 0; i < numVertices; ++i) {
        float x, y, z, density;
        in >> x >> y >> z >> density;
        elements[i].pos = Vec3(x, y, z);
        elements[i].density = density;
        elements[i].time = 1e9f; // 최초 무한대로
        elements[i].offset = 0;
        elements[i].neighborCnt = 0;
    }

    for (int i = 0; i < numEdges; ++i) {
        int v1, v2;
        float dist;
        in >> v1 >> v2 >> dist;
        tempNeigh[v1].emplace_back(v2, dist);
        tempNeigh[v2].emplace_back(v1, dist);
        edges.push_back({ v1, v2, dist });

    }

    neighbors.clear();
    int idx = 0;
    int offset = 0;
    for (int i = 0; i < numVertices; ++i) {
        int cnt = static_cast<int>(tempNeigh[i].size());
        elements[i].offset = offset;
        elements[i].neighborCnt = cnt;
        for (int j = 0; j < cnt; ++j) {
            neighbors.push_back(tempNeigh[i][j]);
            idx++;
        }
        offset = idx;
    }
    in.close();
    return true;
}

bool ChainMail::saveGraph(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) return false;
    out << elements.size() << " " << neighbors.size() / 2 << "\n";
    for (const auto& e : elements)
        out << e.pos.x << " " << e.pos.y << " " << e.pos.z << " " << e.density << "\n";
    // 간선 정보 저장(중복주의: 양방향으로 저장되어 있을 가능성)
    std::unordered_set<uint64_t> written;
    for (int i = 0; i < elements.size(); ++i) {
        const Element& e = elements[i];
        for (int j = 0; j < e.neighborCnt; ++j) {
            const Neighbor& n = neighbors[e.offset + j];
            int a = std::min(i, n.idx), b = std::max(i, n.idx);
            uint64_t key = (static_cast<uint64_t>(a) << 32) | (b);
            if (written.count(key)) continue;
            out << i << " " << n.idx << " " << n.dist << "\n";
            written.insert(key);
        }
    }
    return true;
}

void ChainMail::resetTime() {
    for (auto& e : elements)
        e.time = 1e9f;
}

void ChainMail::movePointPos(int idx, const Vec3& dpos) {
    elements[idx].pos = elements[idx].pos + dpos;
    elements[idx].time = 0.0f;
}


// 변화(전파) wave가 실제로 도달한 정점 인덱스만 activeSet에 기록 (relax에 활용)
void ChainMail::propagate(std::vector<int>& activeSet) {
    size_t N = elements.size();//전체정점 개수
    std::vector<int> active;// 활성화 된 정점들
    activeSet.clear();//결과 배열
    std::vector<bool> propagated(N, false);  // relax용: propagate로 전파된(방문된) 정점 여부
    std::vector<bool> movedFlag(N, false);     // 실제 위치 이동이 발생한 정점 기록

    // 초기 활성화: 타임스탬프 0인 정점
    for (size_t i = 0; i < N; ++i)
        if (elements[i].time == 0.0f)//초기에 활성화된(변형의 시작점찾기) 타임스탬스0인점
            active.push_back(static_cast<int>(i));//활성화 목록에 추가

    int iter = 0;
    // propagation 반복
    while (!active.empty()) {//활성화 목록에서 모두 처리가끝나면 전파끝
        std::vector<int> nextActive;//다음 iter 에서 활성화 할 정점들
        std::vector<bool> visited(N, false); // 같은 iter 에서 중복 생성 방지

        for (int idx : active)
            visited[idx] = true;

        // #pragma omp parallel for (병렬화시 활성화)
        for (size_t aidx = 0; aidx < active.size(); ++aidx) {
            int idx = active[aidx];
            Element& elem = elements[idx];
            int off = elem.offset, nCnt = elem.neighborCnt;

            for (int ni = 0; ni < nCnt; ++ni) {
                const Neighbor& neigh = neighbors[off + ni];
                int nIdx = neigh.idx;
                float dist = neigh.dist;
                Element& neighbor = elements[nIdx];

                // propagation time, 반드시 '내 time + link시간'!
                float newTime = elem.time + propagationTime(elem, neighbor);

                // 이웃이 현재보다 더 빠른 경로로 도달하면 update
                if (neighbor.time > newTime) {
                    neighbor.time = newTime;

                    bool moved = false;
                    shiftElementPoint(neighbor, elem, dist, moved); // neighbor 이동
                    propagated[nIdx] = true;   // t와 pos 모두 변화

                    // activeSet 등록(실제 변화 발생)
                    if (moved) {
                        movedFlag[nIdx] = true;
                    }                    // 활성화 리스트(중복 방지)
                    if (!visited[nIdx]) {
                        nextActive.push_back(nIdx);
                        visited[nIdx] = true;
                    }
                }
            }
        }
        active = std::move(nextActive);
        iter++;
        std::cout << "Iteration " << iter << ", Active Count: " << active.size() << std::endl;
    }

    // propagate wave가 실제로 도달해 변화가 일어난 모든 정점의 index를 activeSet에 기록
    for (size_t i = 0; i < N; ++i)
        if (propagated[i] && movedFlag[i])
            activeSet.push_back(static_cast<int>(i));
    std::cout << "propagate activeSet size (실제 움직임): " << activeSet.size() << std::endl;

}
float ChainMail::propagationTime(const Element& e, const Element& neighbor) {
    // density가 0~1 범위일 때 예시
    float et, nt;
    if (e.density < AIR)        et = 1.0f;        // soft (air)
    else if (e.density < SKIN)   et = 0.3f;        // mid (skin)
    else if (e.density < BONE)  et = 0.05f;       // stiff (bone 등)
    else et = 0.005;

    if (neighbor.density < AIR)      nt = 1.0f;
    else if (neighbor.density < SKIN) nt = 0.3f;
    else if (neighbor.density < BONE) nt = 0.05f;       // stiff (bone 등)
    else nt = 0.005;
    return (et + nt) * 0.5f;

    // 또는 아주 단순하게
    // return 1.0f;   // wave 속도 고정 (실제 변화는 제약값에서 발생)
}
CMConstraint getConstraint(float density) {
    // 0(skin/air) ~ 1(bone)
    float axisC;   // 축 방향 강성
    float shearC;  // 전단 방향 강성

    if (density < AIR) {           // Air
        axisC = 0.07f;
        shearC = 0.07f;
    }
    else if (density < SKIN) {     // Skin
        axisC = 0.05f;
        shearC = 0.05f;
    }
    else {                         // Bone
        axisC = 0.008f;
        shearC = 0.008f;
    }

    return CMConstraint(
        axisC, axisC, axisC,   // dx, dy, dz
        shearC, shearC,        // xShearY, xShearZ
        shearC, shearC,        // yShearX, yShearZ
        shearC, shearC         // zShearX, zShearY
    );
}

void ChainMail::shiftElementPoint(Element& elem, const Element& n, float targDist, bool& moved) {
    // 간단화: density별로 constraint 하드코딩 적용(실제값은 현업코드 참조)
    CMConstraint nConstraint = getConstraint(elem.density);

    Vec3 dir = n.pos - elem.pos;
    float len = dir.length();
    Vec3 nDir = (len > 1e-6f) ? dir.normalized() : Vec3();

    float delta = 0.0f;
    float alpha = 1.0f; // 0 < alpha <= 1.0, 낮을수록 부드럽게
    if (len < targDist - nConstraint.dx) {
        // 너무 가까움 → 멀어져야 함 → -nDir 방향
        delta = (targDist - nConstraint.dx) - len;
        elem.pos = elem.pos - nDir * (delta * alpha);
        moved = true;
    }
    else if (len > targDist + nConstraint.dx) {
        // 너무 멀음 → 가까워져야 함 → +nDir 방향
        delta = len - (targDist + nConstraint.dx);
        elem.pos = elem.pos + nDir * (delta * alpha);
        moved = true;
    }
    else {
        moved = false;
    }


    //if (len < targDist - nConstraint.dx) {//제약 범위보다 가깝거나
    //    delta = (targDist - nConstraint.dx) - len;//제약범위내에서 가능한 가깝게 설정
    //}
    //else if (len > targDist + nConstraint.dx) {//제약범위보다 멀면
    //    delta = (targDist + nConstraint.dx) - len;//거리 제약범위 내에서 최대거리로 설정
    //}

    //// 한 번에 다 보내지 않고 조금씩 따라오게!
    //if (std::abs(delta) > 1e-6f) {
    //    float alpha = 1.0f; // 0 < alpha <= 1.0, 낮을수록 부드럽게
    //    elem.pos = elem.pos + nDir * (delta * alpha);
    //    moved = true;
    //}
    //else {
    //    moved = false;
    //}
}

void ChainMail::relax(const std::vector<int>& activeSet) {
    std::vector<Vec3> newPos(elements.size());

    for (int idx : activeSet) {
        Element& e = elements[idx];

        // 자기 밀도에 따른 제약값 (getConstraint 대체)
        CMConstraint eConstraint;
        if (e.density < AIR) {          // AIR
            eConstraint.dx = 0.8f;  eConstraint.dy = 0.8f;  eConstraint.dz = 0.8f;
            eConstraint.xShearY = eConstraint.xShearZ = 0.8f;
            eConstraint.yShearX = eConstraint.yShearZ = 0.8f;
            eConstraint.zShearX = eConstraint.zShearY = 0.8f;
        }
        else if (e.density < SKIN) {    // SKIN
            eConstraint.dx = 0.4f;  eConstraint.dy = 0.4f;  eConstraint.dz = 0.4f;
            eConstraint.xShearY = eConstraint.xShearZ = 0.4f;
            eConstraint.yShearX = eConstraint.yShearZ = 0.4f;
            eConstraint.zShearX = eConstraint.zShearY = 0.4f;
        }
        else {                          // BONE
            eConstraint.dx = 0.1f;  eConstraint.dy = 0.1f;  eConstraint.dz = 0.1f;
            eConstraint.xShearY = eConstraint.xShearZ = 0.1f;
            eConstraint.yShearX = eConstraint.yShearZ = 0.1f;
            eConstraint.zShearX = eConstraint.zShearY = 0.1f;
        }

        Vec3 sumPos(0, 0, 0);
        Vec3 totalWeight(0, 0, 0);
        int nCnt = 0;

        // 6방향 이웃 처리
        for (int j = 0; j < e.neighborCnt; ++j) {
            const Neighbor& n = neighbors[e.offset + j];
            const Element& nb = elements[n.idx];

            // 이웃 밀도에 따른 제약값
            CMConstraint nConstraint;
            if (nb.density < AIR) {
                nConstraint.dx = 0.8f;  nConstraint.dy = 0.8f;  nConstraint.dz = 0.8f;
                nConstraint.xShearY = nConstraint.xShearZ = 0.8f;
                nConstraint.yShearX = nConstraint.yShearZ = 0.8f;
                nConstraint.zShearX = nConstraint.zShearY = 0.8f;
            }
            else if (nb.density < SKIN) {
                nConstraint.dx = 0.4f;  nConstraint.dy = 0.4f;  nConstraint.dz = 0.4f;
                nConstraint.xShearY = nConstraint.xShearZ = 0.4f;
                nConstraint.yShearX = nConstraint.yShearZ = 0.4f;
                nConstraint.zShearX = nConstraint.zShearY = 0.4f;
            }
            else {
                nConstraint.dx = 0.1f;  nConstraint.dy = 0.1f;  nConstraint.dz = 0.1f;
                nConstraint.xShearY = nConstraint.xShearZ = 0.1f;
                nConstraint.yShearX = nConstraint.yShearZ = 0.1f;
                nConstraint.zShearX = nConstraint.zShearY = 0.1f;
            }

            // 방향별 가중치 계산 (간단화)
            float wX = 1.0f / ((eConstraint.dx + nConstraint.dx) * 0.5f + 1e-6f);
            float wY = 1.0f / ((eConstraint.dy + nConstraint.dy) * 0.5f + 1e-6f);
            float wZ = 1.0f / ((eConstraint.dz + nConstraint.dz) * 0.5f + 1e-6f);

            sumPos.x += nb.pos.x * wX;
            sumPos.y += nb.pos.y * wY;
            sumPos.z += nb.pos.z * wZ;

            totalWeight.x += wX;
            totalWeight.y += wY;
            totalWeight.z += wZ;

            nCnt++;
        }

        if (nCnt > 0) {
            Vec3 newE;
            newE.x = sumPos.x / totalWeight.x;
            newE.y = sumPos.y / totalWeight.y;
            newE.z = sumPos.z / totalWeight.z;
            newPos[idx] = newE;
        }
        else {
            newPos[idx] = e.pos;
        }
    }

    // 위치 갱신
    for (int idx : activeSet) {
        elements[idx].pos = newPos[idx];
    }
}

//void ChainMail::relax(const std::vector<int>& activeSet) {
//    std::vector<Vec3> newPos(elements.size());
//
//    // 활성화된(변형 wave가 도달했던) 정점에 대해서만 relaxation
//    for (int idx : activeSet) {
//        Element& e = elements[idx];
//        Vec3 sumWPos(0, 0, 0);
//        float totalW = 0.0f;
//        for (int j = 0; j < e.neighborCnt; ++j) {
//            const Neighbor& n = neighbors[e.offset + j];
//            const Element& nb = elements[n.idx];
//            // 제약값 등으로 weighted mean
//            float stiffness = 0.3f; // 필요시 material별로 가중치 적용
//            float w = 1.0f / (stiffness + 1e-6f); // 완전히 rigid면 w→0
//            sumWPos = sumWPos + nb.pos * w;
//            totalW += w;
//        }
//        if (totalW > 0)
//            newPos[idx] = sumWPos * (1.0f / totalW);
//        else
//            newPos[idx] = e.pos;
//    }
//    // 좌표 갱신
//    for (int idx : activeSet)
//        elements[idx].pos = newPos[idx];
//}
