// mesh_simplification_harness.cc: colmap/mvs/mesh_simplification.cc's SimplifyMesh restated
// without Eigen, OpenMP or glog, keeping the vertex colors. Built and run by
// oracle/fixture_mesh_simplification.py for the color cases of
// ColmapSharp.Tests/TestData/oracle/mesh_simplification.json. Not part of any build.
//
// Why a harness and not pycolmap: pycolmap 4.2.0's simplify_mesh writes its result with
// WriteBinaryPlyMesh, which has no color properties, and nothing else it exposes returns a
// simplified mesh. So the colors are computed here, and the fixture script checks that this
// harness's positions and faces equal pycolmap's byte for byte on the same input before it
// trusts its colors.
//
// Restatement choices (COLMAP's own code, BSD-3):
// - The queue is std::priority_queue with COLMAP's cost-only comparator, compiled against
//   libc++ like the macOS wheel, so the collapse order is the wheel's.
// - Eigen's 4x4 determinant and inverse become the Laplace expansion over 2x2 minors
//   (Eberly), as ColmapSharp does; Vector3d/3f/Matrix4d arithmetic is written out per
//   coefficient in Eigen's evaluation order.
// - Boundary edges are visited in (smaller, larger) vertex-index order instead of the
//   boost::unordered_node_map order (divergence 73).
// - Built with -ffp-contract=off (CLAUDE.md, "No FMA").
//
// Input (stdin): "nv nf", then nv lines "x y z r g b" (x, y, z as decimal float values),
// then nf lines "i j k". Arguments: target_face_ratio boundary_weight max_error
// interpolate_colors(0/1). Output: "nv nf", nv lines "x y z r g b" with x, y, z as C99 hex
// floats (%a, exact), then nf lines "i j k".

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <vector>

namespace {

constexpr double kEpsilon = 1e-12;

struct V3 {
  double x, y, z;
};
V3 Sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 Add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 Cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

using M4 = std::array<double, 16>;  // row-major

struct VertexData {
  M4 quadric{};
  V3 position{0, 0, 0};
  float color[3] = {200.0f, 200.0f, 200.0f};
  uint32_t timestamp = 0;
  std::vector<size_t> adjacent_faces;
  std::vector<size_t> adjacent_vertices;
  bool removed = false;
};

struct CollapseCandidate {
  double cost = 0.0;
  size_t v1 = 0;
  size_t v2 = 0;
  V3 optimal_position{0, 0, 0};
  float optimal_color[3] = {0, 0, 0};
  uint32_t timestamp_v1 = 0;
  uint32_t timestamp_v2 = 0;
};

struct CompareCandidateCost {
  bool operator()(const CollapseCandidate& a, const CollapseCandidate& b) const {
    return a.cost > b.cost;
  }
};

void SortedInsert(std::vector<size_t>& vec, size_t val) {
  auto it = std::lower_bound(vec.begin(), vec.end(), val);
  if (it == vec.end() || *it != val) vec.insert(it, val);
}

void SortedErase(std::vector<size_t>& vec, size_t val) {
  auto it = std::lower_bound(vec.begin(), vec.end(), val);
  if (it != vec.end() && *it == val) vec.erase(it);
}

double ComputeQuadricError(const M4& q, V3 p) {
  const double x = p.x, y = p.y, z = p.z;
  const double r0 = x * q[0] + y * q[4] + z * q[8] + q[12];
  const double r1 = x * q[1] + y * q[5] + z * q[9] + q[13];
  const double r2 = x * q[2] + y * q[6] + z * q[10] + q[14];
  const double r3 = x * q[3] + y * q[7] + z * q[11] + q[15];
  return r0 * x + r1 * y + r2 * z + r3;
}

bool Solve(const M4& q, V3& out) {
  const double a00 = q[0], a01 = q[1], a02 = q[2], a03 = q[3];
  const double a10 = q[4], a11 = q[5], a12 = q[6], a13 = q[7];
  const double a20 = q[8], a21 = q[9], a22 = q[10], a23 = q[11];
  const double a30 = 0, a31 = 0, a32 = 0, a33 = 1;
  const double s0 = a00 * a11 - a10 * a01, s1 = a00 * a12 - a10 * a02;
  const double s2 = a00 * a13 - a10 * a03, s3 = a01 * a12 - a11 * a02;
  const double s4 = a01 * a13 - a11 * a03, s5 = a02 * a13 - a12 * a03;
  const double c5 = a22 * a33 - a32 * a23, c4 = a21 * a33 - a31 * a23;
  const double c3 = a21 * a32 - a31 * a22, c2 = a20 * a33 - a30 * a23;
  const double c1 = a20 * a32 - a30 * a22, c0 = a20 * a31 - a30 * a21;
  const double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
  if (!(std::abs(det) > kEpsilon)) return false;
  out = {(-a21 * s5 + a22 * s4 - a23 * s3) / det,
         (a20 * s5 - a22 * s2 + a23 * s1) / det,
         (-a20 * s4 + a21 * s2 - a23 * s0) / det};
  return true;
}

CollapseCandidate ComputeEdgeCollapse(const std::vector<VertexData>& vertices,
                                      size_t v1, size_t v2, bool interpolate_colors) {
  CollapseCandidate candidate;
  candidate.v1 = v1;
  candidate.v2 = v2;
  candidate.timestamp_v1 = vertices[v1].timestamp;
  candidate.timestamp_v2 = vertices[v2].timestamp;
  M4 q_bar;
  for (int i = 0; i < 16; ++i) q_bar[i] = vertices[v1].quadric[i] + vertices[v2].quadric[i];
  const double err_v1 = ComputeQuadricError(q_bar, vertices[v1].position);
  const double err_v2 = ComputeQuadricError(q_bar, vertices[v2].position);
  V3 solved;
  if (Solve(q_bar, solved)) {
    candidate.optimal_position = solved;
    candidate.cost = std::max(0.0, ComputeQuadricError(q_bar, solved));
  } else {
    const V3 sum = Add(vertices[v1].position, vertices[v2].position);
    const V3 mid = {0.5 * sum.x, 0.5 * sum.y, 0.5 * sum.z};
    const double err_mid = ComputeQuadricError(q_bar, mid);
    if (err_v1 <= err_v2 && err_v1 <= err_mid) {
      candidate.optimal_position = vertices[v1].position;
      candidate.cost = std::max(0.0, err_v1);
    } else if (err_v2 <= err_mid) {
      candidate.optimal_position = vertices[v2].position;
      candidate.cost = std::max(0.0, err_v2);
    } else {
      candidate.optimal_position = mid;
      candidate.cost = std::max(0.0, err_mid);
    }
  }
  if (interpolate_colors) {
    const V3 edge_dir = Sub(vertices[v2].position, vertices[v1].position);
    const double edge_len_sq = Dot(edge_dir, edge_dir);
    float t = 0.5f;
    if (edge_len_sq > 0) {
      t = static_cast<float>(
          Dot(Sub(candidate.optimal_position, vertices[v1].position), edge_dir) / edge_len_sq);
      t = std::max(0.0f, std::min(t, 1.0f));
    }
    for (int c = 0; c < 3; ++c) {
      candidate.optimal_color[c] = (1.0f - t) * vertices[v1].color[c] + t * vertices[v2].color[c];
    }
  } else {
    const float* color = (err_v1 <= err_v2) ? vertices[v1].color : vertices[v2].color;
    for (int c = 0; c < 3; ++c) candidate.optimal_color[c] = color[c];
  }
  return candidate;
}

bool WouldCauseFlip(const std::vector<VertexData>& vertices,
                    const std::vector<std::array<size_t, 3>>& face_indices,
                    const std::vector<bool>& face_removed, size_t v1, size_t v2, V3 new_pos) {
  const auto would_flip_vertex = [&](size_t v_check, size_t v_other) {
    for (const size_t fi : vertices[v_check].adjacent_faces) {
      if (face_removed[fi]) continue;
      const auto& f = face_indices[fi];
      if (f[0] == v_other || f[1] == v_other || f[2] == v_other) continue;
      const V3 p0 = vertices[f[0]].position, p1 = vertices[f[1]].position,
               p2 = vertices[f[2]].position;
      const V3 old_normal = Cross(Sub(p1, p0), Sub(p2, p0));
      V3 np0 = p0, np1 = p1, np2 = p2;
      if (f[0] == v_check) {
        np0 = new_pos;
      } else if (f[1] == v_check) {
        np1 = new_pos;
      } else if (f[2] == v_check) {
        np2 = new_pos;
      }
      const V3 new_normal = Cross(Sub(np1, np0), Sub(np2, np0));
      if (Dot(old_normal, new_normal) < 0) return true;
    }
    return false;
  };
  return would_flip_vertex(v1, v2) || would_flip_vertex(v2, v1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: harness ratio boundary_weight max_error interpolate_colors\n");
    return 1;
  }
  const double target_face_ratio = std::atof(argv[1]);
  const double boundary_weight = std::atof(argv[2]);
  const double max_error = std::atof(argv[3]);
  const bool interpolate_colors = std::atoi(argv[4]) != 0;

  size_t num_vertices = 0, num_faces = 0;
  if (std::scanf("%zu %zu", &num_vertices, &num_faces) != 2) return 1;
  std::vector<VertexData> vertex_data(num_vertices);
  for (auto& v : vertex_data) {
    float x, y, z;
    int r, g, b;
    if (std::scanf("%f %f %f %d %d %d", &x, &y, &z, &r, &g, &b) != 6) return 1;
    v.position = {x, y, z};
    v.color[0] = static_cast<float>(static_cast<uint8_t>(r));
    v.color[1] = static_cast<float>(static_cast<uint8_t>(g));
    v.color[2] = static_cast<float>(static_cast<uint8_t>(b));
  }
  std::vector<std::array<size_t, 3>> face_indices(num_faces);
  for (auto& f : face_indices) {
    if (std::scanf("%zu %zu %zu", &f[0], &f[1], &f[2]) != 3) return 1;
  }

  const size_t target_faces = std::max(
      static_cast<size_t>(1), static_cast<size_t>(std::floor(num_faces * target_face_ratio)));
  std::vector<bool> face_removed(num_faces, false);
  for (size_t fi = 0; fi < num_faces; ++fi) {
    const auto& f = face_indices[fi];
    if (f[0] == f[1] || f[0] == f[2] || f[1] == f[2]) {
      face_removed[fi] = true;
      continue;
    }
    for (int j = 0; j < 3; ++j) {
      const size_t va = f[j];
      const size_t vb = f[(j + 1) % 3];
      SortedInsert(vertex_data[va].adjacent_faces, fi);
      SortedInsert(vertex_data[va].adjacent_vertices, vb);
      SortedInsert(vertex_data[vb].adjacent_vertices, va);
    }
  }
  size_t current_faces = 0;
  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (!face_removed[fi]) ++current_faces;
  }

  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (face_removed[fi]) continue;
    const auto& f = face_indices[fi];
    const V3 p0 = vertex_data[f[0]].position;
    V3 normal = Cross(Sub(vertex_data[f[1]].position, p0), Sub(vertex_data[f[2]].position, p0));
    const double len = std::sqrt(Dot(normal, normal));
    if (len < kEpsilon) continue;
    normal = {normal.x / len, normal.y / len, normal.z / len};
    const double plane[4] = {normal.x, normal.y, normal.z, -Dot(normal, p0)};
    for (int k = 0; k < 3; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) vertex_data[f[k]].quadric[r * 4 + c] += plane[r] * plane[c];
      }
    }
  }

  if (boundary_weight > 0) {
    for (size_t a = 0; a < num_vertices; ++a) {
      for (const size_t b : vertex_data[a].adjacent_vertices) {
        if (b <= a) continue;
        int num_edge_faces = 0;
        size_t fi = 0;
        for (const size_t c : vertex_data[a].adjacent_faces) {
          const auto& f = face_indices[c];
          if (f[0] == b || f[1] == b || f[2] == b) {
            ++num_edge_faces;
            fi = c;
          }
        }
        if (num_edge_faces != 1) continue;
        const auto& f = face_indices[fi];
        const V3 p0 = vertex_data[f[0]].position;
        const V3 face_normal =
            Cross(Sub(vertex_data[f[1]].position, p0), Sub(vertex_data[f[2]].position, p0));
        V3 cn = Cross(Sub(vertex_data[b].position, vertex_data[a].position), face_normal);
        const double clen = std::sqrt(Dot(cn, cn));
        if (clen < kEpsilon) continue;
        cn = {cn.x / clen, cn.y / clen, cn.z / clen};
        const double cplane[4] = {cn.x, cn.y, cn.z, -Dot(cn, vertex_data[a].position)};
        for (int r = 0; r < 4; ++r) {
          const double weighted = boundary_weight * cplane[r];
          for (int c = 0; c < 4; ++c) {
            vertex_data[a].quadric[r * 4 + c] += weighted * cplane[c];
            vertex_data[b].quadric[r * 4 + c] += weighted * cplane[c];
          }
        }
      }
    }
  }

  std::vector<CollapseCandidate> initial_candidates;
  for (size_t vi = 0; vi < num_vertices; ++vi) {
    for (const size_t vj : vertex_data[vi].adjacent_vertices) {
      if (vi < vj) {
        initial_candidates.push_back(ComputeEdgeCollapse(vertex_data, vi, vj, interpolate_colors));
      }
    }
  }
  std::priority_queue<CollapseCandidate, std::vector<CollapseCandidate>, CompareCandidateCost> pq(
      CompareCandidateCost{}, std::move(initial_candidates));

  while (current_faces > target_faces && !pq.empty()) {
    const CollapseCandidate candidate = pq.top();
    pq.pop();
    if (vertex_data[candidate.v1].removed || vertex_data[candidate.v2].removed) continue;
    if (vertex_data[candidate.v1].timestamp != candidate.timestamp_v1 ||
        vertex_data[candidate.v2].timestamp != candidate.timestamp_v2) {
      continue;
    }
    if (max_error > 0 && candidate.cost > max_error) break;
    if (WouldCauseFlip(vertex_data, face_indices, face_removed, candidate.v1, candidate.v2,
                       candidate.optimal_position)) {
      continue;
    }
    const size_t v1 = candidate.v1;
    const size_t v2 = candidate.v2;
    vertex_data[v1].position = candidate.optimal_position;
    for (int c = 0; c < 3; ++c) vertex_data[v1].color[c] = candidate.optimal_color[c];
    for (int i = 0; i < 16; ++i) vertex_data[v1].quadric[i] += vertex_data[v2].quadric[i];
    for (const size_t fi : vertex_data[v2].adjacent_faces) {
      if (face_removed[fi]) continue;
      auto& f = face_indices[fi];
      const bool has_v1 = (f[0] == v1 || f[1] == v1 || f[2] == v1);
      if (has_v1) {
        face_removed[fi] = true;
        --current_faces;
        for (int j = 0; j < 3; ++j) {
          if (f[j] != v1 && f[j] != v2) SortedErase(vertex_data[f[j]].adjacent_faces, fi);
        }
        SortedErase(vertex_data[v1].adjacent_faces, fi);
      } else {
        for (int j = 0; j < 3; ++j) {
          if (f[j] == v2) {
            f[j] = v1;
            break;
          }
        }
        if (f[0] == f[1] || f[0] == f[2] || f[1] == f[2]) {
          face_removed[fi] = true;
          --current_faces;
          for (int j = 0; j < 3; ++j) SortedErase(vertex_data[f[j]].adjacent_faces, fi);
        } else {
          SortedInsert(vertex_data[v1].adjacent_faces, fi);
        }
      }
    }
    for (const size_t u : vertex_data[v2].adjacent_vertices) {
      if (u == v1) continue;
      SortedErase(vertex_data[u].adjacent_vertices, v2);
      SortedInsert(vertex_data[u].adjacent_vertices, v1);
      SortedInsert(vertex_data[v1].adjacent_vertices, u);
    }
    SortedErase(vertex_data[v1].adjacent_vertices, v2);
    vertex_data[v2].removed = true;
    vertex_data[v2].adjacent_faces.clear();
    vertex_data[v2].adjacent_vertices.clear();
    ++vertex_data[v1].timestamp;
    for (const size_t u : vertex_data[v1].adjacent_vertices) {
      if (vertex_data[u].removed) continue;
      pq.push(ComputeEdgeCollapse(vertex_data, v1, u, interpolate_colors));
    }
  }

  constexpr size_t kUnmapped = std::numeric_limits<size_t>::max();
  std::vector<size_t> old_to_new(num_vertices, kUnmapped);
  std::vector<size_t> out_vertices;
  std::vector<std::array<size_t, 3>> out_faces;
  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (face_removed[fi]) continue;
    const auto& f = face_indices[fi];
    for (int j = 0; j < 3; ++j) {
      if (old_to_new[f[j]] == kUnmapped) {
        old_to_new[f[j]] = out_vertices.size();
        out_vertices.push_back(f[j]);
      }
    }
    out_faces.push_back({old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]});
  }
  std::printf("%zu %zu\n", out_vertices.size(), out_faces.size());
  for (const size_t v : out_vertices) {
    const auto& vd = vertex_data[v];
    int rgb[3];
    for (int c = 0; c < 3; ++c) {
      // Eigen's color.array().round().max(0.0f).min(255.0f), then static_cast<uint8_t>.
      const float clamped = std::min(std::max(std::round(vd.color[c]), 0.0f), 255.0f);
      rgb[c] = static_cast<uint8_t>(clamped);
    }
    std::printf("%a %a %a %d %d %d\n", static_cast<float>(vd.position.x),
                static_cast<float>(vd.position.y), static_cast<float>(vd.position.z), rgb[0],
                rgb[1], rgb[2]);
  }
  for (const auto& f : out_faces) std::printf("%zu %zu %zu\n", f[0], f[1], f[2]);
  return 0;
}
