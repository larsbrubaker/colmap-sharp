#!/usr/bin/env python3
# fixture_mesh_simplification.py: writes
# ColmapSharp.Tests/TestData/oracle/mesh_simplification.json, COLMAP's SimplifyMesh (QEM
# decimation, colmap/mvs/mesh_simplification.cc) on three kinds of input. Read by
# ColmapSharp.Tests/Mvs/MeshSimplificationOracleTests.cs.
#
# - "wavy": a curved 30x30 grid under six option sets, through pycolmap's simplify_mesh. The
#   costs are all distinct; the port matches within 1e-5 (Tier C: boundary systems can differ
#   in the last bits, docs/CPP_DIVERGENCES.md, entry 72) and is byte-identical when generated.
# - "flat": flat grids with boundary_weight 0, through pycolmap. Every collapse costs exactly 0,
#   so the result is decided by std::priority_queue's tie order (libc++'s heap, which
#   ColmapSharp ports); these cases must match byte for byte.
# - "color": a colored curved grid with interpolate_colors true and false. pycolmap's
#   simplify_mesh writes no colors, so oracle/mesh_simplification_harness.cc (SimplifyMesh
#   restated without Eigen, built against libc++) computes them; the script first checks
#   that the harness's positions and faces equal pycolmap's byte for byte on the same input.
#
# Must run on macOS (libc++, like the pycolmap wheel).
# Usage: oracle/.venv/bin/python oracle/fixture_mesh_simplification.py

import json
import pathlib
import struct
import subprocess
import tempfile

import numpy as np
import pycolmap

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle" / "mesh_simplification.json"
HARNESS = HERE / "mesh_simplification_harness.cc"

# (target_face_ratio, boundary_weight, max_error)
WAVY_CASES = [
    (0.1, 1000.0, 0.0),
    (0.3, 1000.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.1, 1e6, 0.0),
    (0.1, 1000.0, 1e-6),
    (0.1, 1000.0, 1e-3),
]

# (grid size, target_face_ratio); boundary_weight 0, max_error 0.
FLAT_CASES = [(4, 0.25), (10, 0.1), (10, 0.3), (10, 0.5), (20, 0.2), (50, 0.1), (50, 0.5), (100, 0.1)]

# (target_face_ratio, boundary_weight, interpolate_colors)
COLOR_CASES = [(0.2, 1000.0, True), (0.2, 1000.0, False)]


def grid_faces(n):
    faces = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = v00 + 1
            v01 = v00 + n + 1
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return faces


def wavy_vertices(n):
    vertices = []
    for j in range(n + 1):
        for i in range(n + 1):
            x = np.float32(i)
            y = np.float32(j)
            z = np.float32(np.sin(x * np.float32(0.5)) * np.cos(y * np.float32(0.5)))
            vertices.append((float(x), float(y), float(z)))
    return vertices


def wavy_colors(n):
    return [(i * 255 // n, j * 255 // n, (i * 37 + j * 91) % 256) for j in range(n + 1) for i in range(n + 1)]


def write_ply(path, vertices, faces):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_index\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for v in vertices:
            f.write(struct.pack("<3f", *v))
        for face in faces:
            f.write(struct.pack("<B3i", 3, *face))


def read_ply(path):
    data = pathlib.Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode()
    assert "binary_little_endian" in header
    assert "property float x\nproperty float y\nproperty float z\nelement face" in header
    num_vertices = int(header.split("element vertex ")[1].split()[0])
    num_faces = int(header.split("element face ")[1].split()[0])
    vertices = np.frombuffer(data, "<f4", num_vertices * 3, end).reshape(-1, 3)
    face_dtype = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    faces = np.frombuffer(data, face_dtype, num_faces, end + num_vertices * 12)
    assert (faces["n"] == 3).all()
    return [float(c) for c in vertices.reshape(-1)], [int(i) for i in faces["i"].reshape(-1)]


def pycolmap_simplify(tmp, vertices, faces, ratio, boundary_weight, max_error):
    input_path = pathlib.Path(tmp) / "input.ply"
    output_path = pathlib.Path(tmp) / "output.ply"
    write_ply(input_path, vertices, faces)
    options = pycolmap.MeshSimplificationOptions()
    options.target_face_ratio = ratio
    options.boundary_weight = boundary_weight
    options.max_error = max_error
    pycolmap.simplify_mesh(input_path, output_path, options)
    return read_ply(output_path)


def harness_simplify(binary, vertices, colors, faces, ratio, boundary_weight, interpolate):
    lines = [f"{len(vertices)} {len(faces)}"]
    lines += [f"{x.hex()} {y.hex()} {z.hex()} {r} {g} {b}" for (x, y, z), (r, g, b) in zip(vertices, colors)]
    lines += [f"{i} {j} {k}" for i, j, k in faces]
    out = subprocess.run([str(binary), repr(ratio), repr(boundary_weight), "0", "1" if interpolate else "0"],
                         input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout.split("\n")
    num_vertices, num_faces = map(int, out[0].split())
    out_vertices, out_colors, out_faces = [], [], []
    for line in out[1:1 + num_vertices]:
        parts = line.split()
        out_vertices += [float.fromhex(p) for p in parts[:3]]
        out_colors += [int(p) for p in parts[3:]]
    for line in out[1 + num_vertices:1 + num_vertices + num_faces]:
        out_faces += [int(p) for p in line.split()]
    return out_vertices, out_colors, out_faces


def main():
    n = 30
    wavy = wavy_vertices(n)
    faces30 = grid_faces(n)
    colors = wavy_colors(n)
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        binary = pathlib.Path(tmp) / "harness"
        subprocess.run(["clang++", "-std=c++17", "-O2", "-stdlib=libc++", "-ffp-contract=off",
                        str(HARNESS), "-o", str(binary)], check=True)

        for ratio, boundary_weight, max_error in WAVY_CASES:
            vertices, faces = pycolmap_simplify(tmp, wavy, faces30, ratio, boundary_weight, max_error)
            cases.append({"input": "wavy", "target_face_ratio": ratio, "boundary_weight": boundary_weight,
                          "max_error": max_error, "interpolate_colors": True,
                          "vertices": vertices, "faces": faces})

        for size, ratio in FLAT_CASES:
            flat = [(float(i), float(j), 0.0) for j in range(size + 1) for i in range(size + 1)]
            vertices, faces = pycolmap_simplify(tmp, flat, grid_faces(size), ratio, 0.0, 0.0)
            cases.append({"input": f"flat{size}", "target_face_ratio": ratio, "boundary_weight": 0.0,
                          "max_error": 0.0, "interpolate_colors": True,
                          "vertices": vertices, "faces": faces})

        for ratio, boundary_weight, interpolate in COLOR_CASES:
            py_vertices, py_faces = pycolmap_simplify(tmp, wavy, faces30, ratio, boundary_weight, 0.0)
            vertices, out_colors, faces = harness_simplify(binary, wavy, colors, faces30, ratio, boundary_weight,
                                                           interpolate)
            assert vertices == py_vertices and faces == py_faces, "harness geometry differs from pycolmap"
            cases.append({"input": "wavy_colored", "target_face_ratio": ratio, "boundary_weight": boundary_weight,
                          "max_error": 0.0, "interpolate_colors": interpolate,
                          "vertices": vertices, "faces": faces, "colors": out_colors})

    fixture = {
        "pycolmap_version": pycolmap.__version__,
        "wavy_vertices": [c for v in wavy for c in v],
        "wavy_colors": [c for rgb in colors for c in rgb],
        "wavy_faces": [i for f in faces30 for i in f],
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(fixture) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
