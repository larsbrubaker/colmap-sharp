#!/usr/bin/env python3
# fixture_poisson_meshing.py: writes ColmapSharp.Tests/TestData/oracle/poisson_meshing.json,
# COLMAP's PoissonMeshing (colmap/mvs/poisson_meshing.cc: RunPoissonRecon, then
# RunSurfaceTrimmer when trim != 0) through the pinned pycolmap wheel's poisson_meshing, on a
# noisy, colored unit sphere with normals written as a fused.ply would be (float x, y, z,
# nx, ny, nz; uchar red, green, blue). Read by ColmapSharp.Tests/Mvs/PoissonMeshingOracleTests.cs
# (Tier C: the wheel is optimized, the C# stages are pinned bit-exact against -O1 harnesses).
#
# Each case runs in its own Python process: PoissonRecon's command-line flags are globals whose
# "set" state persists across calls in one process (a --density or --fullDepth from an earlier
# call leaks into later ones; docs/CPP_DIVERGENCES.md, entry 131).
#
# Each case stores the output PLY's header text, vertex and triangle counts, positions, density
# values (when present), colors and triangles.
#
# Usage: oracle/.venv/bin/python oracle/fixture_poisson_meshing.py

import json
import math
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle" / "poisson_meshing.json"

# (name, depth, point_weight, trim)
CASES = [
    ("depth5", 5, 1.0, 0.0),
    ("depth5trim", 5, 1.0, 3.6),
    ("depth4unscreened", 4, 0.0, 0.0),
    ("depth6trim", 6, 4.0, 3.5),
]


def sphere_points(n, seed):
    rng = random.Random(seed)
    points = []
    for _ in range(n):
        while True:
            v = [rng.gauss(0.0, 1.0) for _ in range(3)]
            length = math.sqrt(sum(c * c for c in v))
            if length > 1e-6:
                break
        normal = [c / length for c in v]
        radius = 1.0 + rng.uniform(-0.01, 0.01)
        position = [0.25 + radius * normal[0], -0.5 + radius * normal[1], 2.0 + radius * normal[2]]
        # Round through float32 so the fixture holds exactly what the PLY holds.
        position = [struct.unpack("<f", struct.pack("<f", c))[0] for c in position]
        normal = [struct.unpack("<f", struct.pack("<f", c))[0] for c in normal]
        color = [int(127.5 + 127.5 * normal[0]), int(127.5 + 127.5 * normal[1]), rng.randrange(256)]
        points.append((position, normal, color))
    return points


def write_points(path, points):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for position, normal, color in points:
            f.write(struct.pack("<6f3B", *position, *normal, *color))


def read_mesh(path):
    data = pathlib.Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    vertex_props = []
    counts = {}
    element = None
    for line in header.splitlines():
        tokens = line.split()
        if tokens[:1] == ["element"]:
            element = tokens[1]
            counts[element] = int(tokens[2])
        elif tokens[:1] == ["property"] and element == "vertex":
            vertex_props.append((tokens[1], tokens[2]))
        elif tokens[:1] == ["format"]:
            assert tokens[1] == "binary_little_endian", line
    fmt = "<" + "".join({"float": "f", "uchar": "B"}[t] for t, _ in vertex_props)
    size = struct.calcsize(fmt)
    names = [n for _, n in vertex_props]
    positions, values, colors, triangles = [], [], [], []
    offset = end
    for _ in range(counts["vertex"]):
        record = dict(zip(names, struct.unpack_from(fmt, data, offset)))
        offset += size
        positions += [record["x"], record["y"], record["z"]]
        if "value" in record:
            values.append(record["value"])
        if "red" in record:
            colors += [record["red"], record["green"], record["blue"]]
    for _ in range(counts["face"]):
        (count,) = struct.unpack_from("<i", data, offset)
        assert count == 3, count
        triangles += list(struct.unpack_from("<3i", data, offset + 4))
        offset += 16
    assert offset == len(data)
    return {
        "header": header,
        "vertexcount": counts["vertex"],
        "trianglecount": counts["face"],
        "positions": positions,
        "values": values,
        "colors": colors,
        "triangles": triangles,
    }


def run_case(input_path, output_path, depth, point_weight, trim):
    script = (
        "import pycolmap, sys\n"
        "o = pycolmap.PoissonMeshingOptions()\n"
        f"o.depth = {depth}\n"
        f"o.point_weight = {point_weight!r}\n"
        f"o.trim = {trim!r}\n"
        "o.num_threads = 1\n"
        "pycolmap.poisson_meshing(sys.argv[1], sys.argv[2], o)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(input_path), str(output_path)], check=True)


def main():
    points = sphere_points(600, seed=7)
    result = {
        "positions": [c for p, _, _ in points for c in p],
        "normals": [c for _, n, _ in points for c in n],
        "colors": [c for _, _, col in points for c in col],
        "cases": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        input_path = pathlib.Path(tmp) / "points.ply"
        write_points(input_path, points)
        for name, depth, point_weight, trim in CASES:
            output_path = pathlib.Path(tmp) / f"{name}.ply"
            run_case(input_path, output_path, depth, point_weight, trim)
            case = read_mesh(output_path)
            case.update({"depth": depth, "point_weight": point_weight, "trim": trim})
            result["cases"][name] = case
            print(name, case["vertexcount"], case["trianglecount"])
    OUTPUT.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
