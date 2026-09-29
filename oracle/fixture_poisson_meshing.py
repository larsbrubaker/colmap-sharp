#!/usr/bin/env python3
# fixture_poisson_meshing.py: writes two oracles for COLMAP's PoissonMeshing
# (colmap/mvs/poisson_meshing.cc: RunPoissonRecon, then RunSurfaceTrimmer when trim != 0) on a
# noisy, colored unit sphere with normals written as a fused.ply would be (float x, y, z,
# nx, ny, nz; uchar red, green, blue), in ColmapSharp.Tests/TestData/oracle/:
#   poisson_meshing.json        the pinned pycolmap wheel's poisson_meshing. Read by
#                               ColmapSharp.Tests/Mvs/PoissonMeshingOracleTests.cs (Tier C: the
#                               wheel is optimized). Each untrimmed case stores the output PLY's
#                               header text, counts, positions, colors and triangles; each
#                               trimmed case, whose vertices come and go near the trim value,
#                               only the header, counts, surface area and the density values'
#                               min, max and mean, which is all the test compares.
#   poisson_meshing_exact.json  the same input and cases through
#                               oracle/poisson_meshing_harness.cc: upstream's PoissonRecon.cpp and
#                               SurfaceTrimmer.cpp built with clang++ -O1 -ffp-contract=off
#                               -DRELEASE and run with one thread, as the other Poisson harnesses
#                               are built (fixture_poisson_bspline.py's header explains why).
#                               Stores the input PLYs and each output PLY whole (base64). Read by
#                               ColmapSharp.Tests/Mvs/PoissonMeshingOracleTests.Exact.cs (Tier A).
#                               It adds EXACT_ONLY_CASES: a PLY with the color properties and no
#                               points, whose output still declares red, green and blue.
#
# Each case runs in its own process: PoissonRecon's command-line flags are globals whose "set"
# state persists across calls in one process (a --density or --fullDepth from an earlier call
# leaks into later ones; divergence 131).
#
# Floats are written in their shortest float32 round-trip form; the C# tests parse them as float.
#
# Usage: oracle/.venv/bin/python oracle/fixture_poisson_meshing.py   (COLMAP_REFERENCE=<checkout>
# when cpp-reference/ is not next to this repo's oracle/, e.g. from a worktree)

import base64
import json
import math
import os
import pathlib
import platform
import random
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle"
OUTPUT = FIXTURES / "poisson_meshing.json"
EXACT_OUTPUT = FIXTURES / "poisson_meshing_exact.json"
REFERENCE = pathlib.Path(os.environ.get("COLMAP_REFERENCE", HERE.parent / "cpp-reference"))
POISSON_RECON = REFERENCE / "src" / "thirdparty" / "PoissonRecon"
# The other Poisson harnesses' flags, plus RELEASE, which COLMAP's CMakeLists.txt defines for
# PoissonRecon.cpp and SurfaceTrimmer.cpp.
FLAGS = ["-std=c++17", "-O1", "-ffp-contract=off", "-DRELEASE"]

# (name, depth, point_weight, trim)
CASES = [
    ("depth5", 5, 1.0, 0.0),
    ("depth5trim", 5, 1.0, 3.6),
    ("depth4unscreened", 4, 0.0, 0.0),
    ("depth6trim", 6, 4.0, 3.5),
]

# (name, depth, point_weight, trim) on the empty input, harness only. Not trimmed: upstream's
# SurfaceTrimmer crashes on an empty mesh (it reads vertices[0] of an empty vector); the port
# does not, which is divergence 132, pinned by a C#-only test instead.
EXACT_ONLY_CASES = [
    ("empty", 5, 1.0, 0.0),
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


def f32(x):
    """The shortest decimal that reads back as the float32 x."""
    bits = struct.pack("<f", x)
    for digits in range(1, 10):
        text = f"{x:.{digits}g}"
        if struct.pack("<f", float(text)) == bits:
            return text
    raise AssertionError(x)


class Raw(str):
    """Text written into the JSON as is (a number or number list)."""


def floats(values):
    return Raw("[" + ",".join(f32(v) for v in values) + "]")


def to_json(value):
    if isinstance(value, Raw):
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + ":" + to_json(v) for k, v in value.items()) + "}"
    return json.dumps(value, separators=(",", ":"))


def area(positions, triangles):
    """Surface area, summed in double as PoissonMeshingOracleTests.Area does."""
    total = 0.0
    for t in range(0, len(triangles), 3):
        a, b, c = (3 * triangles[t + k] for k in range(3))
        u = [positions[b + d] - positions[a + d] for d in range(3)]
        v = [positions[c + d] - positions[a + d] for d in range(3)]
        cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        total += 0.5 * math.sqrt(sum(x * x for x in cross))
    return total


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
    return header, counts["vertex"], counts["face"], positions, values, colors, triangles


def pycolmap_case(path, trim):
    header, vertices, faces, positions, values, colors, triangles = read_mesh(path)
    case = {"header": header, "vertexcount": vertices, "trianglecount": faces}
    if trim == 0:
        case.update({"positions": floats(positions), "colors": colors, "triangles": triangles})
    else:
        case.update({
            "area": area(positions, triangles),
            "value_min": Raw(f32(min(values))),
            "value_max": Raw(f32(max(values))),
            "value_mean": sum(values) / len(values),
        })
    return case


def run_pycolmap(input_path, output_path, depth, point_weight, trim):
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


def build_harness(binary):
    if not POISSON_RECON.is_dir():
        sys.exit(f"missing {POISSON_RECON}: run scripts/fetch-reference.sh first")
    sources = [HERE / "poisson_meshing_harness.cc", POISSON_RECON / "PoissonRecon.cpp",
               POISSON_RECON / "SurfaceTrimmer.cpp"]
    subprocess.run(["clang++", *FLAGS, "-w", f"-I{POISSON_RECON}", *map(str, sources), "-o", str(binary)],
                   check=True)


def write_exact(exact):
    # One case per line keeps the file diffable.
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in exact.items() if k != "cases"]
    lines.append(' "cases": {')
    items = list(exact["cases"].items())
    for i, (name, case) in enumerate(items):
        lines.append(f"  {json.dumps(name)}: {json.dumps(case)}" + ("," if i + 1 < len(items) else ""))
    lines += [" }", "}"]
    EXACT_OUTPUT.write_text("\n".join(lines) + "\n")
    print("wrote", EXACT_OUTPUT)


def main():
    points = sphere_points(600, seed=7)
    result = {
        "positions": floats([c for p, _, _ in points for c in p]),
        "normals": floats([c for _, n, _ in points for c in n]),
        "colors": [c for _, _, col in points for c in col],
        "cases": {},
    }
    compiler = subprocess.run(["clang++", "--version"], capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    exact = {
        "source": "oracle/fixture_poisson_meshing.py + oracle/poisson_meshing_harness.cc",
        "compiler": compiler,
        "platform": f"{platform.system()} {platform.machine()}",
        "flags": " ".join(FLAGS),
        "cases": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        binary = tmp / "harness"
        build_harness(binary)
        input_path = tmp / "points.ply"
        write_points(input_path, points)
        empty_path = tmp / "empty.ply"
        write_points(empty_path, [])
        inputs = {"sphere": input_path, "empty": empty_path}
        exact["inputs"] = {k: base64.b64encode(v.read_bytes()).decode("ascii") for k, v in inputs.items()}
        runs = [(case, "sphere") for case in CASES] + [(case, "empty") for case in EXACT_ONLY_CASES]
        for (name, depth, point_weight, trim), input_name in runs:
            input_path = inputs[input_name]
            if input_name == "sphere":
                output_path = tmp / f"{name}.ply"
                run_pycolmap(input_path, output_path, depth, point_weight, trim)
                case = {"depth": depth, "point_weight": point_weight, "trim": trim}
                case.update(pycolmap_case(output_path, trim))
                result["cases"][name] = case
                print("pycolmap", name, case["vertexcount"], case["trianglecount"])

            exact_path = tmp / f"{name}.exact.ply"
            subprocess.run([str(binary), str(depth), repr(point_weight), repr(trim), str(input_path),
                            str(exact_path)], check=True, stdout=subprocess.DEVNULL)
            exact["cases"][name] = {
                "input": input_name, "depth": depth, "point_weight": point_weight, "trim": trim,
                "ply": base64.b64encode(exact_path.read_bytes()).decode("ascii"),
            }
            print("harness", name, exact_path.stat().st_size, "bytes")
    OUTPUT.write_text(to_json(result) + "\n")
    print("wrote", OUTPUT)
    write_exact(exact)


if __name__ == "__main__":
    main()
