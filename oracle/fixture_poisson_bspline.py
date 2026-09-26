#!/usr/bin/env python3
# fixture_poisson_bspline.py: writes ColmapSharp.Tests/TestData/oracle/poisson_bspline.json,
# the oracle for ColmapSharp.Mvs.PoissonRecon's polynomial and B-spline machinery
# (PoissonPolynomial, BSplineSupportSizes/OverlapSizes, BSplineElements,
# BSplineEvaluationData and its evaluators, BSplineIntegrationData and BSplineData). Tier A,
# bit-exact. Read by ColmapSharp.Tests/Mvs/PoissonRecon/PoissonBSplineOracleTests.cs.
#
# How it is produced: the PoissonRecon templates are header-only, so this script compiles
# oracle/poisson_bspline_harness.cc against COLMAP's vendored copy in cpp-reference/
# (scripts/fetch-reference.sh) with -ffp-contract=off - the arithmetic ColmapSharp does -
# runs it, and records what it prints. COLMAP itself builds PoissonRecon with -ffast-math;
# that build is not reproducible bit for bit and is not the oracle here
# (docs/CPP_DIVERGENCES.md, entry 74).
#
# Usage: oracle/.venv/bin/python oracle/fixture_poisson_bspline.py   (any python3 works)

import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
# COLMAP_REFERENCE overrides the reference checkout's location (e.g. when run from a worktree).
REFERENCE = pathlib.Path(os.environ.get("COLMAP_REFERENCE", HERE.parent / "cpp-reference"))
POISSON_RECON = REFERENCE / "src" / "thirdparty" / "PoissonRecon"
FIXTURES = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle"
FLAGS = ["-std=c++17", "-O1", "-ffp-contract=off"]


def finite_or_name(x):
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def parse(output):
    cases = {}
    for line in output.splitlines():
        name, kind, *values = line.split()
        if kind == "f":
            # Hex floats are exact; json writes Python floats with repr, which round-trips.
            # Non-finite values are written as the strings "NaN", "Infinity" and "-Infinity"
            # (JSON has no literal for them); the C# reader accepts both forms.
            parsed = [finite_or_name(float.fromhex(v)) for v in values]
        elif kind == "c":
            # A checksummed dump (see poisson_harness.h's PrintChunks).
            parsed = {"count": int(values[0]), "float": values[1] == "1",
                      "chunks": [int(v) for v in values[2:]]}
        else:
            parsed = [int(v) for v in values]
        cases[name] = parsed
    return cases


def write_fixture(harness, fixture, postprocess=None):
    """Builds oracle/<harness>, runs it and writes TestData/oracle/<fixture>."""
    source = HERE / harness
    output_path = FIXTURES / fixture
    if not POISSON_RECON.is_dir():
        sys.exit(f"missing {POISSON_RECON}: run scripts/fetch-reference.sh first")

    compiler = subprocess.run(["clang++", "--version"], capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        binary = pathlib.Path(tmp) / "harness"
        subprocess.run(["clang++", *FLAGS, f"-I{POISSON_RECON}", str(source), "-o", str(binary)],
                       check=True)
        output = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout

    cases = parse(output)
    if postprocess:
        postprocess(cases)
    header = {
        "source": f"oracle/{pathlib.Path(sys.argv[0]).name} + oracle/{harness}",
        "compiler": compiler,
        "platform": f"{platform.system()} {platform.machine()}",
        "flags": " ".join(FLAGS),
    }
    # One case per line keeps the file diffable without an indent per number.
    lines = ["{"]
    for key, value in header.items():
        lines.append(f" {json.dumps(key)}: {json.dumps(value)},")
    lines.append(' "cases": {')
    items = list(cases.items())
    for i, (name, values) in enumerate(items):
        comma = "," if i + 1 < len(items) else ""
        lines.append(f"  {json.dumps(name)}: {json.dumps(values)}{comma}")
    lines.append(" }")
    lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(cases)} cases to {output_path}")


def main():
    write_fixture("poisson_bspline_harness.cc", "poisson_bspline.json")


if __name__ == "__main__":
    main()
