#!/usr/bin/env python3
# fixture_poisson_tree.py: writes ColmapSharp.Tests/TestData/oracle/poisson_tree.json, the
# oracle for ColmapSharp.Mvs.PoissonRecon's octree stages (PoissonXForm, PointExtent,
# PowOneThird, PoissonSolutionParameters, FemTree, PoissonSampleSet, NeighborKey,
# SortedTreeNodes, PoissonDensity, PoissonSplat). Tier A, bit-exact. Read by
# ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.cs.
#
# It compiles oracle/poisson_tree_harness.cc against COLMAP's vendored PoissonRecon with
# -ffp-contract=off exactly as oracle/fixture_poisson_bspline.py does (that script's header
# explains why this, and not the -ffast-math pycolmap build, is the oracle), runs it and
# records what it prints.
#
# It also adds "powonethird/correct": the correctly rounded double of x^fl(1/3) for each
# "powonethird/x", computed here with 60-digit decimal arithmetic. The harness's
# "powonethird/y" is the platform libm's pow, which is not correctly rounded for every input
# (docs/CPP_DIVERGENCES.md, entry 75).
#
# Usage: oracle/.venv/bin/python oracle/fixture_poisson_tree.py   (any python3 works;
# COLMAP_REFERENCE=<checkout> when cpp-reference/ is not next to this repo's oracle/)

import math
from decimal import Decimal, getcontext

from fixture_poisson_bspline import write_fixture


def correctly_rounded_pow_one_third(x):
    getcontext().prec = 60
    exact = (Decimal(x).ln() * Decimal(1.0 / 3.0)).exp()
    guess = float(exact)
    candidates = [guess, math.nextafter(guess, math.inf), math.nextafter(guess, -math.inf)]
    return min(candidates, key=lambda c: abs(Decimal(c) - exact))


def add_correct_pow(cases):
    cases["powonethird/correct"] = [correctly_rounded_pow_one_third(x) for x in cases["powonethird/x"]]


if __name__ == "__main__":
    write_fixture("poisson_tree_harness.cc", "poisson_tree.json", add_correct_pow)
