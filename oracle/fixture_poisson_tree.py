#!/usr/bin/env python3
# fixture_poisson_tree.py: writes the oracles for ColmapSharp.Mvs.PoissonRecon's solver
# preparation, one fixture per harness (each stays under about 1 MB), in
# ColmapSharp.Tests/TestData/oracle/:
#   poisson_tree.json   oracle/poisson_tree_harness.cc: the octree stages (PoissonXForm,
#                       PointExtent, PoissonSolutionParameters, FemTree, PoissonSampleSet,
#                       NeighborKey, SortedTreeNodes, PoissonDensity, PoissonSplat,
#                       PoissonInterpolation, PoissonFinalize)
#   poisson_fem.json    oracle/poisson_fem_harness.cc: the FEM integrators and their stencils
#   poisson_system.json oracle/poisson_system_harness.cc: the system assembly after finalizing
#                       (PoissonFemConstraints)
#   poisson_levelset.json oracle/poisson_levelset_harness.cc: what follows the linear solve
#                       (PoissonImplicitEvaluator, the iso-value; PoissonCornerEvaluator)
#   poisson_hypercube.json oracle/poisson_hypercube_harness.cc: the level-set extractor's
#                       hypercube algebra and tables (MarchingCubes.h, FEMTree.LevelSet.inl)
#   poisson_levelset2.json oracle/poisson_levelset2_harness.cc: the extractor's slice and
#                       slab cell indices, corner values and MC indices
#                       (LevelSetCellIndices, PoissonLevelSetExtractor)
#   poisson_levelset3.json oracle/poisson_levelset3_harness.cc: the extractor's iso-vertices
#                       on slice edges (PoissonLevelSetExtractor.IsoVertices)
#   poisson_levelset4.json oracle/poisson_levelset4_harness.cc: every iso-vertex, on slice
#                       and slab edges, in Extract's write order
#                       (PoissonLevelSetExtractor.XSliceIsoVertices)
#   poisson_libm.json   oracle/poisson_libm_harness.cc: libm's pow( x , 1./3 ) and logf
# Tier A, bit-exact. Read by ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests*.cs.
#
# It compiles each harness (they share oracle/poisson_harness.h, and the solve harnesses oracle/poisson_solve.h) against COLMAP's vendored
# PoissonRecon with -ffp-contract=off exactly as oracle/fixture_poisson_bspline.py does (that
# script's header explains why this, and not the -ffast-math pycolmap build, is the oracle),
# runs it and records what it prints.
#
# It also adds to poisson_libm.json "powonethird/correct": the correctly rounded double of
# x^fl(1/3) for each "powonethird/x", computed here with 60-digit decimal arithmetic. The
# harness's "powonethird/y" is the platform libm's pow, which is not correctly rounded for every input
# (docs/CPP_DIVERGENCES.md, entry 75).
#
# Usage: oracle/.venv/bin/python oracle/fixture_poisson_tree.py [fixture.json ...]   (any
# python3 works; COLMAP_REFERENCE=<checkout> when cpp-reference/ is not next to this repo's
# oracle/; naming fixtures regenerates only those)

import math
import sys
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


HARNESSES = [
    ("poisson_tree_harness.cc", "poisson_tree.json", None),
    ("poisson_fem_harness.cc", "poisson_fem.json", None),
    ("poisson_libm_harness.cc", "poisson_libm.json", add_correct_pow),
    ("poisson_system_harness.cc", "poisson_system.json", None),
    ("poisson_levelset_harness.cc", "poisson_levelset.json", None),
    ("poisson_hypercube_harness.cc", "poisson_hypercube.json", None),
    ("poisson_levelset2_harness.cc", "poisson_levelset2.json", None),
    ("poisson_levelset3_harness.cc", "poisson_levelset3.json", None),
    ("poisson_levelset4_harness.cc", "poisson_levelset4.json", None),
]

if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for harness, fixture, postprocess in HARNESSES:
        if not wanted or fixture in wanted:
            write_fixture(harness, fixture, postprocess)
