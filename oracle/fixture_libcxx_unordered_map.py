#!/usr/bin/env python3
# fixture_libcxx_unordered_map.py: writes ColmapSharp.Tests/TestData/oracle/libcxx_unordered_map.json
# from oracle/libcxx_unordered_map_harness.cc, libc++'s std::unordered_map< int , float >
# iteration order after insert sequences, for ColmapSharp.Tests/Util/LibcxxUnorderedMapTests.cs.
# Tier A. It compiles and runs the harness exactly as oracle/fixture_poisson_bspline.py does
# (Apple clang with the macOS SDK's libc++), reusing its write_fixture.
#
# Usage: oracle/.venv/bin/python oracle/fixture_libcxx_unordered_map.py   (any python3 works;
# COLMAP_REFERENCE=<checkout> when cpp-reference/ is not next to this repo's oracle/)

from fixture_poisson_bspline import write_fixture

if __name__ == "__main__":
    write_fixture("libcxx_unordered_map_harness.cc", "libcxx_unordered_map.json")
