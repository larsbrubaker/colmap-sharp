// libcxx_unordered_map_harness.cc: libc++'s std::unordered_map< int , float > iteration order
// after sequences of inserts (operator[] on a missing key, += on a present one, as
// PoissonRecon's SparseMatrix product does), across rehash boundaries, with negative keys,
// duplicates and colliding keys. Built with the macOS SDK's libc++ and run by
// oracle/fixture_libcxx_unordered_map.py, which writes
// ColmapSharp.Tests/TestData/oracle/libcxx_unordered_map.json (read by
// ColmapSharp.Tests/Util/LibcxxUnorderedMapTests.cs). Not part of any build.
//
// Output: "<name> <kind> <values...>", kind i = integer, f = C99 hex float (the format
// oracle/fixture_poisson_bspline.py parses).

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

unsigned int lcg = 2463534242u;
int NextKey(int lo, int hi) {
  lcg = lcg * 1664525u + 1013904223u;
  return lo + (int)((lcg >> 8) % (unsigned int)(hi - lo + 1));
}

void PrintI(const std::string& name, const std::vector<long long>& v) {
  printf("%s i", name.c_str());
  for (long long x : v) printf(" %lld", x);
  printf("\n");
}

void Run(const std::string& name, const std::vector<int>& keys) {
  std::unordered_map<int, float> map;
  std::vector<long long> buckets;
  for (size_t i = 0; i < keys.size(); i++) {
    auto it = map.find(keys[i]);
    if (it == map.end()) map[keys[i]] = (float)keys[i] * 0.5f;
    else it->second += 1.0f;
    buckets.push_back((long long)map.bucket_count());
  }
  std::vector<long long> order;
  std::vector<double> values;
  for (const auto& kv : map) order.push_back(kv.first), values.push_back(kv.second);
  PrintI(name + "/keys", std::vector<long long>(keys.begin(), keys.end()));
  PrintI(name + "/bucketcounts", buckets);
  PrintI(name + "/order", order);
  printf("%s f", (name + "/values").c_str());
  for (double v : values) printf(" %a", v);
  printf("\n");
}

}  // namespace

int main() {
  const int lengths[] = {1, 2, 3, 4, 5, 8, 12, 13, 24, 30, 50, 97, 200, 500};
  for (int n : lengths) {
    std::vector<int> keys;
    for (int i = 0; i < n; i++) keys.push_back(NextKey(-300, 1000));
    Run("random" + std::to_string(n), keys);
  }
  {
    std::vector<int> keys;
    for (int i = 0; i < 120; i++) keys.push_back(i);
    Run("ascending", keys);
  }
  {
    std::vector<int> keys;
    for (int i = 119; i >= 0; i--) keys.push_back(i * 7);
    Run("descendingmultiples", keys);
  }
  {
    // Few distinct keys, many repeats: lookups of present keys between inserts.
    std::vector<int> keys;
    for (int i = 0; i < 300; i++) keys.push_back(NextKey(-20, 40));
    Run("repeats", keys);
  }
  return 0;
}
