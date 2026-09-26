// sift_harness.c: runs VLFeat's SIFT (cpp-reference/src/thirdparty/VLFeat, BSD-2) through the
// same calls COLMAP's SiftCPUFeatureExtractor makes and prints every keypoint, orientation and
// raw float descriptor exactly (C99 hex floats). Built and run by oracle/fixture_sift.py, which
// turns the output into ColmapSharp.Tests/TestData/oracle/sift_vlfeat.json. Not part of any build.
//
// Why a harness in addition to pycolmap: Apple clang contracts a*b + c into fused multiply-adds
// by default (-ffp-contract=on), and the macOS arm64 wheel's VLFeat is built that way, so its
// keypoints differ from an unfused evaluation in the last bits (docs/CPP_DIVERGENCES.md entry 41). This
// harness is compiled with -ffp-contract=off (and without SSE2, like COLMAP's arm64 build) so it
// is the exact, unfused VLFeat the C# port must match bit for bit.
//
// Usage: sift_harness <raw uint8 grey file> <width> <height> <o_min> <noctaves> <upright>
// Output per octave keypoint:  K o ix iy is x y s sigma
//        per used orientation: A angle d0 ... d127   (descriptor before COLMAP's normalization)

#include <stdio.h>
#include <stdlib.h>

#include "sift.h"

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: sift_harness file width height o_min noctaves upright\n");
    return 1;
  }
  int w = atoi(argv[2]), h = atoi(argv[3]), o_min = atoi(argv[4]), noctaves = atoi(argv[5]);
  int upright = atoi(argv[6]);
  unsigned char* data = malloc((size_t)w * h);
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(data, 1, (size_t)w * h, f) != (size_t)w * h) return 1;
  fclose(f);
  float* im = malloc(sizeof(float) * w * h);
  for (int i = 0; i < w * h; ++i) im[i] = (float)data[i] / 255.0f;

  VlSiftFilt* sift = vl_sift_new(w, h, noctaves, 3, o_min);
  vl_sift_set_peak_thresh(sift, 0.02 / 3);
  vl_sift_set_edge_thresh(sift, 10.0);
  int first = 1;
  float desc[128];
  while (1) {
    if (first) {
      if (vl_sift_process_first_octave(sift, im)) break;
      first = 0;
    } else if (vl_sift_process_next_octave(sift)) {
      break;
    }
    vl_sift_detect(sift);
    const VlSiftKeypoint* k = vl_sift_get_keypoints(sift);
    int n = vl_sift_get_nkeypoints(sift);
    for (int i = 0; i < n; ++i) {
      printf("K %d %d %d %d %a %a %a %a\n", k[i].o, k[i].ix, k[i].iy, k[i].is,
             k[i].x, k[i].y, k[i].s, k[i].sigma);
      double angles[4];
      int no;
      if (upright) {
        no = 1;
        angles[0] = 0.0;
      } else {
        no = vl_sift_calc_keypoint_orientations(sift, angles, &k[i]);
      }
      for (int o = 0; o < no; ++o) {
        vl_sift_calc_keypoint_descriptor(sift, desc, &k[i], angles[o]);
        printf("A %a", angles[o]);
        for (int d = 0; d < 128; ++d) printf(" %a", desc[d]);
        printf("\n");
      }
    }
  }
  vl_sift_delete(sift);
  return 0;
}
