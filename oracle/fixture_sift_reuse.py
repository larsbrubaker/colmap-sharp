#!/usr/bin/env python3
# fixture_sift_reuse.py: writes ColmapSharp.Tests/TestData/oracle/sift_reuse.json, pycolmap's
# CPU SIFT on image B both from a FRESH extractor and from one extractor REUSED after image A
# (same size). Read by ColmapSharp.Tests/Feature/SiftExtractorReuseTests.cs.
#
# Why: VLFeat caches the gradient of the octave it last computed one for (grad_o) and resets
# that only in vl_sift_new, not in vl_sift_process_first_octave. COLMAP reuses one VlSiftFilt
# per extractor while the image size stays the same, so when A's last octave with keypoints
# is B's first octave (always with num_octaves = 1; with the defaults when A's keypoints all
# sit in octave -1, as on this low-contrast fine-noise A), B's orientations and descriptors in
# that octave are computed from A's gradient. The fixture records that pycolmap's reused
# output differs from its fresh output; the port resets the cache and must match FRESH
# (divergence 43).
#
# Must run on macOS arm64 (the wheel the fixtures describe).
# Usage: oracle/.venv/bin/python oracle/fixture_sift_reuse.py

import base64
import json
import pathlib

import numpy as np
import pycolmap

from fixture_sift import texture_image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "ColmapSharp.Tests" / "TestData" / "oracle" / "sift_reuse.json"

CASES = [
    ("Defaults", {}),
    ("NumOctaves1", {"num_octaves": 1}),
]


def noise_image(width, height, seed, amplitude):
    # Fine, low-contrast noise: SIFT finds keypoints only at the finest (upsampled) octave.
    rng = np.random.default_rng(seed)
    img = 128 + (rng.random((height, width)) - 0.5) * amplitude
    return np.clip(img, 0, 255).astype(np.uint8)


def create(overrides):
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = False
    for key, value in overrides.items():
        setattr(options.sift, key, value)
    return pycolmap.FeatureExtractor.create(options)


def features(extractor, img):
    keypoints, descriptors = extractor.extract_from_uint8_array(img)
    return {
        "keypoints": [[float(k.x), float(k.y), float(k.a11), float(k.a12), float(k.a21), float(k.a22)]
                      for k in keypoints],
        "descriptors": base64.b64encode(np.ascontiguousarray(descriptors.data).tobytes()).decode(),
    }


def main():
    images = {"a": noise_image(157, 118, 11, 30), "b": texture_image(157, 118, 7)}
    cases = []
    for name, overrides in CASES:
        reused_extractor = create(overrides)
        a = features(reused_extractor, images["a"])
        reused = features(reused_extractor, images["b"])
        fresh = features(create(overrides), images["b"])
        cases.append({"name": name, "options": overrides, "a": a, "fresh": fresh, "reused": reused})
        print(name, "A", len(a["keypoints"]), "B fresh", len(fresh["keypoints"]),
              "B reused", len(reused["keypoints"]))
    out = {
        "images": {name: {"width": int(img.shape[1]), "height": int(img.shape[0]),
                          "pixels": base64.b64encode(img.tobytes()).decode()}
                   for name, img in images.items()},
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
