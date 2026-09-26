#!/usr/bin/env python3
# fixture_synthetic.py: writes ColmapSharp.Tests/TestData/oracle/synthetic.json, the pycolmap
# 4.2.0 oracle for ColmapSharp.Scene.Synthetic.SynthesizeDataset (colmap/scene/synthetic.cc).
# Read by ColmapSharp.Tests/Scene/SyntheticOracleTests.cs.
#
# Each case seeds COLMAP's PRNG (pycolmap.set_random_seed), synthesizes a reconstruction
# without a database and records every camera, rig, frame, image (with all 2D points) and
# 3D point. Values are written with repr(), so doubles round-trip exactly.
#
# NOISE_CASES additionally run synthesize_noise (after pruning tracks, where track_length is set). COLMAP visits
# 3D points and images in hash order there and ColmapSharp in ascending id order
# (docs/CPP_DIVERGENCES.md entry 31), so these record what can be compared order-insensitively:
# the track lengths after pruning, the frame poses after noise (frames are visited in the same
# order), and per image / per 3D point the noise that was added (noisy minus clean value), whose
# chunks of the Gaussian stream match as a multiset. SUMMARIZED_NOISE_CASES record per-image
# summaries of the 2D noise and clean observations instead of every 2D point, plus the sum of
# the clean 3D point positions (exact: uniform draws summed in id order), to keep the file small.
#
# Must run on macOS: the draws go through libc++'s <random> (see fixture_random.py).
#
# Usage: oracle/.venv/bin/python oracle/fixture_synthetic.py

import json
import pathlib
import platform
import sys

import pycolmap

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle" / "synthetic.json"

CASES = [
    # name, seed, options
    ("default", 42, {}),
    ("two_cameras_per_rig", 7, {"num_rigs": 2, "num_cameras_per_rig": 2, "num_frames_per_rig": 3,
                                "num_points3D": 30, "num_points2D_without_point3D": 5}),
]

NOISE_CASES = [
    # name, seed, dataset options, noise options
    ("track_length_noise", 11,
     {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 4, "num_points3D": 30,
      "track_length": 3},
     {"rig_from_world_translation_stddev": 0.1, "rig_from_world_rotation_stddev": 0.5,
      "point3D_stddev": 0.05, "point2D_stddev": 0.5}),
    # The "Nominal" noise of bundle_adjustment_test.cc on a dataset large enough for the
    # hash order to scramble which image and 3D point receive which draws.
    ("nominal_ba_noise_100_frames", 0,
     {"num_rigs": 1, "num_cameras_per_rig": 1, "num_frames_per_rig": 100, "num_points3D": 2000},
     {"rig_from_world_translation_stddev": 0.1, "rig_from_world_rotation_stddev": 0.5,
      "point3D_stddev": 0.1, "point2D_stddev": 0.5}),
]

# Noise cases too large to record every 2D point (100 images x 2010 points): these record
# per-image summaries instead (see summarize_image_noise and summarize_observations).
SUMMARIZED_NOISE_CASES = {"nominal_ba_noise_100_frames"}


def rigid(t):
    q = t.rotation.quat  # (x, y, z, w), Eigen memory order
    return {"q_xyzw": [float(v) for v in q], "t": [float(v) for v in t.translation]}


def dump(reconstruction):
    cameras = []
    for camera_id, camera in sorted(reconstruction.cameras.items()):
        cameras.append({"camera_id": camera_id, "model_id": int(camera.model.value),
                        "width": camera.width, "height": camera.height,
                        "params": [float(p) for p in camera.params]})
    rigs = []
    for rig_id, rig in sorted(reconstruction.rigs.items()):
        sensors = []
        for sensor_id, sensor_from_rig in sorted(rig.non_ref_sensors.items(), key=lambda kv: kv[0].id):
            sensors.append({"camera_id": sensor_id.id, "sensor_from_rig": rigid(sensor_from_rig)})
        rigs.append({"rig_id": rig_id, "ref_camera_id": rig.ref_sensor_id.id, "sensors": sensors})
    frames = []
    for frame_id, frame in sorted(reconstruction.frames.items()):
        frames.append({"frame_id": frame_id, "rig_id": frame.rig_id,
                       "rig_from_world": rigid(frame.rig_from_world),
                       "image_ids": sorted(d.id for d in frame.data_ids)})
    images = []
    for image_id, image in sorted(reconstruction.images.items()):
        points2D = []
        for p in image.points2D:
            pid = p.point3D_id if p.has_point3D() else -1
            points2D.append([float(p.xy[0]), float(p.xy[1]), pid])
        images.append({"image_id": image_id, "name": image.name, "camera_id": image.camera_id,
                       "frame_id": image.frame_id, "points2D": points2D})
    points3D = []
    for point3D_id, point3D in sorted(reconstruction.points3D.items()):
        points3D.append({"point3D_id": point3D_id, "xyz": [float(v) for v in point3D.xyz],
                         "error": float(point3D.error),
                         "track": sorted([e.image_id, e.point2D_idx] for e in point3D.track.elements)})
    return {"cameras": cameras, "rigs": rigs, "frames": frames, "images": images,
            "points3D": points3D}


def naive_sum(values):
    """Left-to-right double addition, as the C# test does. Python 3.12's sum() of floats is
    compensated (Neumaier), so its result can differ from a plain loop in the last bits."""
    total = 0.0
    for v in values:
        total += v
    return total


def summarize_image_noise(deltas):
    """[count, first three (dx, dy), sum dx, sum dy, sum dx^2 + dy^2], summed in index order."""
    flat = [v for d in deltas[:3] for v in d]
    return ([len(deltas)] + flat + [naive_sum(d[0] for d in deltas), naive_sum(d[1] for d in deltas),
                                     naive_sum(d[0] * d[0] + d[1] * d[1] for d in deltas)])


def summarize_observations(image):
    """The clean 2D points of an image, order-insensitively: the number observing a 3D point,
    every point without one (index, x, y), and sums x, y, id * x, id * y over the observing
    points in ascending 3D point id order (which 3D point sits at which index is hash order)."""
    observing = sorted((p.point3D_id, float(p.xy[0]), float(p.xy[1]))
                       for p in image.points2D if p.has_point3D())
    without = [[idx, float(p.xy[0]), float(p.xy[1])]
               for idx, p in enumerate(image.points2D) if not p.has_point3D()]
    sums = [0.0, 0.0, 0.0, 0.0]
    for pid, x, y in observing:
        sums[0] += x
        sums[1] += y
        sums[2] += pid * x
        sums[3] += pid * y
    return {"image_id": image.image_id, "num_with_point3D": len(observing),
            "without_point3D": without, "sums": sums}


def main():
    if platform.system() != "Darwin":
        sys.exit("Run on macOS: the fixture must come from the libc++ pycolmap wheel.")
    cases = []
    for name, seed, overrides in CASES:
        options = pycolmap.SyntheticDatasetOptions()
        for key, value in overrides.items():
            setattr(options, key, value)
        pycolmap.set_random_seed(seed)
        reconstruction = pycolmap.synthesize_dataset(options)
        cases.append({"name": name, "seed": seed, "options": overrides, **dump(reconstruction)})
    noise_cases = []
    for name, seed, overrides, noise_overrides in NOISE_CASES:
        options = pycolmap.SyntheticDatasetOptions()
        for key, value in overrides.items():
            setattr(options, key, value)
        noise_options = pycolmap.SyntheticNoiseOptions()
        for key, value in noise_overrides.items():
            setattr(noise_options, key, value)
        pycolmap.set_random_seed(seed)
        reconstruction = pycolmap.synthesize_dataset(options)
        track_lengths = {pid: p.track.length() for pid, p in reconstruction.points3D.items()}
        clean2D = {iid: [(float(p.xy[0]), float(p.xy[1])) for p in im.points2D]
                   for iid, im in reconstruction.images.items()}
        clean3D = {pid: [float(v) for v in p.xyz] for pid, p in reconstruction.points3D.items()}
        summarize = name in SUMMARIZED_NOISE_CASES
        observations = [summarize_observations(im) for _, im in sorted(reconstruction.images.items())]
        pycolmap.synthesize_noise(noise_options, reconstruction)
        points2D_deltas = []
        for iid, im in sorted(reconstruction.images.items()):
            points2D_deltas.append([[float(p.xy[0]) - c[0], float(p.xy[1]) - c[1]]
                                    for p, c in zip(im.points2D, clean2D[iid])])
        points3D_deltas = [[float(v) - c for v, c in zip(p.xyz, clean3D[pid])]
                           for pid, p in sorted(reconstruction.points3D.items())]
        frames = [{"frame_id": fid, "rig_from_world": rigid(f.rig_from_world)}
                  for fid, f in sorted(reconstruction.frames.items())]
        case = {"name": name, "seed": seed, "options": overrides,
                "noise_options": noise_overrides,
                "track_lengths": [track_lengths[k] for k in sorted(track_lengths)],
                "point3D_ids": sorted(track_lengths),
                "frames": frames, "points3D_deltas": points3D_deltas}
        if summarize:
            case["points2D_delta_summaries"] = [summarize_image_noise(d) for d in points2D_deltas]
            case["observations"] = observations
            case["points3D_clean_sum"] = [naive_sum(clean3D[pid][k] for pid in sorted(clean3D))
                                          for k in range(3)]
        else:
            case["points2D_deltas"] = points2D_deltas
        noise_cases.append(case)
    OUTPUT.write_text(json.dumps({"pycolmap": pycolmap.__version__, "cases": cases,
                                  "noise_cases": noise_cases}, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
