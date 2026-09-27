#!/usr/bin/env python3
# fixture_global_sfm.py: writes ColmapSharp.Tests/TestData/oracle/global_sfm.json, the Tier C
# (outcome) pycolmap 4.2.0 oracle for global SfM: the rotation averaging estimator
# (pycolmap.run_rotation_averaging, ColmapSharp/Estimators/RotationAveraging.cs) and the whole
# global pipeline (pycolmap.global_mapping, ColmapSharp/Controllers/GlobalPipeline.cs, which
# runs rotation averaging, track establishment, global positioning, bundle adjustment and
# retriangulation). Read by ColmapSharp.Tests/Controllers/GlobalSfmOracleTests.cs.
#
# Each case synthesizes a seeded dataset into a database (pycolmap.synthesize_dataset), and
# optionally adds noise and corrupts a few relative rotations. The fixture records the whole
# database the pipeline reads (cameras, rigs, frames, images, keypoints, two-view geometries),
# so the C# test runs on exactly the same input without re-synthesizing it (synthetic noise
# visits images in hash order, docs/CPP_DIVERGENCES.md entry 31). Raw matches and descriptors
# are not recorded: neither the rotation averaging nor the pipeline reads them.
#
# Outputs: the ground-truth cam_from_world of every image, the rotation averaging result
# (cam_from_world rotation per registered image) and every reconstruction of the pipeline
# (registered image ids, cam_from_world per image, number of 3D points). Both runs are
# single-threaded with a fixed random seed. Doubles are written with repr() (exact); the
# float32 keypoints with numpy's shortest float32 repr (exact in C# float.Parse).
#
# Must run on macOS (the libc++ pycolmap wheel, as the C# PRNG matches libc++).
#
# Usage: oracle/.venv/bin/python oracle/fixture_global_sfm.py

import json
import math
import pathlib
import platform
import sys
import tempfile

import numpy as np
import pycolmap

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle" / "global_sfm.json"

SEED = 1

CASES = [
    # name, seed, dataset options, noise options (or None), number of corrupted pairs
    ("noise_free", 5,
     {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 5, "num_points3D": 80,
      "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True},
     None, 0),
    # Outlier matches, 2D noise and a few relative rotations replaced by random ones.
    ("noisy_outliers", 6,
     {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 5, "num_points3D": 80,
      "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True,
      "inlier_match_ratio": 0.8},
     {"point2D_stddev": 0.5}, 3),
    ("two_cameras_per_rig", 7,
     {"num_rigs": 2, "num_cameras_per_rig": 2, "num_frames_per_rig": 3, "num_points3D": 80,
      "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True},
     None, 0),
]


def rigid(t):
    return {"q_xyzw": [float(v) for v in t.rotation.quat], "t": [float(v) for v in t.translation]}


def f32(v):
    return float(np.format_float_positional(np.float32(v), unique=True, trim="-"))


def matrix(m):
    return None if m is None else [float(v) for v in np.asarray(m).reshape(-1)]


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return pycolmap.Rotation3d(np.array([q[0], q[1], q[2], q[3]]))  # (x, y, z, w)


def corrupt_pairs(database, count, rng):
    """Replaces the relative rotation of the first `count` pairs (pair id order) whose two
    images belong to different frames with a random rotation, keeping the translation."""
    corrupted = []
    images = {im.image_id: im for im in database.read_all_images()}
    pair_ids, geometries = database.read_two_view_geometries()
    for pair_id, tvg in sorted(zip(pair_ids, geometries), key=lambda p: p[0]):
        if len(corrupted) == count:
            break
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        if images[id1].frame_id == images[id2].frame_id or tvg.cam2_from_cam1 is None:
            continue
        tvg.cam2_from_cam1 = pycolmap.Rigid3d(random_rotation(rng), tvg.cam2_from_cam1.translation)
        database.update_two_view_geometry(id1, id2, tvg)
        corrupted.append([id1, id2])
    return corrupted


def dump_database(database):
    cameras = [{"camera_id": c.camera_id, "model_id": int(c.model.value), "width": c.width,
                "height": c.height, "params": [float(p) for p in c.params],
                "has_prior_focal_length": bool(c.has_prior_focal_length)}
               for c in sorted(database.read_all_cameras(), key=lambda c: c.camera_id)]
    rigs = []
    for rig in sorted(database.read_all_rigs(), key=lambda r: r.rig_id):
        sensors = [{"camera_id": sid.id, "sensor_from_rig": rigid(t)}
                   for sid, t in sorted(rig.non_ref_sensors.items(), key=lambda kv: kv[0].id)]
        rigs.append({"rig_id": rig.rig_id, "ref_camera_id": rig.ref_sensor_id.id, "sensors": sensors})
    frames = [{"frame_id": f.frame_id, "rig_id": f.rig_id,
               "data_ids": sorted([d.sensor_id.id, d.id] for d in f.data_ids)}
              for f in sorted(database.read_all_frames(), key=lambda f: f.frame_id)]
    images = []
    for im in sorted(database.read_all_images(), key=lambda i: i.image_id):
        keypoints = database.read_keypoints(im.image_id)
        # Synthetic keypoints are FeatureKeypoint(x, y): identity affine shape.
        assert np.array_equal(keypoints[:, 2:], np.tile([1, 0, 0, 1], (len(keypoints), 1))), im.name
        images.append({"image_id": im.image_id, "name": im.name, "camera_id": im.camera_id,
                       "frame_id": im.frame_id,
                       "keypoints": [f32(v) for v in keypoints[:, :2].reshape(-1)]})
    pairs = []
    pair_ids, geometries = database.read_two_view_geometries()
    for pair_id, tvg in sorted(zip(pair_ids, geometries), key=lambda p: p[0]):
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        assert tvg.camera1 is None and tvg.camera2 is None
        pairs.append({"image_id1": id1, "image_id2": id2, "config": int(tvg.config),
                      "E": matrix(tvg.E), "F": matrix(tvg.F), "H": matrix(tvg.H),
                      "cam2_from_cam1": None if tvg.cam2_from_cam1 is None else rigid(tvg.cam2_from_cam1),
                      "inlier_matches": [int(v) for v in np.asarray(tvg.inlier_matches).reshape(-1)]})
    return {"cameras": cameras, "rigs": rigs, "frames": frames, "images": images, "pairs": pairs}


def run_rotation_averaging(database):
    cache = pycolmap.DatabaseCache.create(database, pycolmap.DatabaseCacheOptions())
    reconstruction = pycolmap.Reconstruction()
    reconstruction.load(cache)
    pose_graph = pycolmap.PoseGraph()
    pose_graph.load(cache.correspondence_graph)
    options = pycolmap.RotationEstimatorOptions()
    options.random_seed = SEED
    ok = pycolmap.run_rotation_averaging(options, pose_graph, reconstruction,
                                         database.read_all_pose_priors())
    rotations = [{"image_id": iid, "q_xyzw": [float(v) for v in reconstruction.image(iid).cam_from_world().rotation.quat]}
                 for iid in sorted(reconstruction.reg_image_ids())]
    return {"success": bool(ok), "rotations": rotations}


def run_pipeline(database_path, work):
    options = pycolmap.GlobalPipelineOptions()
    options.num_threads = 1
    options.random_seed = SEED
    reconstructions = pycolmap.global_mapping(database_path, work, work / "sparse", options)
    result = []
    for idx in sorted(reconstructions):
        rec = reconstructions[idx]
        result.append({"reg_image_ids": sorted(rec.reg_image_ids()),
                       "num_points3D": rec.num_points3D(),
                       "cam_from_world": [{"image_id": iid, **rigid(rec.image(iid).cam_from_world())}
                                          for iid in sorted(rec.reg_image_ids())]})
    return result


def main():
    if platform.system() != "Darwin":
        sys.exit("Run on macOS: the fixture must come from the libc++ pycolmap wheel.")
    cases = []
    for name, seed, overrides, noise_overrides, num_corrupted in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            database_path = work / "database.db"
            with pycolmap.Database.open(database_path) as database:
                pycolmap.set_random_seed(seed)
                options = pycolmap.SyntheticDatasetOptions()
                for key, value in overrides.items():
                    setattr(options, key, value)
                gt = pycolmap.synthesize_dataset(options, database)
                if noise_overrides is not None:
                    noise = pycolmap.SyntheticNoiseOptions()
                    for key, value in noise_overrides.items():
                        setattr(noise, key, value)
                    pycolmap.synthesize_noise(noise, gt, database)
                corrupted = corrupt_pairs(database, num_corrupted, np.random.default_rng(seed))
                case = {"name": name, "seed": seed, "options": overrides,
                        "noise_options": noise_overrides, "corrupted_pairs": corrupted,
                        "database": dump_database(database),
                        "gt_cam_from_world": [{"image_id": iid, **rigid(gt.image(iid).cam_from_world())}
                                              for iid in sorted(gt.reg_image_ids())]}
                case["rotation_averaging"] = run_rotation_averaging(database)
            case["pipeline"] = run_pipeline(database_path, work)
            cases.append(case)
            summary = [(len(r["reg_image_ids"]), r["num_points3D"]) for r in case["pipeline"]]
            print(name, "RA", case["rotation_averaging"]["success"], "pipeline", summary)
    OUTPUT.write_text(json.dumps({"pycolmap": pycolmap.__version__, "random_seed": SEED,
                                  "cases": cases}, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
