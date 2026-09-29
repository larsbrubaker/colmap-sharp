#!/usr/bin/env python3
# fixture_global_sfm.py: writes ColmapSharp.Tests/TestData/oracle/global_sfm.json, the Tier C
# (outcome) pycolmap 4.2.0 oracle for global SfM. Read by
# ColmapSharp.Tests/Controllers/GlobalSfmOracleTests.cs. Each case runs the steps listed in its
# "steps", in this order:
# - calibrate: pycolmap.calibrate_view_graph (ColmapSharp/Estimators/ViewGraphCalibration.cs),
#   recording every camera and two-view geometry afterwards;
# - rotation_averaging: pycolmap.run_rotation_averaging (Estimators/RotationAveraging.cs),
#   recording the cam_from_world rotation of every registered image;
# - gravity: pycolmap.run_gravity_refinement (Estimators/GravityRefinement.cs). pycolmap binds
#   its pose_priors argument as a std::vector<PosePrior>& converted from a Python list, so the
#   refined gravities never reach Python; the step records the counts COLMAP logs instead
#   ("Number of error prone frames", "Number of refined gravities"), read from stderr;
# - pipeline: pycolmap.global_mapping (Controllers/GlobalPipeline.cs: rotation averaging, track
#   establishment, global positioning, bundle adjustment, retriangulation), recording every
#   reconstruction (registered image ids, cam_from_world per image, number of 3D points).
#
# Each case synthesizes a seeded dataset into a database (pycolmap.synthesize_dataset), then
# optionally adds noise and perturbs it (relative rotations, focal lengths, focal priors, F
# matrices, gravity priors) with numpy's seeded generator. The fixture records the whole input
# database the steps read (cameras, rigs, frames, images, keypoints, two-view geometries, pose
# priors, and the raw matches where view graph calibration re-estimates relative poses from
# them), so the C# test runs on exactly the same input without re-synthesizing it (synthetic
# noise visits images in hash order, divergence 31). Every step is
# single-threaded with a fixed random seed. Doubles are written with repr() (exact); the
# float32 keypoints with numpy's shortest float32 repr (exact in C# float.Parse).
#
# Must run on macOS (the libc++ pycolmap wheel, as the C# PRNG matches libc++).
#
# Usage: oracle/.venv/bin/python oracle/fixture_global_sfm.py

import json
import os
import pathlib
import platform
import re
import sys
import tempfile

import numpy as np
import pycolmap

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle" / "global_sfm.json"

SEED = 1

PIPELINE_STEPS = ["rotation_averaging", "pipeline"]

CASES = [
    {"name": "noise_free", "seed": 5, "steps": PIPELINE_STEPS,
     "options": {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 5, "num_points3D": 80,
                 "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True}},
    # Outlier matches, 2D noise, and the relative rotations of four pairs with no image in
    # common replaced by random ones.
    {"name": "noisy_outliers", "seed": 6, "steps": PIPELINE_STEPS,
     "options": {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 5, "num_points3D": 80,
                 "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True,
                 "inlier_match_ratio": 0.8},
     "noise": {"point2D_stddev": 0.5}, "corrupt_pairs": 4},
    {"name": "two_cameras_per_rig", "seed": 7, "steps": PIPELINE_STEPS,
     "options": {"num_rigs": 2, "num_cameras_per_rig": 2, "num_frames_per_rig": 3, "num_points3D": 80,
                 "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True}},
    # No relative poses in the database: the pipeline decomposes them from E (pairs between
    # cameras with a focal prior) and F (pairs involving camera 2, whose prior is dropped).
    # Rotation averaging alone has no edges and fails.
    {"name": "no_relative_pose", "seed": 8, "steps": PIPELINE_STEPS,
     "options": {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 5, "num_points3D": 60,
                 "camera_has_prior_focal_length": True},
     "noise": {"point2D_stddev": 0.5}, "drop_focal_priors": [2]},
    # Unknown focal lengths (only F is meaningful), view_graph_calibration_test.cc's Nominal
    # with the focal of cameras 1 and 2 perturbed, plus 2D noise so re-estimating the relative
    # poses from the matches (on here) has something to do; then the pipeline on the
    # calibrated database. The synthetic cameras sit on a sphere looking at its center, so
    # their optical axes meet in one point, the critical configuration for focal lengths from
    # F (Sturm): the unperturbed cameras anchor the solution.
    {"name": "uncalibrated", "seed": 9, "steps": ["calibrate", "pipeline"], "record_matches": True,
     "options": {"num_rigs": 8, "num_cameras_per_rig": 1, "num_frames_per_rig": 1, "num_points3D": 50,
                 "camera_model_id": "SIMPLE_PINHOLE", "camera_params": [1280.0, 512.0, 384.0],
                 "camera_has_prior_focal_length": False},
     "noise": {"point2D_stddev": 0.5}, "perturb_focal_cameras": [1, 2], "focal_perturbation": 50.0},
    # view_graph_calibration_test.cc's ConfigTagging: two F matrices broken, a tight
    # calibration error bound, no re-estimation; they are tagged DEGENERATE.
    {"name": "calibration_config_tagging", "seed": 11, "steps": ["calibrate"],
     "calibration_options": {"reestimate_relative_pose": False, "max_calibration_error": 0.01},
     "options": {"num_rigs": 6, "num_cameras_per_rig": 1, "num_frames_per_rig": 1, "num_points3D": 40,
                 "camera_model_id": "SIMPLE_PINHOLE", "camera_params": [1280.0, 512.0, 384.0],
                 "camera_has_prior_focal_length": False},
     "perturb_F_pairs": 2},
    # Gravity priors, 30% replaced by random directions (gravity_refinement_test.cc's setup).
    {"name": "prior_gravity", "seed": 10, "steps": ["gravity", "pipeline"],
     "options": {"num_rigs": 2, "num_cameras_per_rig": 1, "num_frames_per_rig": 6, "num_points3D": 40,
                 "camera_has_prior_focal_length": True, "two_view_geometry_has_relative_pose": True,
                 "prior_gravity": True},
     "gravity_outlier_ratio": 0.3},
]


def rigid(t):
    return {"q_xyzw": [float(v) for v in t.rotation.quat], "t": [float(v) for v in t.translation]}


def f32(v):
    return float(np.format_float_positional(np.float32(v), unique=True, trim="-"))


def matrix(m):
    return None if m is None else [float(v) for v in np.asarray(m).reshape(-1)]


def vector_or_none(v):
    v = np.asarray(v)
    return [float(x) for x in v] if np.all(np.isfinite(v)) else None


def sorted_pairs(database):
    pair_ids, geometries = database.read_two_view_geometries()
    return [(pycolmap.pair_id_to_image_pair(pair_id), tvg)
            for pair_id, tvg in sorted(zip(pair_ids, geometries), key=lambda p: p[0])]


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return pycolmap.Rotation3d(np.array([q[0], q[1], q[2], q[3]]))  # (x, y, z, w)


def disjoint_pairs(database, count):
    """The first `count` pairs in pair id order whose images belong to different frames and
    appear in no earlier chosen pair, so the perturbations spread over 2 * count images."""
    images = {im.image_id: im for im in database.read_all_images()}
    used = set()
    chosen = []
    for (id1, id2), tvg in sorted_pairs(database):
        if len(chosen) == count:
            break
        if id1 in used or id2 in used or images[id1].frame_id == images[id2].frame_id:
            continue
        used.update((id1, id2))
        chosen.append((id1, id2, tvg))
    return chosen


def corrupt_pairs(database, count, rng):
    """Replaces the relative rotation of `count` disjoint pairs with a random rotation,
    keeping the translation."""
    corrupted = []
    for id1, id2, tvg in disjoint_pairs(database, count):
        tvg.cam2_from_cam1 = pycolmap.Rigid3d(random_rotation(rng), tvg.cam2_from_cam1.translation)
        database.update_two_view_geometry(id1, id2, tvg)
        corrupted.append([id1, id2])
    return corrupted


def perturb_F(database, count):
    """Adds 1 to every entry of F for `count` disjoint pairs (view_graph_calibration_test.cc
    ConfigTagging), which calibration should tag DEGENERATE."""
    perturbed = []
    for id1, id2, tvg in disjoint_pairs(database, count):
        tvg.F = tvg.F + np.ones((3, 3))
        database.update_two_view_geometry(id1, id2, tvg)
        perturbed.append([id1, id2])
    return perturbed


def perturb_focals(database, camera_ids, amplitude, rng):
    """Adds a uniform draw from [-amplitude, amplitude] to the given cameras' focal length."""
    for camera in sorted(database.read_all_cameras(), key=lambda c: c.camera_id):
        if camera.camera_id not in camera_ids:
            continue
        params = np.array(camera.params)
        params[list(camera.focal_length_idxs())] += rng.uniform(-amplitude, amplitude)
        camera.params = params
        database.update_camera(camera)


def drop_focal_priors(database, camera_ids):
    """Clears the focal prior of the given cameras and tags every pair involving them
    UNCALIBRATED (as synthesis does for such cameras), so the pipeline decomposes those
    pairs' relative poses from F rather than E."""
    for camera in database.read_all_cameras():
        if camera.camera_id in camera_ids:
            camera.has_prior_focal_length = False
            database.update_camera(camera)
    images = {im.image_id: im for im in database.read_all_images()}
    for (id1, id2), tvg in sorted_pairs(database):
        if images[id1].camera_id in camera_ids or images[id2].camera_id in camera_ids:
            tvg.config = pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED
            database.update_two_view_geometry(id1, id2, tvg)


def corrupt_gravity(database, outlier_ratio, rng):
    """Replaces each prior's gravity with a random direction with probability outlier_ratio.
    Returns the ids of the replaced priors."""
    replaced = []
    for prior in sorted(database.read_all_pose_priors(), key=lambda p: p.pose_prior_id):
        if rng.uniform() < outlier_ratio:
            g = rng.normal(size=3)
            prior.gravity = g / np.linalg.norm(g)
            database.update_pose_prior(prior)
            replaced.append(prior.pose_prior_id)
    return replaced


def dump_cameras(database):
    return [{"camera_id": c.camera_id, "model_id": int(c.model.value), "width": c.width,
             "height": c.height, "params": [float(p) for p in c.params],
             "has_prior_focal_length": bool(c.has_prior_focal_length)}
            for c in sorted(database.read_all_cameras(), key=lambda c: c.camera_id)]


def dump_pair(id1, id2, tvg, with_matches):
    assert tvg.camera1 is None and tvg.camera2 is None
    pair = {"image_id1": id1, "image_id2": id2, "config": int(tvg.config),
            "E": matrix(tvg.E), "F": matrix(tvg.F), "H": matrix(tvg.H),
            "cam2_from_cam1": None if tvg.cam2_from_cam1 is None else rigid(tvg.cam2_from_cam1)}
    if with_matches:
        pair["inlier_matches"] = [int(v) for v in np.asarray(tvg.inlier_matches).reshape(-1)]
    else:
        pair["num_inliers"] = len(tvg.inlier_matches)
    return pair


def dump_database(database, record_matches):
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
    for (id1, id2), tvg in sorted_pairs(database):
        pair = dump_pair(id1, id2, tvg, with_matches=True)
        if record_matches:
            pair["matches"] = [int(v) for v in np.asarray(database.read_matches(id1, id2)).reshape(-1)]
        pairs.append(pair)
    pose_priors = []
    for prior in sorted(database.read_all_pose_priors(), key=lambda p: p.pose_prior_id):
        assert vector_or_none(prior.position) is None, "position priors are not recorded"
        pose_priors.append({"pose_prior_id": prior.pose_prior_id,
                            "corr_image_id": prior.corr_data_id.id,
                            "corr_camera_id": prior.corr_data_id.sensor_id.id,
                            "coordinate_system": int(prior.coordinate_system.value),
                            "gravity": vector_or_none(prior.gravity)})
    return {"cameras": dump_cameras(database), "rigs": rigs, "frames": frames, "images": images,
            "pairs": pairs, "pose_priors": pose_priors}


def load_reconstruction_and_pose_graph(database):
    cache = pycolmap.DatabaseCache.create(database, pycolmap.DatabaseCacheOptions())
    reconstruction = pycolmap.Reconstruction()
    reconstruction.load(cache)
    pose_graph = pycolmap.PoseGraph()
    pose_graph.load(cache.correspondence_graph)
    return reconstruction, pose_graph


def run_calibration(database_path, overrides):
    options = pycolmap.ViewGraphCalibrationOptions()
    options.random_seed = SEED
    for key, value in overrides.items():
        setattr(options, key, value)
    ok = pycolmap.calibrate_view_graph(database_path, options)
    with pycolmap.Database.open(database_path) as database:
        return {"success": bool(ok), "cameras": dump_cameras(database),
                "pairs": [dump_pair(id1, id2, tvg, with_matches=False)
                          for (id1, id2), tvg in sorted_pairs(database)]}


def run_rotation_averaging(database):
    reconstruction, pose_graph = load_reconstruction_and_pose_graph(database)
    options = pycolmap.RotationEstimatorOptions()
    options.random_seed = SEED
    ok = pycolmap.run_rotation_averaging(options, pose_graph, reconstruction,
                                         database.read_all_pose_priors())
    rotations = [{"image_id": iid, "q_xyzw": [float(v) for v in reconstruction.image(iid).cam_from_world().rotation.quat]}
                 for iid in sorted(reconstruction.reg_image_ids())]
    return {"success": bool(ok), "rotations": rotations}


def capture_stderr(action):
    """Runs action() with file descriptor 2 redirected to a temporary file (glog writes there
    directly) and returns what was written."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as capture:
        os.dup2(capture.fileno(), 2)
        try:
            action()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        capture.seek(0)
        return capture.read().decode()


def run_gravity_refinement(database):
    reconstruction, pose_graph = load_reconstruction_and_pose_graph(database)
    options = pycolmap.GravityRefinerOptions()
    log = capture_stderr(lambda: pycolmap.run_gravity_refinement(
        options, pose_graph, reconstruction, database.read_all_pose_priors()))
    error_prone = re.search(r"Number of error prone frames: (\d+)", log)
    refined = re.search(r"Number of refined gravities: (\d+) / (\d+)", log)
    assert error_prone and refined, log
    return {"num_error_prone_frames": int(error_prone.group(1)),
            "num_refined_gravities": int(refined.group(1))}


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


def synthesize(spec, database):
    """Synthesizes and perturbs the case's input database; returns the ground truth and a
    record of the perturbations."""
    rng = np.random.default_rng(spec["seed"])
    pycolmap.set_random_seed(spec["seed"])
    options = pycolmap.SyntheticDatasetOptions()
    for key, value in spec["options"].items():
        setattr(options, key, getattr(pycolmap.CameraModelId, value) if key == "camera_model_id" else value)
    gt = pycolmap.synthesize_dataset(options, database)
    if "noise" in spec:
        noise = pycolmap.SyntheticNoiseOptions()
        for key, value in spec["noise"].items():
            setattr(noise, key, value)
        pycolmap.synthesize_noise(noise, gt, database)
    perturbations = {}
    if "corrupt_pairs" in spec:
        perturbations["corrupted_pairs"] = corrupt_pairs(database, spec["corrupt_pairs"], rng)
    if "drop_focal_priors" in spec:
        drop_focal_priors(database, spec["drop_focal_priors"])
    if "focal_perturbation" in spec:
        perturb_focals(database, spec["perturb_focal_cameras"], spec["focal_perturbation"], rng)
    if "perturb_F_pairs" in spec:
        perturbations["perturbed_F_pairs"] = perturb_F(database, spec["perturb_F_pairs"])
    if "gravity_outlier_ratio" in spec:
        perturbations["gravity_outlier_prior_ids"] = corrupt_gravity(database, spec["gravity_outlier_ratio"], rng)
    return gt, perturbations


def main():
    if platform.system() != "Darwin":
        sys.exit("Run on macOS: the fixture must come from the libc++ pycolmap wheel.")
    cases = []
    for spec in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            database_path = work / "database.db"
            with pycolmap.Database.open(database_path) as database:
                gt, perturbations = synthesize(spec, database)
                case = {"name": spec["name"], "seed": spec["seed"], "steps": spec["steps"],
                        "options": spec["options"], "noise_options": spec.get("noise"),
                        **perturbations,
                        "database": dump_database(database, spec.get("record_matches", False)),
                        "gt_cam_from_world": [{"image_id": iid, **rigid(gt.image(iid).cam_from_world())}
                                              for iid in sorted(gt.reg_image_ids())],
                        "gt_focal_lengths": {str(cid): float(c.mean_focal_length())
                                             for cid, c in sorted(gt.cameras.items())}}
            if "calibrate" in spec["steps"]:
                case["calibration_options"] = spec.get("calibration_options", {})
                case["calibration"] = run_calibration(database_path, case["calibration_options"])
            with pycolmap.Database.open(database_path) as database:
                if "rotation_averaging" in spec["steps"]:
                    case["rotation_averaging"] = run_rotation_averaging(database)
                if "gravity" in spec["steps"]:
                    case["gravity"] = run_gravity_refinement(database)
            if "pipeline" in spec["steps"]:
                case["pipeline"] = run_pipeline(database_path, work)
            cases.append(case)
            summary = [(len(r["reg_image_ids"]), r["num_points3D"]) for r in case.get("pipeline", [])]
            print(spec["name"], {k: case[k].get("success", case[k]) for k in
                                 ("calibration", "rotation_averaging", "gravity") if k in case},
                  "pipeline", summary)
    OUTPUT.write_text(json.dumps({"pycolmap": pycolmap.__version__, "random_seed": SEED,
                                  "cases": cases}, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
