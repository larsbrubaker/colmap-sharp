#!/usr/bin/env python3
# bundle_adjustment.py: writes ColmapSharp.Tests/TestData/oracle/bundle_adjustment/, the Tier C
# (outcome) oracle for ColmapSharp.Estimators' DefaultBundleAdjuster. Read by
# ColmapSharp.Tests/Estimators/BundleAdjustmentOracleTests.cs.
#
# A synthetic scene (seeded) with COLMAP's "Nominal" noise (0.5 px 2D, 0.1 3D, 0.5 deg /
# 0.1 pose), written as the noisy input model in input/ (binary), then solved by pycolmap's
# Ceres bundle adjuster with every registered image and the TWO_CAMS_FROM_WORLD gauge, and
# written as the solved model in solved/. manifest.json records the solver summary (initial and
# final cost, iteration count, residual and effective parameter counts).
#
# The C# test reads input/, runs the same bundle adjustment, and compares against the manifest
# and solved/. The gauge fixes the same frames on both sides (std::set order of image ids), so
# poses and points compare directly without a Sim3 alignment.
#
# Usage: oracle/.venv/bin/python oracle/bundle_adjustment.py

import json
import pathlib
import shutil

import pycolmap

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "ColmapSharp.Tests" / "TestData" / "oracle" / "bundle_adjustment"


def main():
    pycolmap.set_random_seed(3)
    options = pycolmap.SyntheticDatasetOptions(
        num_rigs=2,
        num_cameras_per_rig=1,
        num_frames_per_rig=8,
        num_points3D=300,
    )
    reconstruction = pycolmap.synthesize_dataset(options)
    noise = pycolmap.SyntheticNoiseOptions(
        point2D_stddev=0.5,
        point3D_stddev=0.1,
        rig_from_world_rotation_stddev=0.5,
        rig_from_world_translation_stddev=0.1,
    )
    pycolmap.synthesize_noise(noise, reconstruction)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "input").mkdir(parents=True)
    (OUTPUT / "solved").mkdir(parents=True)
    reconstruction.write_binary(str(OUTPUT / "input"))

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.print_summary = False
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(ba_options, config, reconstruction)
    summary = adjuster.solve().ceres_summary
    reconstruction.write_binary(str(OUTPUT / "solved"))

    manifest = {
        "pycolmap_version": pycolmap.__version__,
        "termination": str(summary.termination_type),
        "initial_cost": float(summary.initial_cost),
        "final_cost": float(summary.final_cost),
        "num_iterations": int(summary.num_successful_steps + summary.num_unsuccessful_steps),
        "num_residuals_reduced": int(summary.num_residuals_reduced),
        "num_effective_parameters_reduced": int(summary.num_effective_parameters_reduced),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
