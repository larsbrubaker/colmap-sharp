#!/usr/bin/env python3
# fixture_stereo_fusion.py: writes ColmapSharp.Tests/TestData/oracle/stereo_fusion/ and
# stereo_fusion_options/, two small COLMAP MVS workspaces and pycolmap.stereo_fusion's output
# on each. Read by
# ColmapSharp.Tests/Mvs/FusionOracleTests.cs, which runs ColmapSharp/Mvs/Fusion.cs on the same
# workspace and compares point counts, positions, normals, colors and visibility (Tier C).
#
# The scene: a synthetic 3-camera PINHOLE rig looking at a plane. Each view's depth
# map is the exact ray-plane depth, perturbed by a deterministic ripple of up to 1.5% (so the
# 1% depth check rejects some pixels) and with a block of invalid (zero) depths; normal maps
# are the plane normal in camera coordinates.
#
# The second workspace ("options") stresses the option paths: per-pixel normal noise of a few
# degrees with about one pixel in eight tilted 12-20 degrees (past the 10 degree check), a
# bounding box that cuts the plane, max_image_size = 20 (the workspace halves every map and
# image), max_traversal_depth = 2 and max_num_pixels = 3 (both cut traversals short), and a
# mask PNG per image whose masked block is aligned to 2x2 pixels so the box downscale to
# 20 x 15 is exact. Its options are written to options.json for the C# test. The images are not checked in: they are the
# closed-form pattern of image_color() below, which the C# test regenerates in memory.
#
# Fusion runs with num_threads = 1, the traversal order ColmapSharp always uses
# (docs/CPP_DIVERGENCES.md, entry 87), so the points come out in the same order.
#
# Usage: oracle/.venv/bin/python oracle/fixture_stereo_fusion.py

import json
import pathlib
import shutil
import tempfile

import numpy as np
import pycolmap

ORACLE_DIR = pathlib.Path(__file__).resolve().parent.parent / "ColmapSharp.Tests" / "TestData" / "oracle"

WIDTH = 40
HEIGHT = 30


def image_color(image_idx, x, y):
    """The RGB value of pixel (x, y) of the image at registration index image_idx."""
    return ((5 * x + 3 * y + 17 * image_idx) % 256,
            (11 * x + 7 * y) % 256,
            (x + 13 * y + 29 * image_idx) % 256)


def write_mat(path, array):
    """COLMAP's mvs::Mat .bin format: "w&h&d&" then float32 planes (channel-major)."""
    height, width = array.shape[:2]
    depth = 1 if array.ndim == 2 else array.shape[2]
    planes = array.reshape(height, width, depth).transpose(2, 0, 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(f"{width}&{height}&{depth}&".encode())
        f.write(planes.tobytes())


def synthesize():
    pycolmap.set_random_seed(0)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    # Three cameras of one rig in one frame: nearby views looking the same way, so the
    # plane is seen by all of them (frames of a synthetic rig circle the scene).
    options.num_cameras_per_rig = 3
    options.num_frames_per_rig = 1
    options.sensor_from_rig_translation_stddev = 0.3
    options.num_points3D = 50
    options.camera_width = WIDTH
    options.camera_height = HEIGHT
    options.camera_model_id = pycolmap.CameraModelId.PINHOLE
    options.camera_params = [40.0, 40.0, 20.0, 15.0]
    return pycolmap.synthesize_dataset(options)


def tilt(normal, rng, degrees):
    """normal rotated by `degrees` about a random axis perpendicular to it."""
    axis = np.cross(normal, rng.normal(size=3))
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    return normal * np.cos(angle) + np.cross(axis, normal) * np.sin(angle)


def build(name, noisy):
    reconstruction = synthesize()
    rng = np.random.default_rng(1)
    output = ORACLE_DIR / name
    with tempfile.TemporaryDirectory() as tmp:
        ws = pathlib.Path(tmp)
        for sub in ("sparse", "images", "masks", "stereo/depth_maps", "stereo/normal_maps"):
            (ws / sub).mkdir(parents=True)
        reconstruction.write(ws / "sparse")

        image_ids = reconstruction.reg_image_ids()
        images = [reconstruction.image(i) for i in image_ids]
        centroid = np.mean([p.xyz for p in reconstruction.points3D.values()], axis=0)
        view_dirs = [img.cam_from_world().rotation.matrix().T @ np.array([0.0, 0.0, 1.0]) for img in images]
        plane_normal = -np.mean(view_dirs, axis=0)
        plane_normal /= np.linalg.norm(plane_normal)

        names = []
        for idx, img in enumerate(images):
            names.append(img.name)
            cam = reconstruction.camera(img.camera_id)
            k_inv = np.linalg.inv(cam.calibration_matrix())
            rot = img.cam_from_world().rotation.matrix()
            center = img.projection_center()

            depth = np.zeros((HEIGHT, WIDTH))
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    # MVS pixel coordinates have no +0.5: pixel (x, y) is at (x, y).
                    ray_world = rot.T @ (k_inv @ np.array([x, y, 1.0]))
                    denom = plane_normal @ ray_world
                    s = plane_normal @ (centroid - center) / denom if abs(denom) > 1e-12 else 0.0
                    ripple = 1.0 + 0.015 * np.sin(0.7 * x + 1.3 * y + idx)
                    depth[y, x] = s * ripple if s > 0 else 0.0
            depth[5 + 3 * idx:9 + 3 * idx, 10:16] = 0.0
            write_mat(ws / "stereo" / "depth_maps" / f"{img.name}.geometric.bin", depth)

            normal_cam = rot @ plane_normal
            if normal_cam[2] > 0:
                normal_cam = -normal_cam
            normal = np.empty((HEIGHT, WIDTH, 3))
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if not noisy:
                        normal[y, x] = normal_cam
                    elif rng.random() < 0.125:
                        normal[y, x] = tilt(normal_cam, rng, rng.uniform(12.0, 20.0))
                    else:
                        normal[y, x] = tilt(normal_cam, rng, rng.uniform(0.0, 4.0))
            write_mat(ws / "stereo" / "normal_maps" / f"{img.name}.geometric.bin", normal)

            pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    pixels[y, x] = image_color(idx, x, y)
            pycolmap.Bitmap.from_array(pixels).write(ws / "images" / img.name)

            if noisy:
                mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
                mask[mask_rows(idx), mask_cols(idx)] = 0
                pycolmap.Bitmap.from_array(mask).write(ws / "masks" / (img.name + ".png"))

        (ws / "stereo" / "fusion.cfg").write_text("".join(n + "\n" for n in names))

        fusion_options = pycolmap.StereoFusionOptions()
        fusion_options.num_threads = 1
        fusion_options.min_num_pixels = 3
        fusion_options.check_num_images = 10
        written_options = {}
        if noisy:
            fusion_options.min_num_pixels = 2
            fusion_options.max_num_pixels = 3
            fusion_options.max_traversal_depth = 2
            fusion_options.max_image_size = 20
            fusion_options.mask_path = str(ws / "masks")
            box_min = [float(np.float32(c - 100.0)) for c in centroid]
            box_max = [float(np.float32(c + 100.0)) for c in centroid]
            box_max[0] = float(np.float32(centroid[0]))
            fusion_options.bounding_box = (np.array(box_min, dtype=np.float32), np.array(box_max, dtype=np.float32))
            written_options = {
                "min_num_pixels": 2, "max_num_pixels": 3, "max_traversal_depth": 2,
                "max_image_size": 20, "bounding_box_min": box_min, "bounding_box_max": box_max,
            }
        pycolmap.stereo_fusion(ws / "fused.ply", ws, "COLMAP", "", "geometric", fusion_options, output_type="ply")

        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        shutil.copytree(ws / "sparse", output / "sparse")
        shutil.copytree(ws / "stereo", output / "stereo")
        shutil.copy(ws / "fused.ply", output / "fused.ply")
        shutil.copy(ws / "fused.ply.vis", output / "fused.ply.vis")
        (output / "pycolmap_version.txt").write_text(pycolmap.__version__ + "\n")
        if noisy:
            (output / "options.json").write_text(json.dumps(written_options, indent=1) + "\n")

    print(f"wrote {output}")


def mask_rows(image_idx):
    """Masked rows of the image's mask: 2x2-aligned so a 2x box downscale is exact."""
    return slice(2 * image_idx + 16, 2 * image_idx + 24)


def mask_cols(image_idx):
    """Masked columns of the image's mask (see mask_rows)."""
    return slice(4 * image_idx + 20, 4 * image_idx + 30)


def main():
    build("stereo_fusion", noisy=False)
    build("stereo_fusion_options", noisy=True)


if __name__ == "__main__":
    main()
