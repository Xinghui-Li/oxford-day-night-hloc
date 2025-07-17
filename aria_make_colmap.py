import json
import pandas as pd
import numpy as np
import argparse
import pycolmap

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from aria_utils import *
from aria_constant import ARIA_DATASET_ROOT, OUTPUT_ROOT, logger

from projectaria_tools.core.sophus import SE3


def load_points_and_observations(mps_root, vrs_idx):

    logger.info(f"Loading points and observations from {mps_root}/{vrs_idx}")
    
    directory = mps_root / f"{vrs_idx}"
    points_file = directory / "slam" / "semidense_points.csv.gz"
    observations_file = directory / "slam" / "semidense_observations.csv.gz"
    points = pd.read_csv(str(points_file))
    observations = pd.read_csv(str(observations_file))

    output = {
        "points": points,
        "observations": observations
    }

    return output


def get_image_info(image_root):

    logger.info(f"Loading image info from {image_root}")

    image_info = json.load(open(str(image_root / "transforms_opencv.json"), "r"))["frames"]
    filepath2info = {}
    for info in image_info:
        filepath2info[info["file_path"]] = info

    return filepath2info


def load_imagelist(imagelist):

    logger.info(f"Loading image list from {imagelist}")

    with open(str(imagelist), "r") as f:
        lines = f.readlines()

    file_paths = []
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        if len(line) == 0:
            continue
        file_paths.append(line.split(" ")[0])

    return file_paths


def get_slam_serial_numbers():

    vrs_info = pd.read_csv(str(ARIA_DATASET_ROOT / "vrs_info.csv"))
    vrs_info = vrs_info.rename(columns={"vrs_idx": "vrs_idx", "vrs_name": "vrs_name"})
    slam_serial_numbers = defaultdict(dict)
    for i, row in vrs_info.iterrows():
        scene = row["scene_name"]
        vrs_name = row["vrs_name"]
        slam_left_serial = row["slam_left_serial"]
        slam_right_serial = row["slam_right_serial"]

        slam_serial_numbers[scene][vrs_name] = {
            "slam_left_serial": slam_left_serial,
            "slam_right_serial": slam_right_serial
        }

    return slam_serial_numbers


def get_vrs_idx_name_mapper(mps_root):

    file = json.load(open(str(mps_root / "vrs_to_multi_slam.json"), "r"))
    vrs_idx_name_mapper = {}
    for k, v in file.items():
        vrs_idx_name_mapper[k.split("/")[-1]] = int(v)
        vrs_idx_name_mapper[int(v)] = k.split("/")[-1]

    return vrs_idx_name_mapper


def write_colmap_txt(output_dir, idx2cam, idx2image, idx2point3D):
    """
    Write the colmap data to text files.
    """
    # write cameras
    camera_file = output_dir / "cameras.txt"
    with open(str(camera_file), "w") as f:
        f.write(f"# Camera list with one line of data per camera:\n")
        f.write(f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, cam in enumerate(idx2cam.values()):
            if i < len(idx2cam) - 1:
                f.write(f"{cam.id} {cam.type} {cam.width} {cam.height} {cam.fx} {cam.cx} {cam.cy}\n")
            else:
                f.write(f"{cam.id} {cam.type} {cam.width} {cam.height} {cam.fx} {cam.cx} {cam.cy}")

    # write images
    image_file = output_dir / "images.txt"
    with open(str(image_file), "w") as f:
        f.write(f"# Image list with two lines of data per image:\n")
        f.write(f"#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write(f"#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in idx2image.values():
            qvec = ' '.join(f"{q:.6f}" for q in img.qvec)
            tvec = ' '.join(f"{t:.6f}" for t in img.tvec)
            f.write(f"{img.id} {qvec} {tvec} {img.camera_id} {img.name}\n")
            for i, obs in enumerate(img.observations):
                if i < len(img.observations) - 1:
                    f.write(f"{obs[0]:.2f} {obs[1]:.2f} {obs[2]} ")
                else:
                    f.write(f"{obs[0]:.2f} {obs[1]:.2f} {obs[2]}")
            f.write("\n")

    # write points3D
    points3d_file = output_dir / "points3D.txt"
    with open(str(points3d_file), "w") as f:
        f.write(f"# 3D point list with one line of data per point:\n")
        f.write(f"#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pt in idx2point3D.values():
            f.write(f"{pt.id} {' '.join(map(str, pt.xyz))} {' '.join(map(str, pt.rgb))} {pt.error} ")
            for i, track in enumerate(pt.track):
                f.write(f"{track[0]} {track[1]} ")
            else:
                f.write(f"{track[0]} {track[1]}")
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert Oxford Day Night dataset from ARIA to COLMAP format.")
    parser.add_argument("--scene", type=str, required=True, 
                        choices=["keble-college", "hb-allen-centre", "observatory-quarter", "oxford-robotics-institute", "bodleian-library"],
                        default="hb-allen-centre",
                        help="The name of the scene to process."
                    )
    args = parser.parse_args()

    imagelist = Path(OUTPUT_ROOT / args.scene / "imagelists" / "db_imagelist.txt")
    assert imagelist.exists(), f"Image list {imagelist} does not exist"

    image_root = [p for p in (ARIA_DATASET_ROOT / args.scene / "ns_processed" / "multi" / "undistorted_all_valid").glob("day_night_*") \
                  if p.is_dir()][0]
    mps_root = [p for p in (ARIA_DATASET_ROOT / args.scene / "mps" / "multi").glob("day_night_*") \
                if p.is_dir()][0]

    # load all necessary information
    vrs2pointobserv = {}
    image_list = load_imagelist(imagelist)
    filepath2info = get_image_info(image_root)
    slam_serial_numbers = get_slam_serial_numbers()
    vrs_idx_name_mapper = get_vrs_idx_name_mapper(mps_root)

    # container for colmap data
    idx2cam = {}
    idx2image = {}
    idx2point3D = {}

    pbar = tqdm(total=len(image_list), desc="Processing images")
    for img_idx, file_path in enumerate(image_list):
        vrs_idx = int(file_path.split("/")[-1].split("_")[0][3:])
        if vrs_idx not in vrs2pointobserv:
            vrs2pointobserv[vrs_idx] = load_points_and_observations(mps_root, vrs_idx)

        points = vrs2pointobserv[vrs_idx]["points"]
        observations = vrs2pointobserv[vrs_idx]["observations"]

        info = filepath2info[file_path]
        matrix_rgb_world = np.array(info["transform_matrix"])
        rgb_K = build_intrinsics(info)
        rgb_timestamp = info["timestamp"]
        slam_left_serial = slam_serial_numbers[args.scene][vrs_idx_name_mapper[vrs_idx]]["slam_left_serial"]
        slam_right_serial = slam_serial_numbers[args.scene][vrs_idx_name_mapper[vrs_idx]]["slam_right_serial"]

        in_frame_points = get_observed_points(observations, points, slam_left_serial, slam_right_serial, rgb_timestamp)

        uv, in_frame_points = project_in_frame_points(in_frame_points, info, matrix_rgb_world, rgb_K)
        
        # prepare colmap imformation
        camera_model = "SIMPLE_PINHOLE"
        # create ColmapCamera
        cam = ColmapCamera(
            id=1,
            type=camera_model,
            width=info["w"],
            height=info["h"],
            fx=info["fl_x"],
            fy=info["fl_y"],
            cx=info["cx"],
            cy=info["cy"],
        )
        idx2cam[cam.id] = cam

        # create ColmapImage
        SE3_w2c = SE3.from_matrix(matrix_rgb_world)
        quat_and_t = SE3_w2c.to_quat_and_translation()[0]
        qvec = [quat_and_t[0], quat_and_t[1], quat_and_t[2], quat_and_t[3]]
        tvec = [quat_and_t[4], quat_and_t[5], quat_and_t[6]]
        image = ColmapImage(
            id=img_idx+1,
            name=info["file_path"],
            qvec=qvec,
            tvec=tvec,
            camera_id=cam.id,
            observations=[],
        )

        # fill in camera observations and create ColmapPoint3D
        for i in range(uv.shape[1]):
            pt_id = in_frame_points["uid"].iloc[i]
            image.observations.append((uv[0, i], uv[1, i], pt_id))
            
            xyz = in_frame_points.iloc[i][["px_world", "py_world", "pz_world"]].to_list()
            if pt_id not in idx2point3D:
                point3d = ColmapPoint3D(
                    id=pt_id,
                    xyz=xyz,
                    rgb=[128, 128, 128],
                    error=0.0,
                    track=[],
                )
            else:
                point3d = idx2point3D[pt_id]
            point3d.track.append((image.id, i))
            idx2point3D[pt_id] = point3d

        idx2image[image.id] = image

        pbar.update(1)
    
    pbar.close()


    # write to colmap format
    output_dir = OUTPUT_ROOT / args.scene / "colmap"
    output_txt_dir = output_dir / "text"
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    write_colmap_txt(output_txt_dir, idx2cam, idx2image, idx2point3D)

    # generate a binary version
    output_bin_dir = output_dir / "bin"
    output_bin_dir.mkdir(parents=True, exist_ok=True)

    recon = pycolmap.Reconstruction()
    recon.read_text(str(output_txt_dir))
    recon.write_binary(str(output_bin_dir))

