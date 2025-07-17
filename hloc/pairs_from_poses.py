import argparse
from pathlib import Path

import json
import numpy as np
import scipy.spatial

from . import logger
from scipy.spatial.transform import Rotation as R
from aria_constant import ARIA_DATASET_ROOT
from .pairs_from_retrieval import pairs_from_score_matrix
from .utils.read_write_model import read_images_binary

DEFAULT_ROT_THRESH = 30  # in degrees
ARIA_ROT_THRESH = 50


def get_pairwise_distances(images):
    ids = np.array(list(images.keys()))
    Rs = []
    ts = []
    for id_ in ids:
        image = images[id_]
        R = image.qvec2rotmat()
        t = image.tvec
        Rs.append(R)
        ts.append(t)
    Rs = np.stack(Rs, 0)
    ts = np.stack(ts, 0)

    # Invert the poses from world-to-camera to camera-to-world.
    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ts))

    # Instead of computing the angle between two camera orientations,
    # we compute the angle between the principal axes, as two images rotated
    # around their principal axis still observe the same scene.
    axes = Rs[:, :, -1]
    dots = np.einsum("mi,ni->mn", axes, axes, optimize=True)
    dR = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))

    return ids, dist, dR


def main(model, output, num_matched, rotation_threshold=DEFAULT_ROT_THRESH):
    logger.info("Reading the COLMAP model...")
    images = read_images_binary(model / "images.bin")

    logger.info(f"Obtaining pairwise distances between {len(images)} images...")
    ids, dist, dR = get_pairwise_distances(images)
    scores = -dist

    invalid = dR >= rotation_threshold
    np.fill_diagonal(invalid, True)
    pairs = pairs_from_score_matrix(scores, invalid, num_matched)
    pairs = [(images[ids[i]].name, images[ids[j]].name) for i, j in pairs]

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join(p) for p in pairs))


def get_dist_to_db(query_image_info, db_c2w):

    dists = []
    oris = []

    for info in query_image_info:
        c2w = np.linalg.inv(np.array(info["transform_matrix"]))
        translation = c2w[:3, 3][np.newaxis, :]
        orientation = c2w[:3, :3][np.newaxis, :, :]
        
        db_trans = db_c2w[:, :3, 3]
        t_errs = np.linalg.norm(db_trans - translation, axis=1)

        db_ori = db_c2w[:, :3, :3]
        inv_orientation = np.linalg.inv(orientation[0])[np.newaxis, :, :]
        r_errs = db_ori @ inv_orientation
        r_errs = R.from_matrix(r_errs).as_rotvec()
        r_errs = np.linalg.norm(r_errs, axis=1) * 180 / np.pi

        dists.append(t_errs)
        oris.append(r_errs)

    return dists, oris 


def aria_pose_main(
    output,
    scene,
    num_matched,
    query_list,
    db_list,
):
    if output.exists():
        logger.info(f"Output file {output} already exists. Skipping...")
        return

    # load the image list
    image_root = [p for p in (ARIA_DATASET_ROOT / scene / "ns_processed" / "multi" / "undistorted_all_valid").glob("day_night_*") \
                    if p.is_dir()][0]
    all_image_info = json.load(open(str(image_root / "transforms_opencv.json"), "r"))["frames"]
    path2info = {}
    for info in all_image_info:
        path2info[info["file_path"]] = info

    db_image_info = []
    db_c2w = []
    with open(str(db_list), "r") as f:
        db_lines = f.readlines()
        for line in db_lines:
            line = line.strip().split(" ")
            if len(line) == 0:
                continue
            transform = np.array(path2info[line[0]]["transform_matrix"])
            db_c2w.append(np.linalg.inv(transform))
            db_image_info.append(path2info[line[0]])

    db_c2w = np.stack(db_c2w, axis=0)

    query_image_info = []
    with open(str(query_list), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            if len(line) == 0:
                continue
            query_image_info.append(path2info[line[0]])

    dists, oris = get_dist_to_db(query_image_info, db_c2w)

    pairs = []
    for i, (dist, ori) in enumerate(zip(dists, oris)):
        dist_closest_indices = np.argsort(dist)[:num_matched]
        ori_closest = ori[dist_closest_indices]
        selected_indices = dist_closest_indices
        for j in selected_indices:
            pairs.append([query_image_info[i]["file_path"], db_image_info[j]["file_path"]])

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join(p) for p in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    parser.add_argument("--rotation_threshold", default=DEFAULT_ROT_THRESH, type=float)
    args = parser.parse_args()
    main(**args.__dict__)
