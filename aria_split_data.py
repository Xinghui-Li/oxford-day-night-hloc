import cv2
import json
import numpy as np
import argparse
import random

from aria_constant import ARIA_DATASET_ROOT, OUTPUT_ROOT, RANDOM_SEED, logger


def spatial_subsample(image_info, min_dist, min_rot):

    filtered_image_info =[]
    occupied_trans = None
    occupied_ori = None

    for info in image_info:
        c2w = np.linalg.inv(np.array(info["transform_matrix"]))
        translation = c2w[:3, 3][np.newaxis, :]
        orientation = c2w[:3, :3][np.newaxis, :, :]
        
        if occupied_trans is None:
            occupied_trans = translation
            occupied_ori = orientation
            image_info.append(info)
        else:
            dist = np.linalg.norm(occupied_trans - translation, axis=1)
            mask = dist < min_dist
            if np.sum(mask) == 0:
                occupied_trans = np.vstack((occupied_trans, translation))
                occupied_ori = np.vstack((occupied_ori, orientation))
                filtered_image_info.append(info)
            else:
                existing_orientations = occupied_ori[mask]

                for existing_orientation in existing_orientations:
                    new_orientation = orientation[0]
                    r_err = np.matmul(new_orientation, existing_orientation.T)
                    r_err = cv2.Rodrigues(r_err)[0]
                    r_err = np.linalg.norm(r_err) * 180 / np.pi

                    if r_err < min_rot:
                        skip = True
                        break
                    else:
                        skip = False

                if not skip:
                    occupied_trans = np.vstack((occupied_trans, translation))
                    occupied_ori = np.vstack((occupied_ori, orientation))
                    filtered_image_info.append(info)

    return filtered_image_info


def write_imagelist(imagelist, image_info, with_intrinsics=False):
    """
    Write the imagelist to a file.
    """
    if imagelist.exists():
        raise ValueError(f"{imagelist} already exists. Please remove it first.")
    
    with open(str(imagelist), "w") as f:
        for info in image_info:
            if with_intrinsics:
                f.write(f"{info['file_path']} SIMPLE_PINHOLE {info['w']} {info['h']} {info['fl_x']} {info['cx']} {info['cy']}\n")
            else:
                f.write(f"{info['file_path']}\n")

    logger.info(f"Write {imagelist} with {len(image_info)} images.")


def split_image_info_to_day_night(image_info, day_headers, night_headers):
    """
    Split the image info into day and night.
    """
    day_image_info = []
    night_image_info = []

    for info in image_info:
        if info["file_path"].split("/")[1].split("_")[0] in day_headers:
            day_image_info.append(info)
        elif info["file_path"].split("/")[1].split("_")[0] in night_headers:
            night_image_info.append(info)

    return day_image_info, night_image_info


def get_day_night_split_for_scene(scene):
    vrs2slam_file = [p for p in (ARIA_DATASET_ROOT / scene / "mps" / "multi").glob("day_night_*") \
                if p.is_dir()][0] / "vrs_to_multi_slam.json"
    vrs2slam_original = json.load(open(str(vrs2slam_file), "r"))
    vrs2slam = {}
    for k, v in vrs2slam_original.items():
        vrs2slam[k.split("/")[-1]] = v

    day_vrs = [p for p in (ARIA_DATASET_ROOT / scene / "mps" / "multi").glob("day_*.txt") if "day_night" not in p.name][0]
    night_vrs = [p for p in (ARIA_DATASET_ROOT / scene / "mps" / "multi").glob("night_*.txt") if "day_night" not in p.name][0]
    day_vrs = open(str(day_vrs), "r").readlines()
    night_vrs = open(str(night_vrs), "r").readlines()
    day_slam = [vrs2slam[p.strip()] for p in day_vrs]
    night_slam = [vrs2slam[p.strip()] for p in night_vrs]
    day_headers = {f"vrs{int(p):02d}" for p in day_slam}
    night_headers = {f"vrs{int(p):02d}" for p in night_slam}

    return day_headers, night_headers
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split Oxford Day-Night dataset into database, day query and night query sets.")
    parser.add_argument("--scene", type=str, required=True, 
                        choices=["keble-college", "hb-allen-centre", "observatory-quarter", "oxford-robotics-institute", "bodleian-library"],
                        default="hb-allen-centre",
                        help="The name of the scene to process."
                    )
    parser.add_argument("--min_dist", type=float, default=1.5, help="The grid size to use for spatial subsampling.")
    parser.add_argument("--min_rot", type=float, default=20.0, help="The minimum rotation to consider a point as a new point.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="The random seed to use.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # load the image list
    image_root = [p for p in (ARIA_DATASET_ROOT / args.scene / "ns_processed" / "multi" / "undistorted_all_valid").glob("day_night_*") \
                  if p.is_dir()][0]
    all_image_info = json.load(open(str(image_root / "transforms_opencv.json"), "r"))["frames"]

    # load day-night split information
    day_headers, night_headers = get_day_night_split_for_scene(args.scene)
    # split image info into day and night
    day_image_info, night_image_info = split_image_info_to_day_night(all_image_info, day_headers, night_headers)

    random.shuffle(day_image_info)
    filterd_day_image_info = spatial_subsample(day_image_info, args.min_dist, args.min_rot)
    random.shuffle(filterd_day_image_info)
    num_db = int(len(filterd_day_image_info) * 0.66)
    db_image_info = filterd_day_image_info[:num_db]
    db_image_info = sorted(db_image_info, key=lambda x: x["file_path"])
    query_day_image_info = filterd_day_image_info[num_db:]
    query_day_image_info = sorted(query_day_image_info, key=lambda x: x["file_path"])

    random.shuffle(night_image_info)
    filtered_night_image_info = spatial_subsample(night_image_info, args.min_dist, args.min_rot)
    query_night_image_info = filtered_night_image_info
    query_night_image_info = sorted(query_night_image_info, key=lambda x: x["file_path"])

    logger.info(f"Create database set with {len(db_image_info)} images from {len(day_image_info)} day images")
    logger.info(f"Create day query set with {len(query_day_image_info)} images from {len(day_image_info)} day images")
    logger.info(f"Create night query set with {len(query_night_image_info)} images from {len(night_image_info)} night images")

    # create the output directory
    output_dir = OUTPUT_ROOT / args.scene / "imagelists"
    output_dir.mkdir(parents=True, exist_ok=True)
    # create a dictionary that map file path to image information
    filepath2info = {}
    for info in all_image_info:
        filepath2info[info["file_path"]] = info

    # create all imagelists
    db_imagelist = output_dir / "db_imagelist.txt"
    db_imagelist_with_intrinsics = output_dir / "db_imagelist_with_intrinsics.txt"
    query_day_imagelist = output_dir / "query_day_imagelist.txt"
    query_day_imagelist_with_intrinsics = output_dir / "query_day_imagelist_with_intrinsics.txt"
    query_night_imagelist = output_dir / "query_night_imagelist.txt"
    query_night_imagelist_with_intrinsics = output_dir / "query_night_imagelist_with_intrinsics.txt"

    # write the imagelists
    write_imagelist(db_imagelist, db_image_info)
    write_imagelist(db_imagelist_with_intrinsics, db_image_info, with_intrinsics=True)
    write_imagelist(query_day_imagelist, query_day_image_info)
    write_imagelist(query_day_imagelist_with_intrinsics, query_day_image_info, with_intrinsics=True)
    write_imagelist(query_night_imagelist, query_night_image_info)
    write_imagelist(query_night_imagelist_with_intrinsics, query_night_image_info, with_intrinsics=True)
