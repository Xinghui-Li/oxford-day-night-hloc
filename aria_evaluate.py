import cv2
import json
import argparse
import numpy as np

from scipy.spatial.transform import Rotation as R
from aria_constant import ARIA_DATASET_ROOT, OUTPUT_ROOT


def se3_from_quaternion_translation(qvec, tvec):
    """
    Converts a quaternion + translation to a 4x4 SE(3) matrix.
    
    Parameters:
        qvec: array-like, shape (4,) [qw, qx, qy, qz] (COLMAP style)
        tvec: array-like, shape (3,) [tx, ty, tz]

    Returns:
        T: np.ndarray, shape (4, 4) SE(3) transformation matrix
    """
    qvec = np.asarray(qvec)
    tvec = np.asarray(tvec)
    
    # Reorder from [w, x, y, z] → [x, y, z, w] for scipy
    quat_xyzw = qvec[[1, 2, 3, 0]]
    
    # Convert to rotation matrix
    R_mat = R.from_quat(quat_xyzw).as_matrix()
    
    # Compose SE(3)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec
    return T


def compare_transforms(pred_matrix, gt_matrix):
    """
    Compare two SE(3) matrices and print the difference.
    
    Parameters:
        pred_matrix: np.ndarray, shape (4, 4) predicted SE(3) matrix
        gt_matrix: np.ndarray, shape (4, 4) ground truth SE(3) matrix
    """
    pred_R = pred_matrix[:3, :3]
    gt_R = gt_matrix[:3, :3]
    pred_t = pred_matrix[:3, 3]
    gt_t = gt_matrix[:3, 3]
    
    r_err = np.matmul(pred_R, np.linalg.inv(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / np.pi

    t_err = np.linalg.norm(pred_t - gt_t)

    return r_err, t_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, 
                    choices=["keble-college", "hb-allen-centre", "observatory-quarter", "oxford-robotics-institute", "bodleian-library"],
                    default="hb-allen-centre",
                    help="The name of the scene to process."
                )
    args = parser.parse_args()

    scene_dir = [p for p in (ARIA_DATASET_ROOT / args.scene / "ns_processed" / "multi" / "undistorted_all_valid").glob("day_night_*")][0]
    output_dir = OUTPUT_ROOT / args.scene
    # Evaluate all prediction in the scene output folder
    for name in output_dir.iterdir():
        if name.is_file() and name.name.startswith("hloc_") and name.name.endswith(".txt"):
            pred_file = name

            # load ground truth
            gt_file = scene_dir / "transforms_opencv.json"
            images_info = json.load(open(str(gt_file), "r"))
            name2transforms = {}
            for transform in images_info["frames"]:
                name2transforms[transform["file_path"]] = transform

            # load prediction
            with open(str(pred_file), "r") as f:
                pred_lines = f.readlines()

            r_errs, t_errs = [], []
            eval_thresholds = [(0.25, 2), (0.5, 5), (1, 10)] # (translation, rotation)
            for pred_line in pred_lines:
                pred_line = pred_line.strip().split(" ")
                if len(pred_line) == 0:
                    continue
                
                imgname = "camera-rgb/" + pred_line[0]
                pred_qvec = np.array([float(pred_line[1]), float(pred_line[2]), float(pred_line[3]), float(pred_line[4])])
                pred_tvec = np.array([float(pred_line[5]), float(pred_line[6]), float(pred_line[7])])
                pred_matrix = se3_from_quaternion_translation(pred_qvec, pred_tvec)

                gt_matrix = np.array(name2transforms[imgname]["transform_matrix"])
                
                r_err, t_err = compare_transforms(np.linalg.inv(pred_matrix), np.linalg.inv(gt_matrix))
                r_errs.append(r_err)
                t_errs.append(t_err)

            r_errs = np.array(r_errs)
            t_errs = np.array(t_errs)

            ratios = []
            print(pred_file.name)
            for t_thres, r_thres in eval_thresholds:
                t_err_mask = t_errs < t_thres
                r_err_mask = r_errs < r_thres
                mask = np.logical_and(t_err_mask, r_err_mask)
                print(f"t: {t_thres}m, r: {r_thres}\u00B0, num: {np.sum(mask)} / {len(mask)}, ratio: {np.sum(mask) / len(mask) * 100:.2f}%")
                ratios.append(np.sum(mask) / len(mask) * 100)
            print(" / ".join([f"{ratio:.2f}" for ratio in ratios]))
                

