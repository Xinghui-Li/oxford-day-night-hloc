import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
from pprint import pformat

from hloc import (
    extract_features,
    match_dense,
    match_features,
    pairs_from_poses,
    pairs_from_retrieval,
)
from hloc import triangulation, localize_sfm
from aria_constant import ARIA_DATASET_ROOT, OUTPUT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, required=True, 
                    choices=["keble-college", "hb-allen-centre", "observatory-quarter", "oxford-robotics-institute", "bodleian-library"],
                    help="The name of the scene to process."
                )
parser.add_argument("--time", type=str, default="day", choices=["day", "night"], help="The time of day to process.")
parser.add_argument("--retrieval", type=str, default="netvlad", choices=["netvlad", "megaloc", "gt_pose", "dir", "openibl"], help="The retrieval method to use.")
parser.add_argument("--feature", type=str, default=None, choices=["superpoint_aachen", "sift", "superpoint_inloc", "disk"], help="The feature extractor to use.")
parser.add_argument("--matcher", type=str, default=None, choices=["superglue", "NN-superpoint", "NN-ratio", "superpoint+lightglue", "disk+lightglue"], help="The feature matcher to use.")
parser.add_argument("--dense_matcher", type=str, default=None, choices=["loftr_aachen", "roma_outdoor", "roma_indoor"], help="The dense matcher to use.")
parser.add_argument("--num_covis", type=int, default=20, help="Number of covisibility matches to use.")
parser.add_argument("--num_loc", type=int, default=50, help="Number of localization matches to use.")
parser.add_argument("--device", type=int, default=0, help="Device to use for feature extraction and matching.")
args = parser.parse_args()

# Set the device for PyTorch
torch.cuda.set_device(args.device)

# Determine whether run dense matching or not
if args.dense_matcher is None and args.feature is not None:
    DENSE_MATCHING = False
    if args.feature is None:
        raise ValueError("Please specify a matcher for the feature extractor.")
elif args.dense_matcher is not None and args.feature is None:
    DENSE_MATCHING = True
else:
    raise ValueError("You have set both feature and dense matcher. Please choose one.")

images = [p for p in (ARIA_DATASET_ROOT / args.scene / "ns_processed" / "multi" / "undistorted_all_valid").glob("day_night_*")][0]

outputs = OUTPUT_ROOT / args.scene  # where everything will be saved
db_imagelist = outputs / "imagelists" / "db_imagelist_with_intrinsics.txt"  # the image list for the database
query_imagelist = outputs / "imagelists" / f"query_{args.time}_imagelist_with_intrinsics.txt"  # the image list for the query
db_aria2colmap_model = outputs / "colmap" / "bin" # the colmap model for the database
db_pose_pairs = outputs / f"pairs-db-pose{args.num_covis}.txt"  # top 20 closest in Aria colmap model
query_loc_pairs = outputs / f"pairs-query-{args.time}-{args.retrieval}{args.num_loc}.txt"

if DENSE_MATCHING:
    dense_matcher = args.dense_matcher
    reference_sfm = outputs / f"sfm_{dense_matcher}"  # the SfM model we will build
    results = outputs / f"hloc_{args.time}_{dense_matcher}_{args.retrieval}{args.num_loc}.txt"  # the result file

    if args.retrieval != "gt_pose":
        retrieval_conf = extract_features.confs[args.retrieval]
    dense_matcher_conf = match_dense.confs[dense_matcher]

    pairs_from_poses.main(db_aria2colmap_model, db_pose_pairs, num_matched=args.num_covis)

    features, sfm_matches = match_dense.main(dense_matcher_conf, db_pose_pairs, images, outputs)

    reconstruction = triangulation.main(reference_sfm, db_aria2colmap_model, images, db_pose_pairs, features, sfm_matches)

    if args.retrieval == "gt_pose":
        pairs_from_poses.aria_pose_main(query_loc_pairs, args.scene, args.num_loc, db_list=db_imagelist, query_list=query_imagelist)
    else:
        global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=db_imagelist)
        global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=query_imagelist)
        pairs_from_retrieval.main(
            global_descriptors, query_loc_pairs, num_matched=args.num_loc, db_list=db_imagelist, query_list=query_imagelist
    )
        
    features, loc_matches = match_dense.main(dense_matcher_conf, query_loc_pairs, images, outputs)

else:
    feature_name = args.feature
    matcher_name = args.matcher

    if args.feature == "superpoint_aachen" or args.feature == "superpoint_inloc":
        feature_name = "superpoint"

    if args.matcher == "NN-superpoint" and "superpoint" in args.feature:
        matcher_name = "NN"
    elif args.matcher == "superpoint+lightglue" and "superpoint" in args.feature:
        matcher_name = "lightglue"

    reference_sfm = outputs / f"sfm_{feature_name}+{matcher_name}"  # the SfM model we will build
    results = outputs / f"hloc_{args.time}_{feature_name}+{matcher_name}_{args.retrieval}{args.num_loc}.txt"  # the result file

    # list the standard configurations available
    print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
    print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")

    if args.retrieval != "gt_pose":
        retrieval_conf = extract_features.confs[args.retrieval]
    feature_conf = extract_features.confs[args.feature]
    matcher_conf = match_features.confs[args.matcher]

    pairs_from_poses.main(db_aria2colmap_model, db_pose_pairs, num_matched=args.num_covis)

    features = extract_features.main(feature_conf, images, outputs, image_list=db_imagelist)
    sfm_matches = match_features.main(matcher_conf, db_pose_pairs, feature_conf["output"], outputs)

    reconstruction = triangulation.main(reference_sfm, db_aria2colmap_model, images, db_pose_pairs, features, sfm_matches)

    if args.retrieval == "gt_pose":
        pairs_from_poses.aria_pose_main(query_loc_pairs, args.scene, args.num_loc, db_list=db_imagelist, query_list=query_imagelist)
    else:
        global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=db_imagelist)
        global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=query_imagelist)
        pairs_from_retrieval.main(
            global_descriptors, query_loc_pairs, num_matched=args.num_loc, db_list=db_imagelist, query_list=query_imagelist
        )

    features = extract_features.main(feature_conf, images, outputs, image_list=query_imagelist)
    loc_matches = match_features.main(matcher_conf, query_loc_pairs, feature_conf["output"], outputs)


localize_sfm.main(
    reconstruction,
    query_imagelist,
    query_loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False,
)  # not required with SuperPoint+SuperGlue