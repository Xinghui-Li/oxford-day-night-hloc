import logging

from pathlib import Path

ARIA_DATASET_ROOT = Path("datasets/oxford/release_v0.1/aria")
OUTPUT_ROOT = Path("datasets/oxford_output_v0.1")
RANDOM_SEED = 6

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger = logging.getLogger("Convert ARIA dataset to COLMAP")
logger.addHandler(handler)
logger.setLevel(logging.INFO)