"""Paths and global settings. Override the locations with environment variables:
BER_DATA_DIR (folder holding train/ and test/) and BER_WORK_DIR (intermediate files)."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(_HERE)

DATA_DIR = os.environ.get("BER_DATA_DIR", os.path.join(PKG_DIR, "..", "6ab10eb3b23ba_student_resource",
                                                       "student_resource", "dataset"))
WORK_DIR = os.environ.get("BER_WORK_DIR", os.path.join(PKG_DIR, "..", "work"))
OUTPUT_DIR = os.environ.get("BER_OUTPUT_DIR", os.path.join(PKG_DIR, "..", "output"))

N_JOBS = int(os.environ.get("BER_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))
SEED = 42

SOURCES = ("source1", "source2", "source3")


def raw_path(split, source):
    return os.path.join(DATA_DIR, split, f"{split}_{source}.tsv")


def norm_path(split, source):
    return os.path.join(WORK_DIR, "norm", f"{split}_{source}.parquet")


def work(*parts):
    p = os.path.join(WORK_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
