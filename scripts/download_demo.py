"""Use the official downloader with an immutable dataset revision."""

from mani_skill.utils.download_demo import DATASET_SOURCES, main, parse_args

REVISION = "d674485bbffdd533914e52d272fdda34c0515608"
DATASET_SOURCES["PickCube-v1"].raw_dataset_url = (
    f"https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/{REVISION}"
    "/demos/PickCube-v1.zip?download=true"
)
main(parse_args(["PickCube-v1"]))
