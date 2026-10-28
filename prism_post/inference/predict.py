"""
Helper for the inference scripts, runs one model over one dataset with volara.
"""

from pathlib import Path

from volara.datasets import Affs, Raw
from volara.logging import set_log_basedir
from volara.workers import LocalWorker
from volara_ml.blockwise import Predict

from prism_post.blockwise import run
from prism_post.checkpoints import torch_model
from prism_post.config import NEIGHBORHOOD, affs_name


def run_predict(
    model_name: str,
    in_dataset: str,
    out_datasets: list[str | None],
    sample: Path,
    checkpoint_dir: Path,
    num_workers: int = 1,
    affs: bool = False,
) -> None:
    """
    Predict out_datasets from in_dataset (both inside the crop zarr sample)
    with model_name.

    Inputs are uint8 on disk and scaled to [0, 1] for the network. A None in
    out_datasets throws that output away (e.g. the lsds from the affs model).
    affs=True stores the first output as a volara Affs dataset so it carries
    the neighborhood.

    Each worker is its own process (LocalWorker) and loads the model onto
    cuda:<worker_id % num_gpus>, so num_workers=2 uses two gpus. We can't
    fork workers after the model is already on a gpu, cuda doesn't like it.
    """
    # logs and the block done cache live with the crop, see config.py
    set_log_basedir(sample / "volara_logs")

    in_data = Raw(store=sample / in_dataset, scale_shift=(1 / 255, 0))

    out_data = []
    for i, name in enumerate(out_datasets):
        if name is None:
            out_data.append(None)
        elif affs and i == 0:
            out_data.append(
                Affs(store=sample / affs_name(name), neighborhood=NEIGHBORHOOD)
            )
        else:
            out_data.append(Raw(store=sample / name))

    predict = Predict(
        checkpoint=torch_model(model_name, checkpoint_dir),
        in_data=in_data,
        out_data=out_data,
        num_workers=num_workers,
        worker_config=LocalWorker(),
    )
    run(predict)
