"""
Download the example crop and checkpoints from the public e11bio-prism bucket.

We only grab what we need, about 4gb total:

* raw + labels for one crop -> <data_dir>/sample.zarr
* model_checkpoint_* weights for the four models -> <checkpoint_dir>/<model>/

The bucket also has model.pt files, but we can't load those here (see
checkpoints.py), so we skip them.

    uv run python prism_post/data/download.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from prism_post.config import (
    BUCKET,
    CHECKPOINT_KEY,
    CROP_KEY,
    LABELS,
    MODEL_NAMES,
    RAW,
    SAMPLE,
    argument_parser,
)


def _download_one(client, bucket: str, key: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return key


def download_prefix(
    bucket: str,
    prefix: str,
    output: Path,
    include: list[str] | None = None,
    workers: int = 16,
) -> None:
    """
    Download everything under prefix into output, keeping the relative
    paths. If include is given, only files whose path (relative to prefix)
    starts with one of those strings are downloaded.
    """
    cfg = Config(signature_version=UNSIGNED)
    client = boto3.client("s3", config=cfg)
    bucket_resource = boto3.resource("s3", config=cfg).Bucket(bucket)

    keys = []
    for obj in bucket_resource.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue
        rel = obj.key[len(prefix) :].lstrip("/")
        if include is not None and not any(rel.startswith(i) for i in include):
            continue
        keys.append((obj.key, output / rel))

    if not keys:
        print(f"no files found under s3://{bucket}/{prefix}")
        return

    print(f"downloading {len(keys)} files to {output} ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_one, client, bucket, key, dest) for key, dest in keys
        ]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 50 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}")


def download_sample(data_dir: Path) -> Path:
    sample = data_dir / SAMPLE
    if (sample / RAW).exists() and (sample / LABELS).exists():
        print(f"{sample} already exists, skipping")
        return sample
    download_prefix(
        BUCKET, CROP_KEY, sample, include=[".zgroup", f"{RAW}/", f"{LABELS}/"]
    )
    return sample


def download_checkpoints(checkpoint_dir: Path) -> None:
    for name in MODEL_NAMES:
        target = checkpoint_dir / name
        if target.exists() and any(target.glob("model_checkpoint_*")):
            print(f"{target} already exists, skipping")
            continue
        download_prefix(
            BUCKET,
            f"{CHECKPOINT_KEY}/{name}",
            target,
            include=["model_checkpoint_"],
            workers=4,
        )


if __name__ == "__main__":
    args = argument_parser(__doc__).parse_args()
    download_sample(args.data_dir)
    download_checkpoints(args.checkpoint_dir)
