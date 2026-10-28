"""
Look at datasets in a crop zarr with neuroglancer.

    uv run python prism_post/view.py # everything that exists
    uv run python prism_post/view.py raw predictions/enhanced labels   # just some

Needs the vis extra: uv sync --extra vis. If you are running this on a remote
machine, pass --bind-address 0.0.0.0 and open the printed link from your
laptop (replace the host with the machine's address).
"""

import argparse

import neuroglancer
import numpy as np
from funlib.persistence import open_ds

from prism_post.config import CROPS, DEFAULT_CROP, crop_path, default_data_dir

# how to show each dataset, by its path inside the crop zarr. anything not
# listed here (fragments/*, segments/*, ...) is shown as a segmentation
IMAGES = {
    "raw": "barcodes",
    "predictions/enhanced": "barcodes",
    "predictions/binary": "gray",
    "predictions/mask": "mask",
    "predictions/uniform": "rgb",
    "affs/pred_affs": "rgb",
    "affs/pred_affs_raw": "rgb",
    "affs/pseudo_affs": "rgb",
    "affs/combined_affs": "rgb",
}

# sweep outputs are not opened by default, there are many of them.
# prism_post/view_sweep.py is for those
SKIP = ("segments/sweep", "luts/sweep")

# affinities as rgb. channels 3, 4, 5 are the medium range offsets, which
# show boundaries more clearly than the direct neighbours
RGB_SHADER = """
void main() {
    emitRGB(vec3(
        toNormalized(getDataValue(3)),
        toNormalized(getDataValue(4)),
        toNormalized(getDataValue(5))));
}
"""

GRAY_SHADER = """
void main() {
    emitGrayscale(toNormalized(getDataValue()));
}
"""

# 0/1 data, show as black/white
MASK_SHADER = """
void main() {
    emitGrayscale(255.0 * toNormalized(getDataValue()));
}
"""


def barcode_shader(num_channels, ranges):
    """
    Shader for the 18 channel barcode data. Every channel gets a colour
    (red, green, blue, repeating), its own display range (defaults from the
    data percentiles) and a checkbox, plus contrast / brightness / per colour
    scale sliders and a "seed" slider that shuffles which channel gets which
    colour, which makes it much easier to tell neighbouring neurons apart.
    """
    defaults = {}
    for i, (lo, hi) in enumerate(ranges):
        defaults[f"chan{i}_min"] = round(float(lo), 4)
        defaults[f"chan{i}_max"] = round(float(hi), 4)
    return create_shuffle_shader(num_channels, saved_defaults=defaults)


def create_shuffle_shader(
    num_channels, slider_default_min=0.0, slider_default_max=0.03, saved_defaults=None
):
    assert num_channels >= 3, (
        f"Data must be at least three dimensional, got {num_channels} channels!"
    )

    shader = []

    # ui controls for each channel
    for i in range(num_channels):
        # get saved defaults if any
        chan_active_default = (
            saved_defaults.get(f"chan{i}_active", True) if saved_defaults else True
        )
        chan_min_default = (
            saved_defaults.get(f"chan{i}_min", slider_default_min)
            if saved_defaults
            else slider_default_min
        )
        chan_max_default = (
            saved_defaults.get(f"chan{i}_max", slider_default_max)
            if saved_defaults
            else slider_default_max
        )

        # adjust min and max values if necessary
        chan_min = min(0, chan_min_default)
        chan_max = max(1, chan_max_default)

        shader.extend(
            [
                f"#uicontrol bool chan{i}_active checkbox(default={str(chan_active_default).lower()})",
                f"#uicontrol float chan{i}_min slider(min={chan_min}, max=1, step=0.001, default={chan_min_default})",
                f"#uicontrol float chan{i}_max slider(min=0, max={chan_max}, step=0.001, default={chan_max_default})",
            ]
        )

    # extra controls with checks for saved defaults
    contrast_default = saved_defaults.get("contrast", 0) if saved_defaults else 0
    brightness_default = saved_defaults.get("brightness", 0) if saved_defaults else 0
    scaleR_default = saved_defaults.get("scaleR", 1) if saved_defaults else 1
    scaleG_default = saved_defaults.get("scaleG", 1) if saved_defaults else 1
    scaleB_default = saved_defaults.get("scaleB", 1) if saved_defaults else 1
    seed_default = saved_defaults.get("seed", 1) if saved_defaults else 1

    # dynamically adjust contrast, brightness, scaleR min/max values if necessary
    contrast_min = min(-3, contrast_default)
    contrast_max = max(3, contrast_default)
    brightness_min = min(-3, brightness_default)
    brightness_max = max(3, brightness_default)
    scaleR_min = min(-3, scaleR_default)
    scaleR_max = max(3, scaleR_default)
    scaleG_min = min(-3, scaleG_default)
    scaleG_max = max(3, scaleG_default)
    scaleB_min = min(-3, scaleB_default)
    scaleB_max = max(3, scaleB_default)
    scaleB_max = max(3, scaleB_default)
    seed_min = min(0, seed_default)
    seed_max = max(20, seed_default)

    shader.extend(
        [
            "#uicontrol float contrast slider("
            f"min={contrast_min}, max={contrast_max}, step=0.01, default={contrast_default})",
            "#uicontrol float brightness slider("
            f"min={brightness_min}, max={brightness_max}, step=0.01, default={brightness_default})",
            f"#uicontrol float scaleR slider(min={scaleR_min}, max={scaleR_max}, step=0.01, default={scaleR_default})",
            f"#uicontrol float scaleG slider(min={scaleG_min}, max={scaleG_max}, step=0.01, default={scaleG_default})",
            f"#uicontrol float scaleB slider(min={scaleB_min}, max={scaleB_max}, step=0.01, default={scaleB_default})",
            f"#uicontrol float seed slider(min={seed_min}, max={seed_max}, step=1, default={seed_default})",
            "#uicontrol bool invert checkbox(default=false)",
        ]
    )

    # helper functions for display range and random shuffling
    shader.extend(
        [
            "float disp_range(float im, float min, float max) {",
            "    return clamp((im - min) / (max - min), 0.0, 1.0);",
            "}",
            "float rand(float x) {",
            "    return fract(sin(dot(vec2(x, seed), vec2(12.9898, 78.233))) * 43758.5453);",
            "}",
            f"vec3[{num_channels}] shuffleBoolArray(vec3 arr[{num_channels}], int n) {{",
            "    vec3[] output_arr = vec3[]("
            + ", ".join([f"arr[{i}]" for i in range(num_channels)])
            + ");",
            "    if (seed == 0.) {",
            "        return output_arr;",
            "    }",
            "    for (int i = n - 1; i > 0; i--) {",
            "        int j = int(rand(float(i)) * float(i + 1));",
            "        vec3 temp = output_arr[i];",
            "        output_arr[i] = output_arr[j];",
            "        output_arr[j] = temp;",
            "    }",
            "    return output_arr;",
            "}",
        ]
    )

    # base color pattern
    color_pattern = ["vec3(1, 0, 0)", "vec3(0, 1, 0)", "vec3(0, 0, 1)"]

    # repeat pattern for extra channels
    extended_colors = (color_pattern * ((num_channels + 2) // 3))[:num_channels]

    # convert to string
    temp_colors = ", ".join(extended_colors)

    # add to glsl
    temp_colors_definition = (
        f"vec3 temp_colors[{num_channels}] = vec3[]({temp_colors});"
    )

    # main logic
    shader.extend(
        [
            "void main() {",
            "    vec3 colorSum = vec3(0.0);",
            f"    float channelData[{num_channels}] = float[]("
            + ", ".join(
                [f"toNormalized(getDataValue({i}))" for i in range(num_channels)]
            )
            + ");",
            f"    float minVals[{num_channels}] = float[]("
            + ", ".join([f"chan{i}_min" for i in range(num_channels)])
            + ");",
            f"    float maxVals[{num_channels}] = float[]("
            + ", ".join([f"chan{i}_max" for i in range(num_channels)])
            + ");",
            f"    bool active_chans[{num_channels}] = bool[]("
            + ", ".join([f"chan{i}_active" for i in range(num_channels)])
            + ");",
            f"    {temp_colors_definition}",
            f"    vec3[{num_channels}] colors = shuffleBoolArray(temp_colors, {num_channels});",
            f"    for (int i = 0; i < {num_channels}; i++) {{",
            "        if (active_chans[i]) {",
            "            colorSum += colors[i] * disp_range(channelData[i], minVals[i], maxVals[i]);",
            "        }",
            "    }",
            "    vec3 final_im = vec3(colorSum.r * scaleR, colorSum.g * scaleG, colorSum.b * scaleB);",
            "    final_im = clamp(final_im, 0.0, 1.0);",
            "    final_im = (final_im - 0.5) * pow(2.0, contrast) + 0.5 + brightness;",
            "    if (invert == true) {emitRGB(1.0-final_im);} else {emitRGB(final_im);}"
            "}",
        ]
    )

    return "\n".join(shader)


def channel_ranges(array, step=4):
    """Per channel (50th, 99.9th) percentiles in [0, 1], on a subsample for speed."""
    data = array.data[:, ::step, ::step, ::step]
    data = np.asarray(data)
    return [tuple(np.percentile(c, [50, 99.9]) / 255) for c in data]


def find_arrays(root, skip=SKIP):
    """All zarr arrays under root, as paths relative to it."""
    found = []
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if skip and rel.startswith(skip):
            continue
        if p.is_dir() and ((p / ".zarray").exists() or _is_v3_array(p)):
            found.append(rel)
    return found


def _is_v3_array(p):
    meta = p / "zarr.json"
    return meta.exists() and '"node_type": "array"' in meta.read_text()


def local_volume(array):
    """A neuroglancer volume for a funlib array, channels first if it has any."""
    spatial = array.voxel_size.dims
    names = ["z", "y", "x"][-spatial:]
    scales = list(array.voxel_size)
    offset = [o // v for o, v in zip(array.roi.offset, array.voxel_size)]
    if array.channel_dims:
        names = ["c^"] + names
        scales = [1] + scales
        offset = [0] + offset
    dims = neuroglancer.CoordinateSpace(names=names, units="nm", scales=scales)
    return neuroglancer.LocalVolume(
        data=array.data, dimensions=dims, voxel_offset=offset
    )


def make_layer(sample, name):
    """A neuroglancer layer for a dataset in the crop zarr, shader by kind."""
    array = open_ds(sample / name)
    vol = local_volume(array)
    kind = IMAGES.get(name)
    if kind == "barcodes":
        return neuroglancer.ImageLayer(
            source=vol, shader=barcode_shader(array.shape[0], channel_ranges(array))
        )
    if kind == "rgb":
        return neuroglancer.ImageLayer(source=vol, shader=RGB_SHADER)
    if kind == "gray":
        return neuroglancer.ImageLayer(source=vol, shader=GRAY_SHADER)
    if kind == "mask":
        return neuroglancer.ImageLayer(source=vol, shader=MASK_SHADER)
    if np.issubdtype(array.dtype, np.integer):
        # labels, fragments/*, segments/*
        return neuroglancer.SegmentationLayer(source=vol)
    return neuroglancer.ImageLayer(source=vol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="datasets to show (default: all)")
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--crop", type=int, choices=CROPS, default=DEFAULT_CROP)
    parser.add_argument("-b", "--bind-address", default="localhost")
    args = parser.parse_args()

    sample = crop_path(args.data_dir, args.crop)
    if args.datasets:
        names = args.datasets
    else:
        existing = set(find_arrays(sample))
        names = [n for n in IMAGES if n in existing] + sorted(
            n for n in existing if n not in IMAGES
        )

    neuroglancer.set_server_bind_address(args.bind_address)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for name in names:
            s.layers[name] = make_layer(sample, name)
            s.layers[name].visible = name == "raw"

    print(viewer)
    input("press enter to quit\n")
