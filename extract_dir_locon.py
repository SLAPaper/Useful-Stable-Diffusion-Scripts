# Copyright 2024 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""use LyCORIS to extract locon from checkpoints"""

import argparse
import pathlib
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to bake vae into sdxl checkpoint"
    )
    parser.add_argument("base_checkpoint", type=pathlib.Path)
    parser.add_argument("checkpoint_dir", type=pathlib.Path)
    parser.add_argument("target_dir", type=pathlib.Path)
    parser.add_argument("--lyco-dir", type=pathlib.Path, default="LyCORIS")
    parser.add_argument("--sdxl", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")

    args = parser.parse_args()
    base_ckpt: pathlib.Path = args.base_checkpoint
    ckpt_dir: pathlib.Path = args.checkpoint_dir
    target_dir: pathlib.Path = args.target_dir
    device: str = args.device
    dtype: str = args.dtype

    for ckpt in ckpt_dir.glob("*.safetensors"):
        target_file = target_dir / f"{ckpt.stem}_locon-r90.safetensors"

        params = [
            "python",
            str(args.lyco_dir / "tools" / "extract_locon.py"),
        ]

        if args.sdxl:
            params += ["--is_sdxl"]

        params += [
            "--device",
            str(device),
            "--dtype",
            str(dtype),  # official version LyCORIS does not support dtype selection
            "--mode",
            "ratio",
            "--safetensors",
            "--linear_ratio",
            "0.9",
            "--conv_ratio",
            "0.9",
            str(base_ckpt),
            str(ckpt),
            str(target_file),
        ]
        print("Processing:", ckpt)
        subprocess.run(params, check=True)

        # copy preview file
        preview_path = target_file.with_suffix(".preview.png")
        preview_file = ckpt.with_suffix(".preview.png")

        if preview_file.exists():
            shutil.copyfile(preview_file, preview_path)

        # copy thumbnail file
        thumbnail_path = target_file.with_suffix(".webp")
        thumbnail_file = ckpt.with_suffix(".webp")
        if thumbnail_file.exists():
            shutil.copyfile(thumbnail_file, thumbnail_path)

        # copy civitai info
        civitai_path = target_file.with_suffix(".civitai.info")
        civitai_file = ckpt.with_suffix(".civitai.info")
        if civitai_file.exists():
            shutil.copyfile(civitai_file, civitai_path)


if __name__ == "__main__":
    main()
