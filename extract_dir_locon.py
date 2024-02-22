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
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to bake vae into sdxl checkpoint"
    )
    parser.add_argument("base_checkpoint", type=pathlib.Path)
    parser.add_argument("checkpoint_dir", type=pathlib.Path)
    parser.add_argument("target_dir", type=pathlib.Path)
    parser.add_argument("--lyco-dir", type=pathlib.Path, default="LyCORIS")

    args = parser.parse_args()
    base_ckpt: pathlib.Path = args.base_checkpoint
    ckpt_dir: pathlib.Path = args.checkpoint_dir
    target_dir: pathlib.Path = args.target_dir

    for ckpt in ckpt_dir.glob("*.safetensors"):
        target_file = target_dir / f"{ckpt.name}_locon-r90.safetensors"

        params = [
            "python",
            str(args.lyco_dir / "tools" / "extract_locon.py"),
            "--is_sdxl",
            "--device",
            "cuda",
            "--mode",
            "ratio",
            "--safetensors",
            "--linear_ratio",
            "0.9",
            "--conv_ratio",
            "0.9",
            "--use_sparse_bias",
            str(base_ckpt),
            str(ckpt),
            str(target_file),
        ]
        print("Processing:", ckpt)
        subprocess.run(params, check=True)


if __name__ == "__main__":
    main()
