# Copyright 2024 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib

import safetensors as st
import safetensors.torch as stt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to bake vae into sdxl checkpoint"
    )
    parser.add_argument("checkpoint_file", type=pathlib.Path)

    args = parser.parse_args()
    ckpt_file: pathlib.Path = args.checkpoint_file

    data: dict[str, dict[str, torch.Tensor]] = torch.load(ckpt_file, weights_only=True)

    stt.save_file(
        data["state_dict"], ckpt_file.with_name(f"{ckpt_file.stem}.safetensors")
    )


if __name__ == "__main__":
    main()
