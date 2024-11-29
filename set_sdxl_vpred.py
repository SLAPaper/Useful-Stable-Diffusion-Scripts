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
"""Make a SDXL checkpoint detected as vpred"""

import argparse
import pathlib

import safetensors.torch as stt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to add or remove vpred flag from sdxl checkpoints"
    )
    parser.add_argument("checkpoint_file", type=pathlib.Path)
    parser.add_argument("--remove", action="store_true", help="Remove vpred flag from sdxl model.")


    args = parser.parse_args()
    ckpt_file: pathlib.Path = args.checkpoint_file
    is_remove: bool = args.remove

    data: dict[str, torch.Tensor] = stt.load_file(ckpt_file)

    if is_remove:
        if "v_pred" in data:
            del data["v_pred"]

        if "ztsnr" in data:
            del data["ztsnr"]
    else:
        data["v_pred"] = torch.tensor([])
        data["ztsnr"] = torch.tensor([])

    stt.save_file(data, ckpt_file.with_stem(f"{ckpt_file.stem}-vpred"))


if __name__ == "__main__":
    main()
