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

import argparse
import pathlib
import sys

import safetensors.torch as stt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to make sdxl checkpoint from Mac DrawThings match reference checkpoint"
    )
    parser.add_argument("ref_file", type=pathlib.Path)
    parser.add_argument("checkpoint_file", type=pathlib.Path)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    reference_file: pathlib.Path = args.ref_file
    ckpt_file: pathlib.Path = args.checkpoint_file
    device: str = args.device

    data: dict[str, torch.Tensor] = {}

    with stt.safe_open(reference_file, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():
            data[k] = f.get_tensor(k)

    with stt.safe_open(ckpt_file, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():
            if k in data:
                tensor = f.get_tensor(k)
                if tensor.shape == data[k].shape:
                    data[k] = tensor
                elif tensor.numel() == data[k].numel():
                    data[k] = tensor.view(data[k].shape)
                else:
                    print(
                        f"tensor not match: {k} of shape {data[k].shape} and {tensor.shape}",
                        file=sys.stderr,
                    )
            else:
                print(f"extra key: {k}", file=sys.stderr)

    stt.save_file(data, ckpt_file.with_stem(f"{ckpt_file.stem}-fixed"))


if __name__ == "__main__":
    main()
