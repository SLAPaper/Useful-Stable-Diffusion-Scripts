# Copyright 2023 SLAPaper
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
"""fix vae in sdxl"""

import argparse
import pathlib

import safetensors as st
import safetensors.torch as stt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to bake vae into sdxl checkpoint"
    )
    parser.add_argument("vae_file", type=pathlib.Path)
    parser.add_argument("checkpoint_file", type=pathlib.Path)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")

    args = parser.parse_args()
    ckpt_file: pathlib.Path = args.checkpoint_file
    vae_file: pathlib.Path = args.vae_file
    device: str = args.device
    dtype: str = args.dtype

    data: dict[str, torch.Tensor] = {}
    with st.safe_open(ckpt_file, framework="pt", device=device) as f:
        for key in f.keys():
            data[key] = f.get_tensor(key).to(dtype=getattr(torch, dtype))

    with st.safe_open(vae_file, framework="pt", device=device) as f:
        for key in f.keys():
            data[f"first_stage_model.{key}"] = f.get_tensor(key).to(
                dtype=getattr(torch, dtype)
            )

    stt.save_file(data, ckpt_file.with_stem(f"{ckpt_file.stem}-fixed"))


if __name__ == "__main__":
    main()
