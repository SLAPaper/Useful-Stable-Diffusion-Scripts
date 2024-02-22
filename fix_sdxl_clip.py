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
"""fix clip in sdxl"""

import argparse
import pathlib

import safetensors as st
import safetensors.torch as stt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to fix missing position_ids in sdxl clip"
    )
    parser.add_argument("safetensor_file", type=pathlib.Path)

    args = parser.parse_args()
    file: pathlib.Path = args.safetensor_file

    data = {}
    with st.safe_open(file, framework="pt") as f:
        for key in f.keys():
            data[key] = f.get_tensor(key)

    data[
        "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"
    ] = torch.Tensor([list(range(77))]).to(dtype=torch.int64)

    stt.save_file(data, file.with_stem(f"{file.stem}-fixed"))


if __name__ == '__main__':
    main()
