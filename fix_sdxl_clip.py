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
import typing as tg

import safetensors as st
import safetensors.torch as stt
import torch


class SafeOpenStub(dict[str, torch.Tensor]):
    """Stub for return object of safetensors.safe_open"""

    def get_tensor(self, key: str) -> torch.Tensor:
        raise NotImplementedError()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to fix missing position_ids in sdxl clip"
    )
    parser.add_argument("safetensor_file", type=pathlib.Path)

    args = parser.parse_args()
    file: pathlib.Path = args.safetensor_file

    data = {}
    model = tg.cast(SafeOpenStub, st.safe_open(file, framework="pt"))

    for key in model.keys():
        data[key] = model.get_tensor(key)

    data["conditioner.embedders.0.transformer.text_model.embeddings.position_ids"] = (
        torch.Tensor([list(range(77))]).to(dtype=torch.int64)
    )

    if "conditioner.embedders.1.model.logit_scale" not in data:
        data["conditioner.embedders.1.model.logit_scale"] = torch.tensor(
            4.6055, dtype=torch.float16  # value from sd_xl base 1.0
        )

    stt.save_file(data, file.with_stem(f"{file.stem}-fixed"))


if __name__ == "__main__":
    main()
