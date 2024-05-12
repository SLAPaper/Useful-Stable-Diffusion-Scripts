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
import logging
import pathlib
import typing as tg

import safetensors as st
import safetensors.torch as stt
import torch

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class SafeOpenStub(dict[str, torch.Tensor]):
    """Stub for return object of safetensors.safe_open"""

    def get_tensor(self, key: str) -> torch.Tensor:
        raise NotImplementedError()


POS_IDS_KEY = "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"
LOGIT_SCALE_KEY = "conditioner.embedders.1.model.logit_scale"


def try_fix_file(file: pathlib.Path) -> None:
    """try to fix file if it has a broken sdxl clip"""
    model = tg.cast(SafeOpenStub, st.safe_open(file, framework="pt"))
    model_keys = set(model.keys())

    need_fix = False
    while True:
        if POS_IDS_KEY not in model_keys:
            need_fix = True
            break

        pos_id_key = model.get_tensor(POS_IDS_KEY)
        if pos_id_key.numel() != 77:
            need_fix = True
            break

        if pos_id_key.dtype.is_floating_point:
            need_fix = True
            break

        if LOGIT_SCALE_KEY not in model_keys:
            need_fix = True
            break

        break

    if need_fix:
        logger.info(f"try fixing {file.name}")

        data = {}
        for key in model.keys():
            data[key] = model.get_tensor(key)

        data[POS_IDS_KEY] = torch.Tensor([list(range(77))]).to(dtype=torch.int64)

        if LOGIT_SCALE_KEY not in data:
            data[LOGIT_SCALE_KEY] = torch.tensor(
                4.6055, dtype=torch.float16  # value from sd_xl base 1.0
            )

        stt.save_file(data, file.with_stem(f"{file.stem}-fixed"))
    else:
        logger.info(f"no need to fix {file.name}")
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to fix missing position_ids in sdxl clip"
    )
    parser.add_argument("safetensor_file", type=pathlib.Path)

    args = parser.parse_args()
    file: pathlib.Path = args.safetensor_file

    if file.is_dir():
        for f in file.glob("*.safetensors"):
            try_fix_file(f)
    else:
        try_fix_file(file)


if __name__ == "__main__":
    main()
