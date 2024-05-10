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

"""Convert SDXL safetensor (unet part) to fp8e4m3 storage type"""
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


UNET_KEY_PREFIX = "model.diffusion_model."


def check_can_convert(key: str, tensor: torch.Tensor) -> bool:
    """check if the tensor can be converted to fp8e4m3 storage type"""
    if not tensor.dtype.is_floating_point:
        return False

    if key.startswith(UNET_KEY_PREFIX):
        return True

    return False


def try_convert(file: pathlib.Path) -> None:
    """try to convert"""
    model = tg.cast(SafeOpenStub, st.safe_open(file, framework="pt"))

    logger.info(f"try converting {file.name}")

    data = {}
    suffix = "fp8e4m3"
    for key in model.keys():
        tensor = model.get_tensor(key)
        if check_can_convert(key, tensor):
            data[key] = tensor.to(torch.float8_e4m3fn)
        else:
            data[key] = tensor

    stt.save_file(data, file.with_stem(f"{file.stem}-{suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tool to convert weight to fp8e4m3 in sdxl unet"
    )
    parser.add_argument(
        "safetensor_file",
        type=pathlib.Path,
        help="the path of target file or directory containing target files",
    )

    args = parser.parse_args()
    file: pathlib.Path = args.safetensor_file

    if file.is_dir():
        for f in file.glob("*.safetensors"):
            try_convert(f)
    else:
        try_convert(file)


if __name__ == "__main__":
    main()
