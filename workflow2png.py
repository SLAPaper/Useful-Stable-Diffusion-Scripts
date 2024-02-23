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

import argparse
import json
import pathlib

from PIL import Image, PngImagePlugin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Program that embed workflow json into an PNG image")
    parser.add_argument("workflow_file",
                        type=pathlib.Path,
                        help="the workflow json exported by ComfyUI")
    parser.add_argument(
        "image_file",
        type=pathlib.Path,
        help="the image file that you want to embed workflow in")

    args = parser.parse_args()

    with args.workflow_file.open(encoding='utf8') as f:
        data = json.load(f)

        image = Image.open(args.image_file)
        info = PngImagePlugin.PngInfo()
        info.add_text("workflow", json.dumps(data, separators=(',', ':')))
        image.save(args.workflow_file.with_suffix('.png'), "PNG", pnginfo=info)
