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
"""generate empty civitai info"""

import argparse as ap
import itertools as it
import pathlib
import sys


def main(dir: pathlib.Path) -> None:
    """生成空的civitai info文件。

    Args:
        dir (pathlib.Path): 待处理的路径。

    Returns:
        None
    """

    # 定义文件后缀列表
    suffixies = [".bin", ".pt", ".ckpt", ".pth", ".safetensors", ".gguf"]

    # 遍历待处理路径下的所有文件
    for file in it.chain.from_iterable(dir.glob(f"*{suff}") for suff in suffixies):
        # 检查是否有同名的.civitai.info文件，如果没有的话则创建一个内容为"{}"的同名.civitai.info文件
        info_file = file.with_suffix(".civitai.info")
        if not info_file.exists():
            # 输出正在为该文件生成空的civitai info文件的提示信息
            print("generating empty civitai info for", file, file=sys.stderr)

            # 打开同名.civitai.info文件并写入内容"{}"
            with info_file.open("w") as f:
                f.write("{}")
        else:
            print(
                "skipping",
                file,
                "because it already has a civitai info file",
                file=sys.stderr,
            )


# 主函数入口
if __name__ == "__main__":
    # 创建一个命令行参数解析器对象
    parser = ap.ArgumentParser(description="generate empty civitai info")
    # 添加一个命令行参数，类型为pathlib.Path，表示待处理的路径
    parser.add_argument("dir", type=pathlib.Path, help="path to processing dir")
    # 解析命令行参数并存储到args变量中
    args = parser.parse_args()

    main(args.dir)
