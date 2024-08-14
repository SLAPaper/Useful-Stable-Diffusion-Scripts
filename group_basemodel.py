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

"""Group models by baseModel in .civitai.info"""

import argparse as ap
import collections as cl
import itertools as it
import json
import pathlib


def main(dir: pathlib.Path) -> None:
    """根据civitai info寻找Pony模型和其它模型。

    Args:
        dir (pathlib.Path): 待处理的路径。

    Returns:
        None
    """

    # 定义文件后缀列表
    suffixies = [".bin", ".pt", ".ckpt", ".pth", ".safetensors"]

    # 遍历待处理路径下的所有文件
    basemodels = cl.defaultdict(list)
    for file in it.chain.from_iterable(dir.glob(f"*{suff}") for suff in suffixies):
        # 检查是否有同名的.civitai.info文件，如果有的话则尝试读取baseModel字段
        info_file = file.with_suffix(".civitai.info")
        if info_file.exists():
            try:
                base_model = json.loads(info_file.read_text())["baseModel"]
                basemodels[base_model].append(file)
            except KeyError:
                basemodels["No Civitai Info"].append(file)
        else:
            basemodels["No Civitai Info"].append(file)

    # 分组输出结果
    for base_model, files in basemodels.items():
        print("=" * 80)
        print(f"Base Model: {base_model}")
        for file in files:
            print(file)


# 主函数入口
if __name__ == "__main__":
    # 创建一个命令行参数解析器对象
    parser = ap.ArgumentParser(description="generate empty civitai info")
    # 添加一个命令行参数，类型为pathlib.Path，表示待处理的路径
    parser.add_argument("dir", type=pathlib.Path, help="path to processing dir")
    # 解析命令行参数并存储到args变量中
    args = parser.parse_args()

    main(args.dir)
