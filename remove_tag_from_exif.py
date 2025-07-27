# Copyright 2025 SLAPaper
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

import argparse

import piexif
import piexif.helper
from PIL import Image


def remove_tag_from_exif(exif_comment: str, tags: list[str]) -> str:
    """
    Remove a specific tag from the EXIF data of an image.

    Args:
        exif_comment (str): The EXIF comment to modify.
        tags (list[str]): The tag list to remove from the EXIF data.
    """
    for tag in tags:
        exif_comment = exif_comment.replace(tag, "")
    return exif_comment.strip()


def main() -> None:
    """Main function to handle command line arguments and process the image."""
    parser = argparse.ArgumentParser(description="Remove specific tag from EXIF data.")
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument("output_path", help="Path to save the modified image.")
    parser.add_argument("--tag", help="Tag to remove.", action="append")

    args = parser.parse_args()

    img = Image.open(args.image_path)
    exif_data = piexif.load(img.info.get("exif", b""))
    exif_comment = piexif.helper.UserComment.load(
        (exif_data or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
    )

    new_comment = remove_tag_from_exif(exif_comment, args.tag)
    new_bytes = piexif.dump(
        {
            "Exif": {
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                    new_comment, encoding="unicode"
                )
            }
        }
    )

    piexif.insert(new_bytes, args.output_path)


if __name__ == "__main__":
    main()
