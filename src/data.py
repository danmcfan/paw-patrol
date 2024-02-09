import os
from xml.etree import ElementTree

import numpy as np
from PIL import Image as PILImage
from pydantic import BaseModel


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Object(BaseModel):
    name: str
    bounding_box: BoundingBox


class Image(BaseModel):
    folder: str
    filename: str
    image_filepath: str

    width: int
    height: int
    depth: int

    objects: list[Object]

    def get_pil_image(self) -> PILImage.Image:
        return PILImage.open(self.image_filepath)

    def get_numpy_image(self) -> np.ndarray:
        return np.array(self.get_pil_image())


def create_image(filepath: str) -> Image:
    tree = ElementTree.parse(filepath)
    root = tree.getroot()
    size = root.find("size")

    image_filepath = filepath.replace("annotations", "images") + ".jpg"

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox = BoundingBox(
            xmin=int(bndbox.find("xmin").text),
            ymin=int(bndbox.find("ymin").text),
            xmax=int(bndbox.find("xmax").text),
            ymax=int(bndbox.find("ymax").text),
        )
        objects.append(Object(name=name, bounding_box=bbox))

    return Image(
        folder=root.find("folder").text,
        filename=root.find("filename").text,
        image_filepath=image_filepath,
        width=int(size.find("width").text),
        height=int(size.find("height").text),
        depth=int(size.find("depth").text),
        objects=objects,
    )


def create_images() -> list[Image]:
    images = []

    for folder in os.listdir("data/annotations"):
        for filename in os.listdir(f"data/annotations/{folder}"):
            filepath = f"data/annotations/{folder}/{filename}"
            images.append(create_image(filepath))

    return images
