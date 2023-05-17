import sys
import numpy as np
import numpy.typing as npt
import os
import cv2 as cv
import matplotlib.pyplot as plt
from enum import Enum


class Colors(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

    def to_color(self) -> str:
        if self == Colors.RED:
            return "#ff0000"
        elif self == Colors.GREEN:
            return "#00ff00"
        elif self == Colors.BLUE:
            return "#0000ff"
        else:
            raise Exception("Enum not valid")


class Trayectory:
    color: Colors
    x_points: npt.NDArray[np.uint16]
    y_points: npt.NDArray[np.uint16]

    def __init__(
        self, color: Colors, x_points: npt.NDArray[np.uint16], y_points: npt.NDArray[np.uint16]
    ):
        self.color = color
        self.x_points = x_points
        self.y_points = y_points


def get_image() -> cv.Mat:
    if not len(sys.argv) == 2:
        raise Exception("Image path not provided")

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        raise Exception("File not found")

    supported_file_types = ".png", ".jpg"
    if not image_path.endswith(supported_file_types):
        raise Exception(
            f"File type not supported, please use one of the following: {supported_file_types}"
        )

    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    return image


def create_mask_for_color(image: cv.Mat, color: Colors) -> npt.NDArray[np.bool_]:
    red, green, blue = 0, 0, 0
    if color == Colors.RED:
        red = 255
    elif color == Colors.GREEN:
        green = 255
    elif color == Colors.BLUE:
        blue = 255

    red_mask = image[:, :, 0] == red
    green_mask = image[:, :, 1] == green
    blue_mask = image[:, :, 2] == blue

    return np.bitwise_and(red_mask, green_mask, blue_mask)


def get_contours_per_color(image: cv.Mat) -> dict[Colors, tuple]:
    contours_per_color: dict[Colors, tuple] = {}
    for color in Colors:
        color_mask = create_mask_for_color(image, color)
        thresh = (color_mask * 255).astype(np.uint8)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours_per_color[color] = contours
    return contours_per_color


def get_trayectories(image: cv.Mat, contours_per_color: dict[Colors, tuple]) -> list[Trayectory]:
    height = image.shape[1]

    trayectories: list[Trayectory] = []

    for color in Colors:
        for contour in contours_per_color[color]:
            x_points = contour[:, :, 0]
            y_points = height - contour[:, :, 1]

            trayectories.append(Trayectory(color, x_points, y_points))

    return trayectories


def main():
    image = get_image()
    contours_per_color = get_contours_per_color(image)
    trayectories = get_trayectories(image, contours_per_color)

    for trayectory in trayectories:
        plt.xlim([0, 500])
        plt.ylim([0, 500])
        plt.plot(trayectory.x_points, trayectory.y_points, trayectory.color.to_color())
    plt.show()


if __name__ == "__main__":
    main()
