import sys
import random
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import os
import cv2 as cv
import matplotlib.pyplot as plt
from enum import Enum
import matplotlib.animation as ani


class Colors(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

    def to_hex(self) -> str:
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
    x_points: npt.NDArray[np.int32]
    y_points: npt.NDArray[np.int32]

    def __init__(
        self, color: Colors, x_points: npt.NDArray[np.int32], y_points: npt.NDArray[np.int32]
    ):
        self.color = color
        self.x_points = x_points
        self.y_points = y_points


class PlotAnimation:
    x_points: list[np.int32]
    y_points: list[np.int32]
    figure: Figure
    axes: Axes
    image: cv.Mat

    def __init__(self, image: cv.Mat, trayectories: list[Trayectory]):
        self.image = image
        self.figure, self.axes = plt.subplots()
        self.x_points = []
        self.y_points = []

        for trayectory in trayectories:
            for x, y in zip(trayectory.x_points, trayectory.y_points):
                self.x_points.append(x)
                self.y_points.append(y)

    def animate(self, i):
        self.axes.clear()
        self.axes.axis(False)
        self.axes.plot(self.x_points[0 : i + 1], self.y_points[0 : i + 1])
        self.axes.set_xlim([0, self.image.shape[0]])
        self.axes.set_ylim([0, self.image.shape[1]])

    def start(self):
        _ = ani.FuncAnimation(
            self.figure, self.animate, frames=len(self.x_points), interval=0, repeat=False
        )
        plt.show()


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


def get_contours_per_color(image: cv.Mat) -> dict[Colors, tuple[npt.NDArray[np.int32], ...]]:
    contours_per_color: dict[Colors, tuple[npt.NDArray[np.int32]]] = {}
    for color in Colors:
        color_mask = create_mask_for_color(image, color)
        thresh = (color_mask * 255).astype(np.uint8)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours_per_color[color] = contours
    return contours_per_color


def get_trayectories(
    image: cv.Mat, contours_per_color: dict[Colors, tuple[npt.NDArray[np.int32], ...]]
) -> list[Trayectory]:
    height = image.shape[1]

    trayectories: list[Trayectory] = []

    for color in Colors:
        for contour in contours_per_color[color]:
            x_points = contour[:, :, 0]
            y_points = height - contour[:, :, 1]

            trayectories.append(Trayectory(color, x_points, y_points))

    return trayectories


def plot_trayectories(image: cv.Mat, trayectories: list[Trayectory]) -> None:
    for trayectory in trayectories:
        plt.xlim([0, image.shape[0]])
        plt.ylim([0, image.shape[1]])
        plt.axis(False)
        plt.plot(trayectory.x_points, trayectory.y_points, trayectory.color.to_hex())
    plt.show()


def main():
    image = get_image()
    contours_per_color = get_contours_per_color(image)
    trayectories = get_trayectories(image, contours_per_color)
    plot_trayectories(image, trayectories)
    animation = PlotAnimation(image, trayectories)
    animation.start()


if __name__ == "__main__":
    main()
