import sys
import os
import cv2 as cv
import matplotlib.pyplot as plt


def get_image_path() -> str:
    if not len(sys.argv) == 2:
        raise Exception("Image path not provided")

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        raise Exception("File not found")

    return image_path


def main():
    image_path = get_image_path()
    image = cv.imread(image_path)

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
