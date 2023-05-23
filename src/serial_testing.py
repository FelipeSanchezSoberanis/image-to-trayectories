from serial import Serial
from time import sleep


def main():
    serial = Serial("/dev/ttyUSB0", 9600)
    sleep(5)

    byts = 0
    for i in range(100):
        command = "1" if i % 2 == 0 else "0"
        byts += serial.write((command + "\n").encode("utf-8"))  # type: ignore
        print(f"Iteration {i} has {byts} bytes: {round(byts / 64 * 100, 2)}%")


if __name__ == "__main__":
    main()
