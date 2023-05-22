import serial
from main import Commands, CommandManager, Colors


def main():
    arduino = serial.Serial("/dev/ttyUSB0", 9600)
    for i in range(109):
        command = "ON" if i % 2 == 0 else "OFF"
        arduino.write(command.encode("utf-8"))
        arduino.read_until("DONE".encode("utf-8"))


if __name__ == "__main__":
    main()
