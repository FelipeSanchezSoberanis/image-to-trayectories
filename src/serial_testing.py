from main import Colors, CommandManager


def main():
    cm = CommandManager("/dev/ttyUSB0", 9600)
    cm.move_to(100, 200)
    cm.move_to(300, 500)
    cm.change_color(Colors.RED)
    cm.change_color(Colors.BLUE)
    cm.tool_up()
    cm.tool_down()


if __name__ == "__main__":
    main()
