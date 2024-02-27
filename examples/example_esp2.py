from rapp.motion_controller import ESP301


MOTION_CONTROLLER_PORT = "COM4"
MOTION_CONTROLLER_BAUDRATE = 921600


def main():
    print("Instantiating ESP...")
    # Build HalfWavePlate

    motion_controller = ESP301.build(MOTION_CONTROLLER_PORT, b=MOTION_CONTROLLER_BAUDRATE)
    motion_controller.motor_on(axis=1)
    motion_controller.set_velocity(4, axis=1)
    # motion_controller.set_home(0, axis=1)
    # print("Setting poisition...")
    motion_controller.reset_axis(axis=1)
    # time.sleep(5)
    # motion_controller.set_position(5, axis=1)


if __name__ == '__main__':
    main()
