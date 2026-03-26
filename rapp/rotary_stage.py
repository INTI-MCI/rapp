import time
import math
import logging
from rapp.motion_controller import ESP301Error, ESP301
import rapp.constants as ct

from collections.abc import Iterator
import numpy as np

logger = logging.getLogger(__name__)


class RotaryStageError(Exception):
    pass


class RotaryStage(Iterator):
    """Represents Rotary Stage object. Implements iterator interface.

     Args:
        motion_controller: driver to control the stage.
        cycles: number of cycles to traverse.
        step: step between positions.
        delay_position: time to wait after changing position.
        velocity: velocity of the stage.
        axis: axis in the motion controller.
    """
    def __init__(
        self,
        motion_controller: ESP301,
        cycles=1,
        step=45,
        delay_position=0,
        velocity=4,
        acceleration=4,
        deceleration=4,
        axis=1,
        name=''
    ):
        self._motion_controller = motion_controller
        self.cycles = cycles
        self.step = step
        self._delay_position = delay_position
        self._velocity = velocity
        self._acceleration = acceleration
        self._deceleration = deceleration
        self._axis = axis
        self._name = name

        self._index = 0

        self._prepare_rotations()

    @classmethod
    def build(cls, mock=False, **kwargs):
        if mock:
            return RotaryStageMock(**kwargs)
        return cls(**kwargs)

    def __str__(self):
        return "{} - {}".format(type(self).__name__, self._name)

    def __len__(self):
        return len(self._positions)

    def __next__(self):
        if self._index < len(self._positions):
            position = self._positions[self._index]
            self._index += 1

            try:
                self._motion_controller.set_position(position, axis=self._axis)
            except ESP301Error as e:
                raise RotaryStageError(e)

            time.sleep(self._delay_position)

            return position
        else:
            raise StopIteration

    def _prepare_rotations(self):
        self._positions = self._generate_positions()
        self.motor_on()

        logger.info("{} - Positions: {}.".format(str(self), self._positions))

    def _generate_positions(self):
        end = 360 * math.copysign(1, self.step)

        if self.cycles == 0:
            initial_position = self._motion_controller.get_position(axis=self._axis)
            return [initial_position]

        return np.arange(0, end * self.cycles + self.step, self.step, dtype=float)

    def reset(self):
        """Resets position of the stage."""
        self._motion_controller.set_acceleration(ct.ROTARY_HOME_ACCELERATION, axis=self._axis)
        self._motion_controller.set_deceleration(ct.ROTARY_HOME_DECELERATION, axis=self._axis)
        if self.cycles != 0:
            logger.info("{} - Searching HOME ".format(str(self))
                        + "using velocity= {}  deg/s, ".format(ct.ROTARY_HOME_VELOCITY)
                        + "acceleration= {} deg/s**2, ".format(ct.ROTARY_HOME_ACCELERATION)
                        + "and deceleration= {}  deg/s**2...".format(ct.ROTARY_HOME_DECELERATION))
            self._index = 0
            self._motion_controller.reset_axis(axis=self._axis)
            logger.info("HOME found.")
        self._motion_controller.set_acceleration(self._acceleration, axis=self._axis)
        self._motion_controller.set_deceleration(self._deceleration, axis=self._axis)

    def reconnect(self):
        """Reconnects motion controller."""
        self._motion_controller.reconnect()

    def set_home(self, position):
        """Sets a numeric value to current mechanical position."""
        self._motion_controller.set_home(position, axis=self._axis)

    def close(self):
        """Closes connection to the motion controller."""
        self._motion_controller.close()

    def motor_on(self):
        logger.info("{} - Turning motor ON.".format(str(self)))
        self._motion_controller.motor_on(axis=self._axis)

        logger.info("{} - Setting velocity to {} deg/s.".format(str(self), self._velocity))
        self._motion_controller.set_velocity(self._velocity, axis=self._axis)
        self._motion_controller.set_home_velocity(ct.ROTARY_HOME_VELOCITY, axis=self._axis)

        logger.info(
            "{} - Setting acceleration to {} deg/s**2.".format(str(self), self._acceleration))
        self._motion_controller.set_acceleration(self._acceleration, axis=self._axis)

        logger.info("{} - Setting deceleration to {} deg/s**2.".format(
            str(self), self._deceleration))
        self._motion_controller.set_deceleration(self._deceleration, axis=self._axis)

        self._motion_controller.check_errors()

    def info_current_position(self):
        return f"{self._name} angle: {self.current_position()}°, "

    def current_position_for_filename(self):
        return self.current_position()

    def current_position(self):
        if self._index == 0:
            return self._motion_controller.get_position(axis=self._axis)
        elif 1 <= self._index <= len(self._positions):
            self._positions[self._index - 1]
        else:
            raise ValueError("Index out of range.")


class RotaryStageMock(RotaryStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_rotations(self):
        self._positions = [self._motion_controller.get_position(axis=self._axis)]

    def __next__(self):
        if self._index < len(self._positions):
            position = self._positions[self._index]
            self._index += 1

            return position
        else:
            raise StopIteration

    def reset(self):
        self._index = 0

    def set_home(self, position):
        logger.info("set_home ignored.")

    def motor_on(self):
        logger.info("motor_on ignored.")

    def current_position_for_filename(self):
        return None
