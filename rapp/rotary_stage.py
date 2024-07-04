import time
import math
import logging
from rapp.motion_controller import ESP301Error

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
        motion_controller,
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

        self._positions = self._generate_positions()
        self.motor_on()

        logger.info("{} - Positions: {}.".format(str(self), self._positions))

        self._index = 0

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

    def _generate_positions(self):
        end = 360 * math.copysign(1, self.step)

        if self.cycles == 0:
            return [0]

        return np.arange(0, end * self.cycles + self.step, self.step, dtype=float)

    def reset(self):
        """Resets position of the stage."""
        logger.info("{} - Searching HOME...".format(str(self)))
        self._index = 0
        self._motion_controller.reset_axis(axis=self._axis)
        logger.info("HOME found.")

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
        self._motion_controller.set_home_velocity(self._velocity, axis=self._axis)

        logger.info(
            "{} - Setting acceleration to {} deg/s**2.".format(str(self), self._acceleration))
        self._motion_controller.set_acceleration(self._acceleration, axis=self._axis)

        logger.info("{} - Setting deceleration to {} deg/s**2.".format(
            str(self), self._deceleration))
        self._motion_controller.set_deceleration(self._deceleration, axis=self._axis)
