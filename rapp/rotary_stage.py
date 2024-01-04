import time
import logging
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)


class RotaryStage(Iterator):
    """Represents an iterator Rotary Stage.

     Args:
        motion_controller: driver to control the rotary stage.
        cycles: number of cycles to trasverse.
        step: step between positions.
        init_position: initial position.
        delay_position: time to wait after changing position.
        velocity: velocity of the analyzer.
        axis: axis in the motion controller.
    """
    def __init__(
        self, motion_controller, cycles=1, step=45, init_position=None,
        delay_position=1, velocity=4, axis=1
    ):
        self._motion_controller = motion_controller
        self.cycles = cycles
        self.step = step
        self.init_position = init_position
        self._delay_position = delay_position
        self._velocity = velocity
        self._axis = axis

        self._motor_on()

        self._positions = self.generate_positions()
        logger.info("{} - Positions: {}.".format(str(self), self._positions))

        self._index = 0

    def __str__(self):
        return type(self).__name__

    def __len__(self):
        return len(self._positions)

    def __next__(self):
        if self._index < len(self._positions):
            position = self._positions[self._index]
            self._index += 1

            logger.debug("{} - Changing position to: {}°.".format(str(self), position))
            self._motion_controller.set_position(position, axis=self._axis)
            time.sleep(self._delay_position)

            return position
        else:
            raise StopIteration

    def generate_positions(self):
        positions = np.arange(0, 360 * self.cycles + self.step, self.step, dtype=float)
        if self.init_position is not None:
            positions += self.init_position

        return positions

    def reset(self):
        self._index = 0

    def close(self):
        self._motion_controller.close()

    def _motor_on(self):
        logger.info("{} - Turning motor ON.".format(str(self)))
        self._motion_controller.motor_on(axis=self._axis)

        logger.info("{} - Setting velocity to {} deg/s.".format(str(self), self._velocity))
        self._motion_controller.set_velocity(self._velocity)
        self._motion_controller.set_home_velocity(self._velocity)

        if self.init_position is None:
            logger.info("{} - Setting initial position as current position.".format(str(self)))
            self.init_position = self._motion_controller.get_position()


class Analyzer(RotaryStage):
    pass


class HalfWavePlate(RotaryStage):
    pass
