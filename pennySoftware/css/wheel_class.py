from enum import Enum
import time


class wheel(object):
    """
        Class to setup and control a wheel
    """
    def __init__(self, in1, in2, pwm, stby):
        """
            @param : in1 GPIO number for in1,
                     in2 GPIO number for in2,
                     pwm GPIO number for pwm,
                     stby GPIO number for stby
        """
        self.in1 = in1
        self.in2 = in2
        self.pwm = pwm
        self.stby = stby

    def initGPIO():
        pass

    def SHORTBREAK(self):
        pass

    def CCW(self):
        pass

    def CW(self):
        pass

    def STOP(self):
        pass

    def STANDBY(self):
        pass


class wheelPosition(Enum):
    wheel1 = "RIGHT_FRONT"
    wheel2 = "LEFT_FRONT"
    wheel3 = "BACK"


class controlWheels(object):
    """
        Interface to control three wheel objects

    """
    def __init__(self, wheel1, wheel2, wheel3):
        self.wheel1 = wheel1
        self.wheel2 = wheel2
        self.wheel3 = wheel3

    def forward(self):
        self.wheel1.CCW
        self.wheel2.CW
        self.wheel3.STOP

    def backward(self):
        self.wheel1.CW
        self.wheel2.CCW
        self.wheel3.STOP

    def stop(self):
        self.wheel1.stop
        self.wheel2.stop
        self.wheel3.stop

    def moveHorizontalRight(self):
        self.wheel1.CW
        self.wheel2.CW
        self.wheel3.CCW

    def moveHorizontalLeft(self):
        self.wheel1.CCW
        self.wheel2.CCW
        self.wheel3.CW

    def spinClockWise(self):
        self.wheel1.CW
        self.wheel2.CW
        self.wheel3.CW

    def spinCounterClockWise(self):
        self.wheel1.CCW
        self.wheel2.CCW
        self.wheel3.CCW


class controlPenny(controlWheels):

    def __init__(self, wheel1, wheel2, wheel3):
        super(controlPenny, self).__init__(
            wheel1,
            wheel2,
            wheel3
        )

    def goForwardNsec(self, seconds):
        ts = time.timestamp()
        while True:
            self.forward()
            now = time.timestamp()
            if ts - now > 5:
                self.stop()
                break
    




















































