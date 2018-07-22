from enum import Enum
import time
import RPi.GPIO as GPIO


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
        self.isParent = False
        self.in1 = in1
        self.in2 = in2
        self.pwm = pwm
        self.stby = stby

    def initGPIO(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(self.pwm, GPIO.OUT)
        try:
            GPIO.input(self.stby)
            print("STANDBY GPIO is already setup")
        except RuntimeError:
            print("WHEEL SETUP AS PARENT")
            self.isParent = True
            GPIO.setup(self.stby, GPIO.OUT)
            GPIO.output(self.stby, GPIO.HIGH)

    def SHORTBREAK(self):
        high = [self.in1, self.in2, self.pwm]
        low = []
        GPIO.output(high, GPIO.HIGH)

    def CCW(self):
        high = [self.in2, self.pwm]
        low = [self.in1]
        GPIO.output(high, GPIO.HIGH)
        GPIO.output(low, GPIO.LOW)

    def CW(self):
        high = [self.in1, self.pwm]
        low = [self.in2]
        GPIO.output(high, GPIO.HIGH)
        GPIO.output(low, GPIO.LOW)

    def STOP(self):
        high = [self.pwm]
        low = [self.in1, self.in2]
        GPIO.output(high, GPIO.HIGH)
        GPIO.output(low, GPIO.LOW)

    def __STANDBY(self):
        low = [self.in1, self.in2, self.pwm, self.stby]
        GPIO.output(low, GPIO.LOW)

    def cleanup(self):
        if self.isParent:
            self.STOP()
            GPIO.cleanup()
        else:
            self.STOP()


class wheelPosition(Enum):
    wheel1 = "RIGHT_FRONT"
    wheel2 = "LEFT_FRONT"
    wheel3 = "BACK"


class controlWheels(object):
    """
        Interface to control three wheel objects()

    """
    def __init__(self, wheel1, wheel2, wheel3):
        self.wheel1 = wheel1
        self.wheel2 = wheel2
        self.wheel3 = wheel3
        self.timeToMoveAnAngle = None

    def forward(self):
        self.wheel1.CCW()
        self.wheel2.CW()
        self.wheel3.STOP()

    def backward(self):
        self.wheel1.CW()
        self.wheel2.CCW()
        self.wheel3.STOP()
        print("GOING BACKWARD")

    def stop(self):
        self.wheel1.STOP()
        self.wheel2.STOP()
        self.wheel3.STOP()

    def moveHorizontalRight(self):
        self.wheel1.CW()
        self.wheel2.CW()
        self.wheel3.CCW()

    def moveHorizontalLeft(self):
        self.wheel1.CCW()
        self.wheel2.CCW()
        self.wheel3.CW()

    def spinClockWise(self):
        self.wheel1.CW()
        self.wheel2.CW()
        self.wheel3.CW()

    def spinCounterClockWise(self):
        self.wheel1.CCW()
        self.wheel2.CCW()
        self.wheel3.CCW()
    
    def calibrateRotation(self):
        ts = time.time()
        while True:
            try:
                self.spinClockWise()
            except KeyboardInterrupt:
                self.stop()
                now = time.time()
                timeToFullRotation = now - ts
                self.timeToMoveAnAngle = timeToFullRotation / 360
                break

    def moveSomeAngles(self, angle):
        if self.timeToMoveAnAngle is None:
            self.calibrateRotation()
        while True:
            if angle < 0:
                self.spinClockWise()
            else:
                self.spinCounterClockWise()
            time.sleep(self.timeToMoveAnAngle * abs(angle))
            break
        self.stop()
            

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
    


