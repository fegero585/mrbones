"""Test the jaw by itself: open and close it 5 times."""
import time
from rpi_hardware_pwm import HardwarePWM

CLOSED = 9.4    # same values as bones.py
OPEN = 7.9

jaw = HardwarePWM(pwm_channel=0, hz=50, chip=0)
jaw.start(CLOSED)
try:
    for i in range(5):
        print("open")
        jaw.change_duty_cycle(OPEN)
        time.sleep(0.5)
        print("closed")
        jaw.change_duty_cycle(CLOSED)
        time.sleep(0.5)
finally:
    jaw.stop()
    print("done")

