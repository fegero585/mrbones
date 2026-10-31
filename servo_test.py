import time
from rpi_hardware_pwm import HardwarePWM

# Servo positions as "duty cycle" percentages at 50 Hz:
#   5.0 = one end, 7.5 = middle, 10.0 = other end
CLOSED = 11.0
OPEN = 8.0

# Channel 0 is GPIO18 on a Pi 4
servo = HardwarePWM(pwm_channel=0, hz=50, chip=0)
servo.start(CLOSED)
time.sleep(1)

print("Opening and closing the jaw 5 times...")
for i in range(5):
    servo.change_duty_cycle(OPEN)
    time.sleep(0.4)
    servo.change_duty_cycle(CLOSED)
    time.sleep(0.4)

servo.stop()
print("Done!")
