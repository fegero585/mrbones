from rpi_hardware_pwm import HardwarePWM

# Safe range for the servo. Don't go past these.
MIN_POS = 5.0
MAX_POS = 12.0

servo = HardwarePWM(pwm_channel=0, hz=50, chip=0)
position = 7.5
servo.start(position)

print("Jaw tuner: type a number between 5.0 and 10.0 and press Enter.")
print("Type 'q' to quit.\n")

try:
    while True:
        entry = input(f"Current {position} -> new position: ")
        if entry.strip().lower() == "q":
            break
        try:
            new_pos = float(entry)
        except ValueError:
            print("Please type a number, like 8.5")
            continue
        if MIN_POS <= new_pos <= MAX_POS:
            position = new_pos
            servo.change_duty_cycle(position)
        else:
            print(f"Stay between {MIN_POS} and {MAX_POS}")
finally:
    servo.stop()
    print(f"Done. Last position was {position}")
