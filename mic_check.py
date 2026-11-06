import wave
import sounddevice as sd
import numpy as np

RATE = 16000      # Vosk wants 16 kHz
SECONDS = 5

# Find the reSpeaker by name instead of hard-coding a device number
device_index = None
for i, dev in enumerate(sd.query_devices()):
    print(i, dev["name"], "inputs:", dev["max_input_channels"])
    if "respeaker" in dev["name"].lower() and dev["max_input_channels"] > 0:
        device_index = i

if device_index is None:
    raise SystemExit("reSpeaker not found - check the list above for its name")

channels = sd.query_devices(device_index)["max_input_channels"]
print(f"\nUsing device {device_index} with {channels} channel(s). Talk now...")

audio = sd.rec(SECONDS * RATE, samplerate=RATE, channels=channels,
               dtype="int16", device=device_index)
sd.wait()

# Save each channel to its own file so you can compare them
for ch in range(channels):
    filename = f"mic_ch{ch}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio[:, ch].tobytes())
    print("Saved", filename)
