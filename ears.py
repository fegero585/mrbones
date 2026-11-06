"""Mr. Bones's ears: listen for the wake word, then hear a question."""
import json
import queue
import time
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

# ---- Ear settings ----
SAMPLE_RATE = 16000       # Vosk wants 16,000 samples per second
MIC_NAME = "respeaker"    # part of the mic's name to look for (not case sensitive)
MIC_CHANNEL = 0           # reSpeaker channel 0 = the cleaned-up voice channel
WAKE_WORD = "mister bones"
WAIT_FOR_QUESTION = 8     # seconds to wait for someone to start talking after waking
MAX_QUESTION = 15         # seconds before we stop listening, even mid-sentence

MODEL_PATH = Path(__file__).parent / "models" / "vosk-model-small-en-us-0.15"

SetLogLevel(-1)           # hide Vosk's chatty log messages
model = Model(str(MODEL_PATH))
audio_queue = queue.Queue()


def _find_mic():
    """Find the reSpeaker by name. Returns (device number, channel count).
    Falls back to the default mic if the reSpeaker isn't plugged in."""
    for i, dev in enumerate(sd.query_devices()):
        if MIC_NAME in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(f"Ears: using {dev['name']}")
            return i, dev["max_input_channels"]
    print("Ears: reSpeaker not found, using the default mic")
    return None, 1


MIC_DEVICE, MIC_CHANNELS = _find_mic()
if MIC_CHANNEL >= MIC_CHANNELS:   # default mic may only have 1 channel
    MIC_CHANNEL = 0


def _mic_callback(indata, frames, time_info, status):
    """sounddevice calls this every time a chunk of mic audio is ready.
    indata has one column per channel - keep only the clean voice channel."""
    audio_queue.put(indata[:, MIC_CHANNEL].tobytes())


stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=4000,       # 4000 samples = a quarter second per chunk
    device=MIC_DEVICE,
    dtype="int16",
    channels=MIC_CHANNELS,
    callback=_mic_callback,
)


def _clear_old_audio():
    """Throw away any audio left over from before we started listening."""
    while not audio_queue.empty():
        audio_queue.get_nowait()


def wait_for_wake_word():
    """Block until someone says 'Hey Mr. Bones'."""
    # A grammar tells Vosk to ONLY listen for these phrases.
    # "[unk]" means "anything else," so random talk doesn't get forced into a match.
    grammar = json.dumps(["hey mister bones", "mister bones", "[unk]"])
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)

    _clear_old_audio()
    stream.start()
    try:
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result())["text"]
            else:
                text = json.loads(recognizer.PartialResult())["partial"]
            if WAKE_WORD in text:
                return
    finally:
        stream.stop()


def listen_for_question():
    """Listen for one sentence and return it as text ('' if nobody spoke)."""
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    heard_something = False
    start = time.time()

    _clear_old_audio()
    stream.start()
    try:
        while True:
            try:
                data = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(data):
                # Vosk decided the person finished talking (they paused)
                text = json.loads(recognizer.Result())["text"]
                if text:
                    return text
            elif json.loads(recognizer.PartialResult())["partial"]:
                heard_something = True

            elapsed = time.time() - start
            if not heard_something and elapsed > WAIT_FOR_QUESTION:
                return ""
            if elapsed > MAX_QUESTION:
                return json.loads(recognizer.FinalResult())["text"]
    finally:
        stream.stop()
