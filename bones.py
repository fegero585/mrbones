import re
import subprocess
import time
import wave
from datetime import datetime

import numpy as np
import anthropic
from piper import PiperVoice
from rpi_hardware_pwm import HardwarePWM

# ---- Voice settings ----
PITCH = "-300"    # lower = deeper
TEMPO = "0.85"    # lower = slower

# ---- Jaw settings ----
CLOSED = 9.75         # jaw shut
OPEN = 6.5            # jaw open
STEP = 0.1            # seconds per jaw move (gives the servo time to open AND close)
THRESHOLD = 0.2       # quieter than this = mouth closed (silence between words)
GAIN = 1.5            # higher = mouth opens wider on normal speech
MIN_OPEN = 0.6        # every syllable opens the mouth at least this much (0.0 to 1.0)
MAX_OPEN_STEPS = 2    # force the jaw closed after this many open steps in a row
LOCAL_WINDOW = 4      # how many steps around each moment count as "nearby"
JAW_TIMING = 0.2      # jaw late? raise it (0.3). jaw early? lower it (0.1, 0.0)
DEBUG = True          # print jaw info for each reply

# ---- Radio stations ----
# key: (station name, stream address)
STATIONS = {
    "doomed": ("Doomed, the Halloween special", "https://somafm.com/specials.pls"),
    "classicrock": ("Classic Rock", "https://cast1.torontocast.com:4610/stream"),
    "darkzone": ("The Dark Zone", "https://somafm.com/darkzone.pls"),
    "dronezone": ("Drone Zone", "https://somafm.com/dronezone.pls"),
    "secretagent": ("Secret Agent", "https://somafm.com/secretagent.pls"),
    "metal": ("Metal Detector", "https://somafm.com/metal.pls"),
    "seventies": ("Left Coast 70s", "https://somafm.com/seventies.pls"),
    "groovesalad": ("Groove Salad", "https://somafm.com/groovesalad.pls"),
    "indiepop": ("Indie Pop Rocks", "https://somafm.com/indiepop.pls"),
    "illstreet": ("Illinois Street Lounge", "https://somafm.com/illstreet.pls"),
}

# ---- Tools Claude can use ----
TOOLS = [
    {
        "name": "play_radio",
        "description": (
            "Play an internet radio station on Mr. Bones's speaker. "
            "Use this whenever the user asks for music. Stations: "
            "doomed = Halloween dark industrial, the best pick for spooky music; "
            "classicrock = classic rock like Pink Floyd and Led Zeppelin; "
            "darkzone = dark, spooky ambient; dronezone = eerie atmospheric; "
            "secretagent = mysterious spy lounge; metal = heavy metal; "
            "seventies = mellow 1970s rock; groovesalad = chill downtempo; "
            "indiepop = indie pop; illstreet = vintage lounge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "enum": list(STATIONS.keys())}
            },
            "required": ["station"],
        },
    },
    {
        "name": "stop_music",
        "description": "Stop the music that is playing.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

client = anthropic.Anthropic()
voice = PiperVoice.load("en_US-ryan-medium.onnx")
jaw = HardwarePWM(pwm_channel=0, hz=50, chip=0)

SYSTEM_PROMPT = (
    "You are Mr. Bones, a spooky-but-fun talking skeleton. "
    "You speak with playful creepiness and lots of bone puns, "
    "but you are always friendly and never truly scary for kids. "
    "Keep answers short (1-3 sentences) because they will be spoken aloud. "
    "Only write the words you say out loud. Never describe actions or sounds, "
    "and never use asterisks, emojis, or special formatting. "
    "You can play internet radio with the play_radio tool and stop it "
    "with the stop_music tool. When you start music, say one short line "
    "introducing it."
)

music_player = None    # the running mpv program, if music is playing
next_station = None    # station to start after Mr. Bones finishes talking


def clean_text(text):
    """Remove anything that shouldn't be spoken, like *actions*."""
    text = re.sub(r"\*.+?\*", "", text)
    text = text.replace("*", "")
    return " ".join(text.split())


def loudness_levels(wav_path):
    """Return a list of loudness values (0.0 to 1.0), one per STEP of audio."""
    with wave.open(wav_path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    chunk = int(rate * STEP)
    levels = []
    for i in range(0, len(samples), chunk):
        part = samples[i:i + chunk]
        if len(part) > 0:
            levels.append(float(np.sqrt(np.mean(part ** 2))))
        else:
            levels.append(0.0)
    if levels and max(levels) > 0:
        peak = float(np.percentile(levels, 90)) or max(levels)
    else:
        peak = 1.0
    return [level / peak for level in levels]


def plan_jaw(levels):
    """Turn loudness into a list of jaw openings (0.0 = shut, 1.0 = fully open)."""
    plan = []
    open_run = 0
    for i, loud in enumerate(levels):
        lo = max(0, i - LOCAL_WINDOW)
        hi = min(len(levels), i + LOCAL_WINDOW + 1)
        local_avg = sum(levels[lo:hi]) / (hi - lo)

        is_syllable = loud >= THRESHOLD and loud >= local_avg * 0.9

        if is_syllable and open_run < MAX_OPEN_STEPS:
            amount = max(MIN_OPEN, min(1.0, loud * GAIN) ** 0.5)
            plan.append(amount)
            open_run += 1
        else:
            plan.append(0.0)
            open_run = 0
    return plan


def play_with_jaw(wav_path):
    """Play a sound and move the jaw along with its syllables."""
    plan = plan_jaw(loudness_levels(wav_path))
    look_ahead = max(0, int(round(JAW_TIMING / STEP)))

    if DEBUG:
        opened = sum(1 for amount in plan if amount > 0)
        print(f"[jaw] {len(plan)} steps, {opened} open")

    player = subprocess.Popen(["pw-play", wav_path])
    if JAW_TIMING < 0:
        time.sleep(-JAW_TIMING)

    start = time.time()
    for i in range(len(plan)):
        index = min(i + look_ahead, len(plan) - 1)
        jaw.change_duty_cycle(CLOSED + (OPEN - CLOSED) * plan[index])
        wait = start + (i + 1) * STEP - time.time()
        if wait > 0:
            time.sleep(wait)
    player.wait()
    jaw.change_duty_cycle(CLOSED)


def speak(text):
    """Make speech with Piper, lower the pitch and slow it with sox, then talk."""
    text = clean_text(text)
    if not text:
        return
    raw_path = "/tmp/bones_raw.wav"
    voice_path = "/tmp/bones.wav"

    with wave.open(raw_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    subprocess.run(["sox", raw_path, voice_path,
                    "pitch", PITCH, "tempo", TEMPO, "norm", "-1"])

    play_with_jaw(voice_path)


def start_radio(station):
    """Start a radio station playing in the background."""
    global music_player
    stop_music()
    name, url = STATIONS[station]
    print(f"[radio] playing {name}")
    music_player = subprocess.Popen(["mpv", "--no-video", "--no-terminal", url],
    stdin=subprocess.DEVNULL,
    )


def stop_music():
    """Stop the radio if it's playing."""
    global music_player
    if music_player is not None and music_player.poll() is None:
        music_player.terminate()
        music_player.wait()
    music_player = None


def run_tool(name, tool_input):
    """Do what Claude asked for and return a short note about what happened."""
    global next_station
    if name == "play_radio":
        station = tool_input.get("station")
        if station not in STATIONS:
            return f"Unknown station. Choose from: {', '.join(STATIONS)}."
        next_station = station
        return f"{STATIONS[station][0]} will start right after you finish talking."
    if name == "stop_music":
        stop_music()
        return "The music is stopped."
    return f"Unknown tool: {name}"


def ask_claude(user_text):
    """Send a message to Claude, run any tools it asks for, and return its reply."""
    history.append({"role": "user", "content": user_text})

    while True:
        now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=300,
            system=SYSTEM_PROMPT + f" The current date and time is {now}.",
            tools=TOOLS,
            messages=history,
        )
        history.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": run_tool(block.name, block.input),
                })
        history.append({"role": "user", "content": results})

    reply = "".join(block.text for block in response.content if block.type == "text")
    return clean_text(reply)


history = []
jaw.start(CLOSED)

try:
    print("Mr. Bones is awake! Type 'quit' to exit.\n")
    speak("Mister Bones has risen!")

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() == "quit":
            stop_music()
            print("Mr. Bones: Bone voyage!")
            speak("Bone voyage!")
            break

        reply = ask_claude(user_text)
        print(f"Mr. Bones: {reply}\n")
        speak(reply)

        if next_station:
            start_radio(next_station)
            next_station = None
finally:
    stop_music()
    jaw.stop()
