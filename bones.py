import argparse
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import anthropic
from piper import PiperVoice
from rpi_hardware_pwm import HardwarePWM

import ears

# ---- Voice settings ----
PITCH = "-300"        # lower = deeper
TEMPO = "0.85"        # lower = slower
BASS = "+8"           # higher = more low-end rumble
OVERDRIVE = "4"       # higher = more growl (0 to 10)

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

# ---- Eye settings ----
EYES_PIN = 23         # GPIO pin the eye LEDs are on

# ---- Input settings ----
USE_MIC = True        # False = type to Mr. Bones instead of talking

# ---- Weather settings ----
HOME_CITY = "Rochester, New York"
HOME_LAT = 43.1566
HOME_LON = -77.6088

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
    {
        "name": "get_weather",
        "description": (
            "Get the current weather and a 3-day forecast. Use this whenever "
            "the user asks about weather, temperature, rain, snow, or what to wear. "
            "Leave city out for home (Rochester, New York). For another place, "
            "give just the city name, like 'Lander'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name only, no state. Leave out for home.",
                }
            },
        },
    },
]

# ---- Personality (prompt library) ----
# Each .txt file in the prompts folder is a personality.
# Pick one when starting:  python bones.py -p apocalypse
PROMPTS_FOLDER = Path(__file__).parent / "prompts"
PROMPT_NAMES = sorted(file.stem for file in PROMPTS_FOLDER.glob("*.txt"))

parser = argparse.ArgumentParser(description="Run Mr. Bones, the talking skeleton.")
parser.add_argument(
    "-p", "--prompt",
    choices=PROMPT_NAMES,
    default="normal",
    help="which personality file to use from the prompts folder (default: normal)",
)
args = parser.parse_args()

PERSONALITY = (PROMPTS_FOLDER / f"{args.prompt}.txt").read_text().strip()

# Required: always added, because everything he says is spoken out loud.
SPEAKING_RULES = (
    "Keep answers short (1-3 sentences) because they will be spoken aloud. "
    "Only write the words you say out loud. Never describe actions or sounds, "
    "and never use asterisks, emojis, lists, or special formatting. "
    "You can play internet radio with the play_radio tool and stop it "
    "with the stop_music tool. When you start music, say one short line "
    "introducing it. You can check the weather with the get_weather tool."
)

SYSTEM_PROMPT = PERSONALITY + " " + SPEAKING_RULES

client = anthropic.Anthropic()
voice = PiperVoice.load("en_US-ryan-medium.onnx")
jaw = HardwarePWM(pwm_channel=0, hz=50, chip=0)

music_player = None    # the running mpv program, if music is playing
next_station = None    # station to start after Mr. Bones finishes talking
history = []           # the conversation so far


# ================= Eyes =================

def eyes(on):
    """Turn the eye LEDs on or off."""
    level = "dh" if on else "dl"
    subprocess.run(["pinctrl", "set", str(EYES_PIN), "op", level])


# ================= Talking and the jaw =================

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
    """Make speech with Piper, then make it deep, slow and gritty with sox, then talk."""
    text = clean_text(text)
    if not text:
        return
    raw_path = "/tmp/bones_raw.wav"
    voice_path = "/tmp/bones.wav"

    with wave.open(raw_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    subprocess.run(["sox", raw_path, voice_path,
                    "pitch", PITCH,
                    "tempo", TEMPO,
                    "bass", BASS,
                    "overdrive", OVERDRIVE,
                    "gain", "-n", "-2"], check=True)

    play_with_jaw(voice_path)


# ================= Music =================

def start_radio(station):
    """Start a radio station playing in the background."""
    global music_player
    stop_music()
    name, url = STATIONS[station]
    print(f"[radio] playing {name}")
    music_player = subprocess.Popen(
        ["mpv", "--no-video", "--no-terminal", url],
        stdin=subprocess.DEVNULL,
    )


def stop_music():
    """Stop the radio if it's playing."""
    global music_player
    if music_player is not None and music_player.poll() is None:
        music_player.terminate()
        music_player.wait()
    music_player = None


# ================= Weather =================

# Open-Meteo sends a number for the weather; this turns it into words.
WEATHER_WORDS = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


def fetch_json(url, params):
    """Download a web address and turn the JSON reply into a Python dictionary."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full_url, timeout=10) as response:
        return json.loads(response.read())


def get_weather(city=None):
    """Return the current weather and a 3-day forecast as plain sentences."""
    # Step 1: figure out where
    if city:
        found = fetch_json("https://geocoding-api.open-meteo.com/v1/search",
                           {"name": city, "count": 1})
        if not found.get("results"):
            return f"Could not find a place called {city}."
        place = found["results"][0]
        name = f"{place['name']}, {place.get('admin1', place.get('country', ''))}"
        lat, lon = place["latitude"], place["longitude"]
    else:
        name, lat, lon = HOME_CITY, HOME_LAT, HOME_LON

    # Step 2: get the weather for that spot
    data = fetch_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "forecast_days": 3,
    })

    # Step 3: turn the numbers into sentences
    now = data["current"]
    sentences = [
        f"Weather for {name}.",
        f"Right now it is {round(now['temperature_2m'])} degrees, "
        f"feels like {round(now['apparent_temperature'])}, "
        f"{WEATHER_WORDS.get(now['weather_code'], 'unknown conditions')}, "
        f"with wind at {round(now['wind_speed_10m'])} miles per hour.",
    ]

    daily = data["daily"]
    for i, date in enumerate(daily["time"]):
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
        chance = daily["precipitation_probability_max"][i] or 0
        sentences.append(
            f"{day_name}: high of {round(daily['temperature_2m_max'][i])}, "
            f"low of {round(daily['temperature_2m_min'][i])}, "
            f"{WEATHER_WORDS.get(daily['weather_code'][i], 'unknown conditions')}, "
            f"{chance} percent chance of rain or snow."
        )

    return " ".join(sentences)


# ================= Claude =================

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
    if name == "get_weather":
        try:
            return get_weather(tool_input.get("city"))
        except Exception as error:
            return f"The weather service could not be reached: {error}"
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


# ================= Main program =================

jaw.start(CLOSED)
eyes(True)

try:
    print(f"[prompt] using prompts/{args.prompt}.txt")
    print("Mr. Bones is awake! Say 'Hey Mister Bones' (Ctrl+C to exit).\n")
    speak("Mister Bones has risen!")

    while True:
        if USE_MIC:
            ears.wait_for_wake_word()
            print("[ears] wake word heard")
            speak("Yes?")
            user_text = ears.listen_for_question()
            if not user_text:
                speak("Speak up, mortal. I have no ears.")
                continue
            print(f"You: {user_text}")
        else:
            user_text = input("You: ")

        if user_text.strip().lower() in ("quit", "go to sleep"):
            stop_music()
            print("Mr. Bones: Bone voyage!")
            speak("Bone voyage!")
            break

        saved = len(history)    # remember where the conversation was
        try:
            reply = ask_claude(user_text)
        except anthropic.APIError as error:
            print(f"[error] {error}")
            del history[saved:]   # forget the half-finished exchange
            reply = "My skull is foggy. Ask me again in a moment."

        print(f"Mr. Bones: {reply}\n")
        speak(reply)

        if next_station:
            start_radio(next_station)
            next_station = None
finally:
    stop_music()
    eyes(False)
    jaw.stop()
