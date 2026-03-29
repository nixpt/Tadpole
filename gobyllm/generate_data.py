"""
Generate training data for EdgeSense LLM.

Every sample is procedurally generated — no fixed example lists.
Randomizes: entities, values, phrasing, thinking style, response style,
tool combinations, system prompts.

Skills trained:
1. Natural language → tool call (any domain)
2. Read arbitrary tool schemas
3. Step-by-step reasoning
4. Conversational ability
5. Knowing when NOT to act
"""

import json
import random
import os
from collections import Counter

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS — combinatorial parts for generating unique samples
# ══════════════════════════════════════════════════════════════════════════════

def T(name, desc, params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        },
    }


def pick(lst):
    return random.choice(lst)


def pick_n(lst, n):
    return random.sample(lst, min(n, len(lst)))


# ── Entities ────────────────────────────────────────────────────────────────

ROOMS = ["living room", "bedroom", "kitchen", "bathroom", "office", "hallway",
         "garage", "basement", "attic", "dining room", "nursery", "guest room",
         "master bedroom", "laundry room", "patio", "porch", "den", "study"]

ZONES = ["zone A", "zone B", "zone C", "zone D", "north wing", "south wing",
         "east wing", "west wing", "building 1", "building 2", "floor 1",
         "floor 2", "floor 3", "main hall", "reception", "loading dock"]

INDUSTRIAL_LOCATIONS = ["server room", "warehouse", "lab", "clean room",
                        "control room", "workshop", "storage", "boiler room",
                        "compressor room", "electrical room", "rooftop",
                        "greenhouse", "cold storage", "paint booth"]

DEVICES = {
    "lights": ["lights", "lamp", "light strip", "chandelier", "overhead light", "desk lamp", "floor lamp"],
    "fans": ["fan", "ceiling fan", "exhaust fan", "desk fan", "tower fan"],
    "heaters": ["heater", "space heater", "radiator", "heat lamp", "floor heater"],
    "coolers": ["AC", "air conditioner", "cooler", "mini split", "window unit"],
    "speakers": ["speaker", "smart speaker", "soundbar", "bedroom speaker", "kitchen speaker"],
    "doors": ["front door", "back door", "garage door", "side door", "gate"],
    "blinds": ["blinds", "curtains", "shades", "shutters"],
    "appliances": ["coffee maker", "oven", "dishwasher", "washing machine", "dryer", "robot vacuum", "dehumidifier", "humidifier", "air purifier"],
}

SENSOR_TYPES = {
    "temperature": {"unit": "°C", "units_alt": ["celsius", "degrees", "°C"], "normal": (18, 26), "warn": (10, 35), "crit": (-5, 50)},
    "humidity": {"unit": "%", "units_alt": ["percent", "%", "% humidity"], "normal": (30, 60), "warn": (15, 80), "crit": (5, 95)},
    "co2": {"unit": "ppm", "units_alt": ["ppm", "parts per million"], "normal": (350, 800), "warn": (800, 1500), "crit": (1500, 5000)},
    "noise": {"unit": "dB", "units_alt": ["dB", "decibels"], "normal": (20, 55), "warn": (55, 80), "crit": (80, 120)},
    "light_level": {"unit": "lux", "units_alt": ["lux"], "normal": (200, 500), "warn": (30, 800), "crit": (0, 2000)},
    "pressure": {"unit": "hPa", "units_alt": ["hPa", "millibars", "mbar"], "normal": (1000, 1025), "warn": (980, 1040), "crit": (950, 1060)},
    "gas": {"unit": "ppm", "units_alt": ["ppm"], "normal": (0, 10), "warn": (10, 50), "crit": (50, 500)},
    "vibration": {"unit": "mm/s", "units_alt": ["mm/s"], "normal": (0, 2.5), "warn": (2.5, 7), "crit": (7, 30)},
    "voltage": {"unit": "V", "units_alt": ["volts", "V"], "normal": (220, 240), "warn": (200, 250), "crit": (180, 270)},
    "battery": {"unit": "%", "units_alt": ["percent", "%"], "normal": (50, 100), "warn": (15, 50), "crit": (0, 15)},
    "water_level": {"unit": "cm", "units_alt": ["cm", "centimeters"], "normal": (10, 50), "warn": (5, 80), "crit": (0, 120)},
    "soil_moisture": {"unit": "%", "units_alt": ["percent", "%"], "normal": (40, 70), "warn": (15, 85), "crit": (5, 100)},
    "wind_speed": {"unit": "km/h", "units_alt": ["km/h", "kph"], "normal": (0, 25), "warn": (25, 60), "crit": (60, 150)},
    "motion": {"unit": "events/hr", "units_alt": ["events", "events per hour"], "normal": (0, 5), "warn": (5, 20), "crit": (20, 100)},
    "pm25": {"unit": "µg/m³", "units_alt": ["µg/m³", "ug/m3"], "normal": (0, 12), "warn": (12, 35), "crit": (35, 300)},
}

SENSOR_RISKS = {
    "temperature": {"high": "overheating, equipment damage, discomfort", "low": "freezing pipes, hypothermia, equipment malfunction"},
    "humidity": {"high": "mold, condensation, corrosion", "low": "static, dry skin, respiratory irritation"},
    "co2": {"high": "drowsiness, headaches, poor concentration", "low": "unusual, check sensor"},
    "noise": {"high": "hearing damage, stress", "low": "check sensor"},
    "light_level": {"high": "glare, eye strain", "low": "eye strain, safety hazard"},
    "gas": {"high": "explosion risk, poisoning — life safety threat", "low": "not applicable"},
    "vibration": {"high": "bearing failure, equipment damage", "low": "not typical"},
    "voltage": {"high": "equipment damage, fire risk", "low": "brownout, equipment malfunction"},
    "battery": {"high": "not applicable", "low": "device shutdown, data loss"},
    "water_level": {"high": "overflow, flooding", "low": "dry run damage, insufficient supply"},
    "soil_moisture": {"high": "root rot, waterlogging", "low": "plant stress, wilting"},
    "wind_speed": {"high": "structural damage, flying debris", "low": "not typical"},
    "motion": {"high": "security concern, unauthorized access", "low": "sensor may be blocked"},
    "pressure": {"high": "over-pressurization", "low": "vacuum, cavitation"},
    "pm25": {"high": "respiratory issues, poor air quality", "low": "not typical"},
}

PEOPLE = ["John", "Sarah", "Mike", "Emma", "David", "Lisa", "Tom", "Anna",
          "James", "Maria", "Alex", "Chris", "Sam", "Pat", "Jordan", "Taylor"]

TIMES = ["7:00 AM", "7:30 AM", "8:00 AM", "8:15 AM", "9:00 AM", "10:00 AM",
         "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "2:30 PM", "3:00 PM",
         "4:00 PM", "5:00 PM", "6:00 PM", "6:30 PM", "7:00 PM", "8:00 PM",
         "9:00 PM", "10:00 PM", "10:30 PM", "11:00 PM"]

DURATIONS = ["5 minutes", "10 minutes", "15 minutes", "20 minutes", "30 minutes",
             "45 minutes", "1 hour", "2 hours", "3 hours"]

MUSIC_GENRES = ["jazz", "classical", "rock", "pop", "lo-fi", "ambient", "blues",
                "country", "electronic", "hip hop", "R&B", "folk", "indie"]

TASKS_TO_REMEMBER = ["call the plumber", "check the sensors", "water the plants",
                     "pick up groceries", "submit the report", "feed the dog",
                     "take out the trash", "pay the electric bill", "order supplies",
                     "schedule maintenance", "review the logs", "update firmware",
                     "replace batteries", "clean the filters", "test the backup generator"]

PUMP_IDS = [f"pump_{i}" for i in range(1, 13)]
VALVE_IDS = [f"valve_{i}" for i in range(1, 16)]
FAN_IDS = [f"fan_{i}" for i in range(1, 10)] + [f"{z}_fan" for z in ["office", "warehouse", "lab", "server_room", "kitchen"]]
SENSOR_IDS = [f"sensor_{i:03d}" for i in range(1, 80)]

BRIGHTNESS_LEVELS = list(range(5, 100, 5)) + [1, 15, 25, 42, 67, 73, 88]
TEMPERATURES = [t / 2 for t in range(32, 56)]  # 16.0 to 27.5 in 0.5 steps

# ── Phrasing variants ──────────────────────────────────────────────────────

TURN_ON_PHRASES = [
    "Turn on the {device} in the {room}",
    "Can you turn on the {room} {device}?",
    "Switch on the {device} in the {room}",
    "Please turn on the {device} in the {room}",
    "Enable the {room} {device}",
    "I need the {device} on in the {room}",
    "Start the {device} in the {room}",
    "Power on the {room} {device}",
    "Hey, turn the {room} {device} on",
    "Could you switch the {device} on in the {room}?",
]

TURN_OFF_PHRASES = [
    "Turn off the {device} in the {room}",
    "Can you turn off the {room} {device}?",
    "Switch off the {device} in the {room}",
    "Please shut off the {device} in the {room}",
    "Disable the {room} {device}",
    "Kill the {device} in the {room}",
    "I don't need the {device} in the {room} anymore",
    "Shut down the {room} {device}",
    "Turn the {room} {device} off please",
    "Power off the {room} {device}",
]

BRIGHTNESS_PHRASES = [
    "Set the {room} lights to {level}%",
    "Dim the {room} lights to {level} percent",
    "Make the {room} lights {level}%",
    "Set {room} light brightness to {level}",
    "Change the {room} lights to {level}%",
    "I want the {room} lights at {level}%",
    "Put the {room} lights on {level} percent",
    "Adjust {room} lighting to {level}%",
]

TEMP_PHRASES = [
    "Set the temperature to {temp} degrees",
    "Make it {temp} degrees in here",
    "Set thermostat to {temp}",
    "I want it {temp} degrees",
    "Can you set the temperature to {temp}?",
    "Change the temperature to {temp} celsius",
    "Adjust the thermostat to {temp}",
    "Set heating to {temp} degrees",
]

IMPLICIT_HOT_PHRASES = [
    "It's too hot in here",
    "It's really warm in the {room}",
    "I'm sweating, it's boiling in here",
    "Can you cool down the {room}?",
    "The {room} is way too warm",
    "It's stuffy in here",
    "I'm overheating",
    "This room is like an oven",
    "Way too warm in the {room}",
    "I need it cooler in here",
]

IMPLICIT_COLD_PHRASES = [
    "It's freezing in here",
    "It's really cold in the {room}",
    "I'm cold, can you warm it up?",
    "The {room} is too chilly",
    "It's way too cold in here",
    "Brrr, it's cold",
    "Can you heat up the {room}?",
    "I need more warmth in here",
    "This room is an icebox",
    "I'm shivering, turn up the heat",
]

IMPLICIT_DARK_PHRASES = [
    "It's too dark in here",
    "I can't see anything in the {room}",
    "The {room} is really dim",
    "It's dark in the {room}",
    "I need more light in the {room}",
    "Can you brighten up the {room}?",
    "It's gloomy in here",
    "The lighting is terrible in the {room}",
]

IMPLICIT_BRIGHT_PHRASES = [
    "The lights are too bright in the {room}",
    "It's way too bright in here",
    "Can you dim the {room} a bit?",
    "The {room} lights are blinding",
    "Too much light in the {room}",
    "Tone down the lights in the {room}",
]

CONFIRM_RESPONSES = [
    "Done.", "Got it.", "All set.", "Sure thing.",
    "On it.", "You got it.", "No problem.", "Taken care of.",
]

THINK_STARTERS = [
    "The user wants to {action}. ",
    "I need to {action}. ",
    "The request is to {action}. ",
    "Let me {action}. ",
    "I should {action}. ",
]

SYSTEMS = [
    "You are a helpful assistant. Use the provided tools when appropriate.",
    "You are a smart home assistant. Help the user control their home and answer questions.",
    "You are an IoT monitoring assistant. Analyze data and take action when needed.",
    "You are a helpful AI assistant running on an edge device. Be concise and useful.",
    "You are an intelligent assistant. Think step by step, then respond or use tools as needed.",
    "You are a facility management assistant. Monitor conditions and control equipment.",
    "You are a personal assistant. Help with tasks, reminders, and device control.",
    "You are an assistant. Respond to commands, answer questions, and use tools when helpful.",
    "You are a building automation assistant. Control systems and respond to sensor data.",
    "You are a helpful AI. Use your tools to assist the user.",
]


# ── Tool pools ──────────────────────────────────────────────────────────────

SMART_HOME_TOOLS = [
    T("turn_on", "Turn on a device", {"device": {"type": "string", "description": "Device name"}}),
    T("turn_off", "Turn off a device", {"device": {"type": "string", "description": "Device name"}}),
    T("set_temperature", "Set thermostat temperature and mode", {
        "temperature": {"type": "number", "description": "Target temperature in celsius"},
        "mode": {"type": "string", "enum": ["heat", "cool", "auto"], "description": "HVAC mode"},
    }),
    T("set_brightness", "Set light brightness", {
        "device": {"type": "string", "description": "Light name"},
        "level": {"type": "integer", "description": "Brightness 0-100"},
    }),
    T("lock_door", "Lock a door", {"door": {"type": "string", "description": "Door name"}}),
    T("unlock_door", "Unlock a door", {"door": {"type": "string", "description": "Door name"}}),
    T("set_fan_speed", "Set fan speed", {
        "device": {"type": "string", "description": "Fan name"},
        "speed": {"type": "string", "enum": ["off", "low", "medium", "high"]},
    }),
    T("play_music", "Play music", {
        "speaker": {"type": "string"},
        "query": {"type": "string", "description": "Song, artist, or genre"},
    }),
    T("set_alarm", "Set an alarm", {"time": {"type": "string", "description": "Alarm time"}}),
    T("set_timer", "Set a countdown timer", {
        "duration": {"type": "string", "description": "Duration"},
        "label": {"type": "string", "description": "What the timer is for"},
    }),
    T("open_blinds", "Open window blinds/curtains", {"room": {"type": "string"}}),
    T("close_blinds", "Close window blinds/curtains", {"room": {"type": "string"}}),
    T("start_vacuum", "Start robot vacuum", {"room": {"type": "string", "description": "Room or 'all'"}}),
    T("send_notification", "Send a push notification", {
        "message": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
    }),
]

INDUSTRIAL_TOOLS = [
    T("send_alert", "Send monitoring alert", {
        "level": {"type": "string", "enum": ["info", "warning", "critical", "emergency"]},
        "message": {"type": "string"},
        "zone": {"type": "string"},
    }),
    T("start_pump", "Start a pump", {
        "pump_id": {"type": "string"},
        "flow_rate": {"type": "string", "enum": ["low", "medium", "high", "max"]},
    }),
    T("stop_pump", "Stop a pump", {"pump_id": {"type": "string"}}),
    T("open_valve", "Open a valve", {"valve_id": {"type": "string"}}),
    T("close_valve", "Close a valve", {"valve_id": {"type": "string"}}),
    T("enable_fan", "Enable ventilation fan", {
        "fan_id": {"type": "string"},
        "speed": {"type": "string", "enum": ["low", "medium", "high"]},
    }),
    T("request_maintenance", "Create maintenance work order", {
        "device_id": {"type": "string"},
        "issue": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
    }),
    T("evacuate_zone", "Issue evacuation alert", {
        "zone": {"type": "string"},
        "reason": {"type": "string"},
    }),
    T("log_event", "Record event to system log", {
        "event": {"type": "string"},
        "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
    }),
    T("calibrate_sensor", "Start sensor calibration", {
        "sensor_id": {"type": "string"},
        "reason": {"type": "string"},
    }),
]

GENERAL_TOOLS = [
    T("search", "Search for information", {"query": {"type": "string"}}),
    T("get_weather", "Get weather for a location", {"location": {"type": "string"}}),
    T("calculate", "Evaluate math expression", {"expression": {"type": "string"}}),
    T("send_message", "Send a message", {
        "to": {"type": "string"},
        "message": {"type": "string"},
    }),
    T("create_reminder", "Create a reminder", {
        "text": {"type": "string"},
        "when": {"type": "string", "description": "When to remind"},
    }),
    T("take_note", "Save a note", {
        "title": {"type": "string"},
        "content": {"type": "string"},
    }),
    T("convert_units", "Convert between units", {
        "value": {"type": "number"},
        "from_unit": {"type": "string"},
        "to_unit": {"type": "string"},
    }),
]

AGRICULTURE_TOOLS = [
    T("water_plants", "Water a garden zone", {
        "zone": {"type": "string"},
        "duration": {"type": "string"},
    }),
    T("adjust_greenhouse", "Adjust greenhouse settings", {
        "parameter": {"type": "string", "enum": ["temperature", "humidity", "ventilation", "shade"]},
        "value": {"type": "string"},
    }),
    T("feed_animals", "Dispense animal feed", {
        "feeder": {"type": "string"},
        "amount": {"type": "string", "enum": ["small", "normal", "large"]},
    }),
]

ALL_TOOL_POOLS = [SMART_HOME_TOOLS, INDUSTRIAL_TOOLS, GENERAL_TOOLS, AGRICULTURE_TOOLS]


def pick_tools(must_include_names=None, pool=None, n=None):
    if pool is None:
        pool = pick(ALL_TOOL_POOLS + [SMART_HOME_TOOLS + GENERAL_TOOLS, INDUSTRIAL_TOOLS + GENERAL_TOOLS])
    if n is None:
        n = random.randint(3, 7)
    pool = list(pool)
    selected = []
    if must_include_names:
        for name in must_include_names:
            for t in pool:
                if t["function"]["name"] == name:
                    selected.append(t)
                    pool.remove(t)
                    break
    remaining = max(0, n - len(selected))
    if remaining > 0 and pool:
        selected.extend(random.sample(pool, min(remaining, len(pool))))
    random.shuffle(selected)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT
# ══════════════════════════════════════════════════════════════════════════════

def format_sample(s):
    parts = []
    sys_content = s["system"]
    if s.get("tools"):
        sys_content += f"\n\n# Tools\n{json.dumps(s['tools'], separators=(',', ':'))}"
    parts.append(f"<|im_start|>system\n{sys_content}<|im_end|>")
    for m in s.get("history", []):
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append(f"<|im_start|>user\n{s['input']}<|im_end|>")
    a = ""
    if s.get("think"):
        a += f"<think>{s['think']}</think>\n"
    if s.get("tool_call"):
        tc = s["tool_call"]
        a += f"<tool_call>{json.dumps({'name': tc['name'], 'arguments': tc['arguments']}, separators=(',', ':'))}</tool_call>\n"
    a += s["response"]
    parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")
    return "\n".join(parts)


def to_openai(s):
    msgs = [{"role": "system", "content": s["system"]}]
    for m in s.get("history", []):
        msgs.append(m)
    msgs.append({"role": "user", "content": s["input"]})
    amsg = {"role": "assistant", "content": s["response"]}
    if s.get("tool_call"):
        amsg["tool_calls"] = [{"id": f"call_{random.randint(100000,999999)}", "type": "function", "function": s["tool_call"]}]
    msgs.append(amsg)
    row = {"messages": msgs}
    if s.get("tools"):
        row["tools"] = s["tools"]
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATORS — each call produces a UNIQUE sample
# ══════════════════════════════════════════════════════════════════════════════


def gen_turn_on():
    room = pick(ROOMS)
    dev_type = pick(list(DEVICES.keys()))
    device = pick(DEVICES[dev_type])
    phrase = pick(TURN_ON_PHRASES).format(device=device, room=room)
    full_name = f"{room} {device}"
    confirm = pick(CONFIRM_RESPONSES)

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["turn_on"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": pick(THINK_STARTERS).format(action=f"turn on the {device} in the {room}") + "I have the turn_on tool for this.",
        "response": f"{confirm} {full_name.capitalize()} is now on.",
        "tool_call": {"name": "turn_on", "arguments": {"device": full_name}},
        "category": "natural_command",
    }


def gen_turn_off():
    room = pick(ROOMS)
    dev_type = pick(list(DEVICES.keys()))
    device = pick(DEVICES[dev_type])
    phrase = pick(TURN_OFF_PHRASES).format(device=device, room=room)
    full_name = f"{room} {device}"
    confirm = pick(CONFIRM_RESPONSES)

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["turn_off"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": pick(THINK_STARTERS).format(action=f"turn off the {device} in the {room}") + "I'll use the turn_off tool.",
        "response": f"{confirm} {full_name.capitalize()} is off.",
        "tool_call": {"name": "turn_off", "arguments": {"device": full_name}},
        "category": "natural_command",
    }


def gen_set_brightness():
    room = pick(ROOMS)
    level = pick(BRIGHTNESS_LEVELS)
    phrase = pick(BRIGHTNESS_PHRASES).format(room=room, level=level)

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["set_brightness"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": f"The user wants the {room} lights at {level}%. I'll use set_brightness.",
        "response": f"{room.capitalize()} lights set to {level}%.",
        "tool_call": {"name": "set_brightness", "arguments": {"device": f"{room} lights", "level": level}},
        "category": "natural_command",
    }


def gen_set_temp():
    temp = pick(TEMPERATURES)
    mode = pick(["heat", "cool", "auto"])
    phrase = pick(TEMP_PHRASES).format(temp=temp)
    if random.random() < 0.3:
        mode_word = {"heat": "heating", "cool": "cooling", "auto": "auto"}[mode]
        phrase += f", {mode_word} mode"

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["set_temperature"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": f"Setting thermostat to {temp}°C, {mode} mode.",
        "response": f"Thermostat set to {temp}°C in {mode} mode.",
        "tool_call": {"name": "set_temperature", "arguments": {"temperature": temp, "mode": mode}},
        "category": "natural_command",
    }


def gen_implicit_hot():
    room = pick(ROOMS)
    phrase = pick(IMPLICIT_HOT_PHRASES).format(room=room)
    temp = pick([t for t in TEMPERATURES if t <= 22])

    responses = [
        f"I'll cool it down — setting the AC to {temp}°C.",
        f"Setting the thermostat to cool mode at {temp}°C. Should feel better soon.",
        f"Turning on cooling. Target: {temp}°C.",
        f"I've set cooling to {temp}°C. Give it a few minutes.",
    ]

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["set_temperature"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": f"The user is too hot. I should turn on cooling to a comfortable temperature like {temp}°C.",
        "response": pick(responses),
        "tool_call": {"name": "set_temperature", "arguments": {"temperature": temp, "mode": "cool"}},
        "category": "implicit_command",
    }


def gen_implicit_cold():
    room = pick(ROOMS)
    phrase = pick(IMPLICIT_COLD_PHRASES).format(room=room)
    temp = pick([t for t in TEMPERATURES if t >= 22])

    responses = [
        f"I'll warm it up — setting heating to {temp}°C.",
        f"Turning on the heater. Target: {temp}°C.",
        f"Setting heat mode to {temp}°C. Should warm up soon.",
        f"Heater set to {temp}°C.",
    ]

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["set_temperature"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": f"The user is cold. I'll set heating to {temp}°C.",
        "response": pick(responses),
        "tool_call": {"name": "set_temperature", "arguments": {"temperature": temp, "mode": "heat"}},
        "category": "implicit_command",
    }


def gen_implicit_dark():
    room = pick(ROOMS)
    phrase = pick(IMPLICIT_DARK_PHRASES).format(room=room)
    level = random.randint(60, 90)

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["set_brightness"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": f"The user says it's too dark in the {room}. I should increase the light level.",
        "response": f"I've turned up the {room} lights to {level}%.",
        "tool_call": {"name": "set_brightness", "arguments": {"device": f"{room} lights", "level": level}},
        "category": "implicit_command",
    }


def gen_implicit_bright():
    room = pick(ROOMS)
    phrase = pick(IMPLICIT_BRIGHT_PHRASES).format(room=room)
    level = random.randint(20, 45)

    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["set_brightness"], pool=SMART_HOME_TOOLS),
        "input": phrase,
        "think": f"Too bright in the {room}. I'll dim the lights.",
        "response": f"Dimmed the {room} lights to {level}%.",
        "tool_call": {"name": "set_brightness", "arguments": {"device": f"{room} lights", "level": level}},
        "category": "implicit_command",
    }


def gen_lock_door():
    door = pick(DEVICES["doors"])
    phrases = [f"Lock the {door}", f"Can you lock the {door}?", f"Please lock the {door}",
               f"Make sure the {door} is locked", f"Secure the {door}"]
    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["lock_door"], pool=SMART_HOME_TOOLS),
        "input": pick(phrases),
        "think": f"Lock the {door}.",
        "response": f"{door.capitalize()} is now locked.",
        "tool_call": {"name": "lock_door", "arguments": {"door": door}},
        "category": "natural_command",
    }


def gen_music():
    genre = pick(MUSIC_GENRES)
    room = pick(ROOMS)
    speaker = f"{room} speaker"
    phrases = [f"Play some {genre}", f"Put on {genre} music", f"Play {genre} in the {room}",
               f"I want to hear {genre}", f"Can you play {genre} music?"]
    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["play_music"], pool=SMART_HOME_TOOLS),
        "input": pick(phrases),
        "think": f"The user wants {genre} music. Playing on {speaker}.",
        "response": f"Playing {genre} on the {speaker}. Enjoy!",
        "tool_call": {"name": "play_music", "arguments": {"speaker": speaker, "query": genre}},
        "category": "natural_command",
    }


def gen_timer_alarm():
    if random.random() < 0.5:
        # Timer
        dur = pick(DURATIONS)
        label = pick(["pasta", "laundry", "meeting", "break", "oven", "meds", "workout", "tea", "rice", "nap"])
        phrases = [f"Set a timer for {dur} for the {label}", f"{dur} timer for {label}",
                   f"Timer {dur}, {label}", f"Start a {dur} countdown for {label}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["set_timer"], pool=SMART_HOME_TOOLS),
            "input": pick(phrases),
            "think": f"Setting a {dur} timer for {label}.",
            "response": f"{dur.capitalize()} timer set for {label}.",
            "tool_call": {"name": "set_timer", "arguments": {"duration": dur, "label": label}},
            "category": "natural_command",
        }
    else:
        # Alarm
        time = pick(TIMES)
        phrases = [f"Set an alarm for {time}", f"Wake me up at {time}",
                   f"Alarm at {time} please", f"I need an alarm for {time}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["set_alarm"], pool=SMART_HOME_TOOLS),
            "input": pick(phrases),
            "think": f"Setting alarm for {time}.",
            "response": f"Alarm set for {time}.",
            "tool_call": {"name": "set_alarm", "arguments": {"time": time}},
            "category": "natural_command",
        }


def gen_blinds():
    room = pick(ROOMS)
    action = pick(["open", "close"])
    cover = pick(DEVICES["blinds"])
    phrases = [f"{action.capitalize()} the {cover} in the {room}", f"Can you {action} the {room} {cover}?",
               f"Please {action} the {cover} in the {room}", f"I want the {room} {cover} {action}ed"]
    tool_name = f"{action}_blinds"
    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=[tool_name], pool=SMART_HOME_TOOLS),
        "input": pick(phrases),
        "think": f"{action.capitalize()} the {cover} in the {room}.",
        "response": f"{room.capitalize()} {cover} are now {action + ('d' if action == 'close' else '')}.",
        "tool_call": {"name": tool_name, "arguments": {"room": room}},
        "category": "natural_command",
    }


def gen_vacuum():
    room = pick(ROOMS + ["all rooms", "the whole house", "everywhere"])
    target = "all" if room in ["all rooms", "the whole house", "everywhere"] else room
    phrases = [f"Vacuum the {room}", f"Start the vacuum in the {room}", f"Clean the {room}",
               f"Can you vacuum {room}?", f"Run the robot vacuum in the {room}"]
    return {
        "system": pick(SYSTEMS),
        "tools": pick_tools(must_include_names=["start_vacuum"], pool=SMART_HOME_TOOLS),
        "input": pick(phrases),
        "think": f"Starting vacuum in {target}.",
        "response": f"Robot vacuum is cleaning {target if target != 'all' else 'the whole house'} now.",
        "tool_call": {"name": "start_vacuum", "arguments": {"room": target}},
        "category": "natural_command",
    }


# ── Industrial commands ────────────────────────────────────────────────────

def gen_industrial_command():
    cmd_type = pick(["pump_start", "pump_stop", "valve", "fan", "maintenance", "alert", "calibrate"])
    loc = pick(INDUSTRIAL_LOCATIONS + ZONES)

    if cmd_type == "pump_start":
        pid = pick(PUMP_IDS)
        rate = pick(["low", "medium", "high", "max"])
        phrases = [f"Start {pid} at {rate} flow", f"Turn on {pid}, {rate} rate",
                   f"Fire up {pid} at {rate}", f"Enable {pid} at {rate} flow rate"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["start_pump"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"Starting {pid} at {rate} flow rate.",
            "response": f"{pid.capitalize().replace('_', ' ')} running at {rate} flow.",
            "tool_call": {"name": "start_pump", "arguments": {"pump_id": pid, "flow_rate": rate}},
            "category": "natural_command",
        }

    elif cmd_type == "pump_stop":
        pid = pick(PUMP_IDS)
        phrases = [f"Stop {pid}", f"Shut down {pid}", f"Turn off {pid}", f"Kill {pid}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["stop_pump"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"Stopping {pid}.",
            "response": f"{pid.capitalize().replace('_', ' ')} stopped.",
            "tool_call": {"name": "stop_pump", "arguments": {"pump_id": pid}},
            "category": "natural_command",
        }

    elif cmd_type == "valve":
        vid = pick(VALVE_IDS)
        action = pick(["open", "close"])
        phrases = [f"{action.capitalize()} {vid}", f"Can you {action} {vid}?",
                   f"I need {vid} {action}ed", f"Please {action} {vid}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=[f"{action}_valve"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"{action.capitalize()} {vid}.",
            "response": f"{vid.capitalize().replace('_', ' ')} is now {action + ('d' if action == 'close' else '')}.",
            "tool_call": {"name": f"{action}_valve", "arguments": {"valve_id": vid}},
            "category": "natural_command",
        }

    elif cmd_type == "fan":
        fid = pick(FAN_IDS)
        speed = pick(["low", "medium", "high"])
        phrases = [f"Turn on {fid} at {speed}", f"Enable {fid}, {speed} speed",
                   f"Start {fid} on {speed}", f"Set {fid} to {speed}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["enable_fan"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"Enabling {fid} at {speed} speed.",
            "response": f"{fid.capitalize().replace('_', ' ')} running at {speed}.",
            "tool_call": {"name": "enable_fan", "arguments": {"fan_id": fid, "speed": speed}},
            "category": "natural_command",
        }

    elif cmd_type == "maintenance":
        sid = pick(SENSOR_IDS)
        issues = ["making unusual noise", "readings seem off", "intermittent connection",
                  "response time is slow", "physical damage visible", "needs recalibration",
                  "error light is on", "display is blank", "overheating"]
        issue = pick(issues)
        priority = pick(["low", "medium", "high"])
        phrases = [f"The {sid} is {issue}", f"We need maintenance on {sid} — {issue}",
                   f"Something's wrong with {sid}, it's {issue}", f"Can you log a maintenance request for {sid}? It's {issue}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["request_maintenance"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"{sid} has an issue: {issue}. Submitting a {priority}-priority maintenance request.",
            "response": f"Maintenance request submitted for {sid}: {issue}. Priority: {priority}.",
            "tool_call": {"name": "request_maintenance", "arguments": {"device_id": sid, "issue": issue, "priority": priority}},
            "category": "natural_command",
        }

    elif cmd_type == "alert":
        level = pick(["info", "warning", "critical"])
        situations = [
            f"unusual activity in {loc}", f"equipment malfunction in {loc}",
            f"environmental reading out of range in {loc}", f"scheduled inspection overdue for {loc}",
            f"visitor reported issue in {loc}", f"power fluctuation in {loc}",
        ]
        situation = pick(situations)
        phrases = [f"Send a {level} alert about {situation}", f"Alert: {situation}",
                   f"Flag {situation} as {level}", f"Log a {level} alert for {situation}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["send_alert"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"Sending {level} alert: {situation}.",
            "response": f"{level.capitalize()} alert sent: {situation}.",
            "tool_call": {"name": "send_alert", "arguments": {"level": level, "message": situation, "zone": loc}},
            "category": "natural_command",
        }

    else:  # calibrate
        sid = pick(SENSOR_IDS)
        reasons = ["routine calibration", "readings seem off", "after maintenance", "quarterly schedule", "drift detected"]
        reason = pick(reasons)
        phrases = [f"Calibrate {sid}", f"Start calibration on {sid} — {reason}",
                   f"I need {sid} recalibrated", f"Run calibration for {sid}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["calibrate_sensor"], pool=INDUSTRIAL_TOOLS),
            "input": pick(phrases),
            "think": f"Starting calibration for {sid}. Reason: {reason}.",
            "response": f"Calibration started for {sid} ({reason}).",
            "tool_call": {"name": "calibrate_sensor", "arguments": {"sensor_id": sid, "reason": reason}},
            "category": "natural_command",
        }


# ── General commands ───────────────────────────────────────────────────────

def gen_general_command():
    cmd_type = pick(["reminder", "message", "weather", "calculate", "convert", "note"])

    if cmd_type == "reminder":
        task = pick(TASKS_TO_REMEMBER)
        when = pick(["in 1 hour", "in 2 hours", "in 30 minutes", "tomorrow morning",
                     "tomorrow at 9am", "tonight", "in 3 hours", "at 5pm"])
        phrases = [f"Remind me to {task} {when}", f"Set a reminder: {task} {when}",
                   f"Don't let me forget to {task} — remind me {when}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["create_reminder"], pool=GENERAL_TOOLS),
            "input": pick(phrases),
            "think": f"Setting reminder: {task}, {when}.",
            "response": f"Reminder set: {task} ({when}).",
            "tool_call": {"name": "create_reminder", "arguments": {"text": task, "when": when}},
            "category": "natural_command",
        }

    elif cmd_type == "message":
        person = pick(PEOPLE)
        messages = [f"I'll be {random.randint(5,45)} minutes late",
                    "On my way", "Can you call me?", "Meeting is postponed",
                    "I'll handle it", "Running behind, start without me",
                    "Pick up some milk please", "Everything's fine here"]
        msg = pick(messages)
        phrases = [f"Send a message to {person}: {msg}", f"Text {person} saying {msg}",
                   f"Tell {person} that {msg}", f"Message {person}: {msg}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["send_message"], pool=GENERAL_TOOLS),
            "input": pick(phrases),
            "think": f"Sending message to {person}: \"{msg}\".",
            "response": f"Message sent to {person}.",
            "tool_call": {"name": "send_message", "arguments": {"to": person, "message": msg}},
            "category": "natural_command",
        }

    elif cmd_type == "weather":
        cities = ["Tokyo", "London", "New York", "Paris", "Sydney", "Berlin",
                  "Toronto", "Mumbai", "Seoul", "Portland", "Seattle", "Austin",
                  "Denver", "Chicago", "Boston", "Miami", "San Francisco"]
        city = pick(cities)
        phrases = [f"What's the weather in {city}?", f"How's the weather in {city}?",
                   f"Weather for {city}", f"Is it raining in {city}?",
                   f"What's it like outside in {city}?"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["get_weather"], pool=GENERAL_TOOLS),
            "input": pick(phrases),
            "think": f"The user wants weather for {city}. Using get_weather tool.",
            "response": f"Let me check the weather in {city} for you.",
            "tool_call": {"name": "get_weather", "arguments": {"location": city}},
            "category": "natural_command",
        }

    elif cmd_type == "calculate":
        ops = [
            (f"{random.randint(1,99)} * {random.randint(1,99)}", None),
            (f"{random.randint(100,999)} + {random.randint(100,999)}", None),
            (f"{random.randint(1,50)}% of {random.randint(50,500)}", None),
            (f"{random.randint(10,200)} / {random.randint(2,10)}", None),
            (f"square root of {random.choice([4,9,16,25,36,49,64,81,100,144])}", None),
        ]
        expr_natural, _ = pick(ops)
        expr_math = expr_natural.replace("% of", "/100 *").replace("square root of", "sqrt(")
        if "sqrt(" in expr_math:
            expr_math += ")"
        phrases = [f"What's {expr_natural}?", f"Calculate {expr_natural}",
                   f"How much is {expr_natural}?", f"{expr_natural}?"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["calculate"], pool=GENERAL_TOOLS),
            "input": pick(phrases),
            "think": f"Math question: {expr_natural}. Using calculate tool.",
            "response": f"Let me calculate that for you.",
            "tool_call": {"name": "calculate", "arguments": {"expression": expr_math}},
            "category": "natural_command",
        }

    elif cmd_type == "convert":
        conversions = [
            (random.randint(50, 100), "fahrenheit", "celsius"),
            (random.randint(1, 50), "miles", "kilometers"),
            (random.randint(1, 200), "pounds", "kilograms"),
            (random.randint(1, 100), "gallons", "liters"),
            (random.randint(100, 1000), "feet", "meters"),
            (round(random.uniform(1, 50), 1), "inches", "centimeters"),
        ]
        val, from_u, to_u = pick(conversions)
        phrases = [f"Convert {val} {from_u} to {to_u}", f"How many {to_u} is {val} {from_u}?",
                   f"What's {val} {from_u} in {to_u}?", f"{val} {from_u} to {to_u}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["convert_units"], pool=GENERAL_TOOLS),
            "input": pick(phrases),
            "think": f"Converting {val} {from_u} to {to_u}.",
            "response": f"Let me convert that for you.",
            "tool_call": {"name": "convert_units", "arguments": {"value": val, "from_unit": from_u, "to_unit": to_u}},
            "category": "natural_command",
        }

    else:  # note
        titles = ["Shopping list", "Meeting notes", "Ideas", "To-do", "Bug report",
                  "Observation", "Recipe", "Maintenance log", "Reading list"]
        contents = [f"Need to buy {', '.join(pick_n(['milk','eggs','bread','butter','cheese','apples','rice','pasta','chicken','soap'], random.randint(2,4)))}",
                    f"Discussed {pick(['budget', 'timeline', 'staffing', 'roadmap'])} with team",
                    f"Try {pick(['new sensor layout', 'different filter settings', 'backup power config', 'weekly calibration'])}",
                    f"{pick(['Replace', 'Check', 'Update', 'Review'])} {pick(['filters', 'firmware', 'batteries', 'certificates', 'backup logs'])}"]
        title = pick(titles)
        content = pick(contents)
        phrases = [f"Save a note: {content}", f"Take a note — {title}: {content}",
                   f"Note this down: {content}", f"Jot down: {content}"]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["take_note"], pool=GENERAL_TOOLS),
            "input": pick(phrases),
            "think": f"Saving note: {title}.",
            "response": f"Note saved: \"{title}\".",
            "tool_call": {"name": "take_note", "arguments": {"title": title, "content": content}},
            "category": "natural_command",
        }


# ── Sensor reading + action ────────────────────────────────────────────────

def gen_sensor():
    sensor_type = pick(list(SENSOR_TYPES.keys()))
    info = SENSOR_TYPES[sensor_type]
    loc = pick(INDUSTRIAL_LOCATIONS + ZONES + ROOMS)
    sid = pick(SENSOR_IDS)
    unit_display = pick(info["units_alt"])

    # Pick severity
    severity = random.choices(["normal", "warning", "critical"], weights=[0.3, 0.4, 0.3])[0]
    nlo, nhi = info["normal"]
    wlo, whi = info["warn"]
    clo, chi = info["crit"]

    if severity == "normal":
        val = round(random.uniform(nlo, nhi), 1)
    elif severity == "warning":
        if random.random() < 0.5:
            val = round(random.uniform(wlo, nlo), 1)
        else:
            val = round(random.uniform(nhi, whi), 1)
    else:
        if random.random() < 0.5:
            val = round(random.uniform(clo, wlo), 1)
        else:
            val = round(random.uniform(whi, chi), 1)

    if sensor_type == "motion":
        val = round(val)

    # Input phrasing
    input_phrases = [
        f"{sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}",
        f"{sensor_type.replace('_', ' ')} reading: {val} {unit_display} at {loc}",
        f"Sensor {sid} at {loc}: {sensor_type.replace('_', ' ')} is {val} {unit_display}",
        f"Getting {val} {unit_display} for {sensor_type.replace('_', ' ')} in {loc}",
        f"The {sensor_type.replace('_', ' ')} in {loc} is showing {val} {unit_display}",
    ]

    direction = "high" if val > nhi else "low" if val < nlo else None
    risk = SENSOR_RISKS.get(sensor_type, {"high": "unknown risk", "low": "unknown risk"})

    if severity == "normal":
        think = f"The {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}. Normal range is {nlo}-{nhi}. This is fine, no action needed."
        responses = [
            f"All good — {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}, within normal range ({nlo}-{nhi}).",
            f"{sensor_type.replace('_', ' ').capitalize()} at {loc} reads {val} {unit_display}. That's normal. No action needed.",
            f"Reading is {val} {unit_display} at {loc}. Within normal range.",
        ]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(pool=pick(ALL_TOOL_POOLS), n=random.randint(3, 6)),
            "input": pick(input_phrases),
            "think": think,
            "response": pick(responses),
            "tool_call": None,
            "category": "sensor_normal",
        }

    elif severity == "warning":
        risk_text = risk.get(direction, "potential issues")
        think = f"The {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}. Normal is {nlo}-{nhi}, so this is {direction}. Risk: {risk_text}. I should flag this."
        responses = [
            f"Warning: {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display} — that's {direction} (normal: {nlo}-{nhi}). Risk: {risk_text}. I've sent a warning alert.",
            f"The {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}, outside the normal range. This could lead to {risk_text}. Alert sent.",
            f"Heads up — {sensor_type.replace('_', ' ')} at {loc} is {direction} at {val} {unit_display}. Normal is {nlo}-{nhi}. Flagging this.",
        ]
        tc = {"name": "send_alert", "arguments": {"level": "warning", "message": f"{sensor_type} at {loc} is {val} {unit_display} ({direction})", "zone": loc}}
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["send_alert"], pool=INDUSTRIAL_TOOLS),
            "input": pick(input_phrases),
            "think": think,
            "response": pick(responses),
            "tool_call": tc,
            "category": "sensor_warning",
        }

    else:  # critical
        risk_text = risk.get(direction, "serious danger")
        if sensor_type == "gas" and direction == "high":
            think = f"CRITICAL: gas at {loc} is {val} {unit_display}. This is dangerous — {risk_text}. Evacuation needed immediately."
            resp = f"DANGER: Gas at {loc} is {val} {unit_display} — critically high. Evacuating the zone. Leave the area immediately."
            tc = {"name": "evacuate_zone", "arguments": {"zone": loc, "reason": f"Dangerous gas level: {val} ppm"}}
            must = ["evacuate_zone"]
        elif sensor_type == "water_level" and direction == "high":
            think = f"CRITICAL: water level at {loc} is {val} {unit_display}. Flooding imminent. Starting emergency pump."
            resp = f"Critical: Water level at {loc} is {val} {unit_display} — overflow risk. Emergency pump activated."
            tc = {"name": "start_pump", "arguments": {"pump_id": f"pump_{loc.replace(' ', '_')}", "flow_rate": "max"}}
            must = ["start_pump"]
        else:
            think = f"CRITICAL: {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}. Way outside safe range ({nlo}-{nhi}). Risk: {risk_text}. Immediate alert needed."
            resp = f"CRITICAL: {sensor_type.replace('_', ' ')} at {loc} is {val} {unit_display}. This is dangerous — {risk_text}. I've sent an emergency alert. Immediate attention needed."
            tc = {"name": "send_alert", "arguments": {"level": "critical", "message": f"CRITICAL: {sensor_type} at {loc} is {val} {unit_display}", "zone": loc}}
            must = ["send_alert"]

        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=must, pool=INDUSTRIAL_TOOLS),
            "input": pick(input_phrases),
            "think": think,
            "response": resp,
            "tool_call": tc,
            "category": "sensor_critical",
        }


# ── Questions without tool calls ──────────────────────────────────────────

def gen_question():
    q_type = pick(["sensor_knowledge", "general_knowledge", "greeting", "meta", "advice"])

    if q_type == "sensor_knowledge":
        sensor_type = pick(list(SENSOR_TYPES.keys()))
        info = SENSOR_TYPES[sensor_type]
        nlo, nhi = info["normal"]
        risk = SENSOR_RISKS.get(sensor_type, {"high": "various issues", "low": "various issues"})

        sub = pick(["range", "high_meaning", "low_meaning", "broken", "how_often"])
        if sub == "range":
            questions = [f"What's the normal range for {sensor_type.replace('_', ' ')}?",
                        f"What should {sensor_type.replace('_', ' ')} readings be?"]
            resp = f"Normal {sensor_type.replace('_', ' ')} range is {nlo}-{nhi} {info['unit']}. Outside that, you should investigate."
        elif sub == "high_meaning":
            questions = [f"What happens if {sensor_type.replace('_', ' ')} is too high?",
                        f"Why is high {sensor_type.replace('_', ' ')} bad?"]
            resp = f"High {sensor_type.replace('_', ' ')} can cause: {risk['high']}. The higher it goes, the more urgent the response needs to be."
        elif sub == "low_meaning":
            questions = [f"What if {sensor_type.replace('_', ' ')} drops too low?",
                        f"Is low {sensor_type.replace('_', ' ')} a problem?"]
            resp = f"Low {sensor_type.replace('_', ' ')} can cause: {risk['low']}."
        elif sub == "broken":
            symptoms = pick(["stuck at the same value", "jumping between zero and normal readings",
                           "drifting steadily in one direction", "showing impossible values"])
            questions = [f"My {sensor_type.replace('_', ' ')} sensor might be broken. It's {symptoms}.",
                        f"Is the {sensor_type.replace('_', ' ')} sensor faulty? It's {symptoms}"]
            resp = f"That does sound like a sensor fault. \"{symptoms.capitalize()}\" typically means a hardware issue. I'd recommend recalibrating it first, and replacing it if the problem persists."
        else:
            questions = [f"How often should I calibrate {sensor_type.replace('_', ' ')} sensors?"]
            intervals = {"gas": "every 3-6 months (safety-critical)", "temperature": "every 6-12 months",
                        "humidity": "every 6 months", "pressure": "annually", "co2": "every 6 months"}
            interval = intervals.get(sensor_type, "every 6-12 months")
            resp = f"For {sensor_type.replace('_', ' ')} sensors, calibrate {interval}. Always follow the manufacturer's specs."

        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(n=random.randint(2, 5)),
            "input": pick(questions),
            "think": f"The user is asking about {sensor_type.replace('_', ' ')}. I should give them practical, accurate info.",
            "response": resp,
            "tool_call": None,
            "category": "question",
        }

    elif q_type == "greeting":
        greetings_in = ["Hello!", "Hi", "Hey there", "Good morning", "Good evening",
                       "Hey", "Howdy", "What's up?", "Yo"]
        greetings_out = [
            "Hi! How can I help you?",
            "Hello! What can I do for you?",
            "Hey! Ready to help. What do you need?",
            "Hi there! I can control devices, check sensors, set reminders, and more. What's up?",
            "Good to see you! What can I help with today?",
        ]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(n=random.randint(2, 5)),
            "input": pick(greetings_in),
            "think": "The user is greeting me. I'll respond friendly and offer help.",
            "response": pick(greetings_out),
            "tool_call": None,
            "category": "question",
        }

    elif q_type == "meta":
        meta_qs = [
            ("Who are you?", "I'm a small AI assistant designed to run on edge devices. I can understand commands, control devices, analyze sensor data, and answer questions — all locally without needing cloud access."),
            ("What can you do?", "I can control smart home devices, monitor sensors, set reminders and timers, send messages, answer questions, and use whatever tools are configured. Just ask!"),
            ("How do you work?", "You send me messages, I think through what you need, and I either respond directly or use one of my configured tools to take action. I always explain my reasoning."),
            ("Are you connected to the internet?", "I run locally on your edge device. I can only access external services through the tools that have been configured for me."),
            ("Can I trust your sensor readings?", "I report what the sensors tell me. If a reading seems off, I'll flag it. But sensors can fail — if something doesn't match what you see in person, the sensor might need calibration or replacement."),
        ]
        q, a = pick(meta_qs)
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(n=random.randint(2, 5)),
            "input": q,
            "think": "The user wants to know about me. I'll give a helpful, concise answer.",
            "response": a,
            "tool_call": None,
            "category": "question",
        }

    elif q_type == "advice":
        advice_qs = [
            ("What temperature should I keep the server room at?",
             "Server rooms should be 18-27°C, ideally 20-22°C. Below 18°C wastes energy, above 27°C risks equipment overheating. Keep humidity at 40-60% too."),
            ("How can I save energy?",
             "Biggest wins: reduce HVAC when spaces are unoccupied, use timers/schedules, close blinds during peak sun, and make sure equipment turns off when not needed. I can automate most of this."),
            ("What's a good humidity level?",
             "30-60% is the sweet spot. Below 30% causes dry skin and static. Above 60% promotes mold. For server rooms, aim for 40-55%."),
            ("Should I worry about CO2 levels?",
             "If you're in an enclosed space with people, yes. Below 800 ppm is fine. 800-1200 ppm means you need better ventilation. Above 1500 ppm causes drowsiness and headaches."),
            ("How do I know if my HVAC is working properly?",
             "Check: is the space reaching the target temperature within a reasonable time? Is it cycling on/off too frequently (short-cycling)? Are energy bills unusually high? Any unusual noises? If any of these are off, it may need maintenance."),
        ]
        q, a = pick(advice_qs)
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(n=random.randint(2, 5)),
            "input": q,
            "think": "Practical question. I'll give concise, actionable advice.",
            "response": a,
            "tool_call": None,
            "category": "question",
        }

    else:  # general_knowledge
        gk_qs = [
            ("Thanks!", "You're welcome! Let me know if you need anything else."),
            ("Never mind", "No problem. I'm here when you need me."),
            ("That worked, thanks", "Glad it worked! Let me know if there's anything else."),
            ("Can you explain that in simpler terms?", "Of course — which part would you like me to clarify? I'll break it down."),
            ("I don't understand", "No worries — what specifically is confusing? I'll try to explain it differently."),
        ]
        q, a = pick(gk_qs)
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(n=random.randint(2, 5)),
            "input": q,
            "think": "Conversational response needed.",
            "response": a,
            "tool_call": None,
            "category": "question",
        }


# ── Novel / unfamiliar tools ──────────────────────────────────────────────

def gen_novel_tool():
    """Procedurally generate novel tool schemas the model hasn't seen."""
    novel_domains = [
        {
            "tool": T("order_food", "Order food for delivery", {
                "item": {"type": "string", "description": "Food item to order"},
                "quantity": {"type": "integer"},
                "address": {"type": "string"},
            }),
            "inputs": lambda: (
                f"Order {random.randint(1,4)} {pick(['pizza', 'burger', 'salad', 'sandwich', 'sushi', 'tacos'])}s to {random.randint(1,200)} {pick(['Oak', 'Elm', 'Main', 'Park', 'Lake', 'River'])} Street",
                None  # We'll parse
            ),
        },
        {
            "tool": T("book_room", "Book a meeting room", {
                "room": {"type": "string"},
                "time": {"type": "string"},
                "duration": {"type": "string"},
            }),
        },
        {
            "tool": T("control_robot", "Send command to a robot", {
                "robot_id": {"type": "string"},
                "action": {"type": "string", "enum": ["move_forward", "move_back", "turn_left", "turn_right", "stop", "grab", "release"]},
                "parameter": {"type": "string", "description": "Action parameter (distance, angle, etc.)"},
            }),
        },
        {
            "tool": T("deploy_service", "Deploy a software service", {
                "service": {"type": "string"},
                "version": {"type": "string"},
                "environment": {"type": "string", "enum": ["staging", "production"]},
            }),
        },
        {
            "tool": T("restock_item", "Restock an inventory item", {
                "item": {"type": "string"},
                "quantity": {"type": "integer"},
                "warehouse": {"type": "string"},
            }),
        },
        {
            "tool": T("schedule_meeting", "Schedule a meeting with attendees", {
                "title": {"type": "string"},
                "attendees": {"type": "string", "description": "Comma-separated names"},
                "time": {"type": "string"},
            }),
        },
        {
            "tool": T("adjust_irrigation", "Control irrigation system", {
                "zone": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "flow_rate": {"type": "string", "enum": ["low", "medium", "high"]},
            }),
        },
    ]

    domain = pick(novel_domains)
    tool = domain["tool"]
    fname = tool["function"]["name"]
    params = tool["function"]["parameters"]["properties"]
    param_names = list(params.keys())

    # Generate a plausible request and matching arguments
    if fname == "order_food":
        item = pick(["pizza", "burger", "salad", "sandwich", "sushi roll"])
        qty = random.randint(1, 5)
        addr = f"{random.randint(1,500)} {pick(['Oak', 'Elm', 'Main', 'Park', 'Maple'])} Street"
        inp = f"Order {qty} {item}{'s' if qty > 1 else ''} to {addr}"
        args = {"item": item, "quantity": qty, "address": addr}
    elif fname == "book_room":
        room = pick(["conference room A", "meeting room 3", "board room", "huddle space", "training room"])
        time = pick(TIMES)
        dur = pick(["30 minutes", "1 hour", "2 hours"])
        inp = f"Book {room} at {time} for {dur}"
        args = {"room": room, "time": time, "duration": dur}
    elif fname == "control_robot":
        rid = f"robot_{random.randint(1,5)}"
        action = pick(["move_forward", "turn_left", "turn_right", "stop", "grab"])
        param = f"{random.randint(1,10)} meters" if "move" in action else f"{random.randint(30,180)} degrees" if "turn" in action else ""
        inp = f"{'Move' if 'move' in action else action.replace('_', ' ').capitalize()} {rid}" + (f" {param}" if param else "")
        args = {"robot_id": rid, "action": action, "parameter": param or "none"}
    elif fname == "deploy_service":
        svc = pick(["auth-service", "api-gateway", "user-service", "payment-service", "notification-service"])
        ver = f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}"
        env = pick(["staging", "production"])
        inp = f"Deploy {svc} {ver} to {env}"
        args = {"service": svc, "version": ver, "environment": env}
    elif fname == "restock_item":
        item = pick(["sensor batteries", "replacement filters", "cleaning supplies", "ethernet cables", "thermal paste", "zip ties"])
        qty = random.randint(10, 200)
        wh = pick(["main warehouse", "warehouse B", "storage room 3"])
        inp = f"Restock {qty} {item} at {wh}"
        args = {"item": item, "quantity": qty, "warehouse": wh}
    elif fname == "schedule_meeting":
        attendees = ", ".join(pick_n(PEOPLE, random.randint(2, 4)))
        time = pick(TIMES)
        titles = ["Sprint planning", "Design review", "Standup", "Retrospective", "Budget review", "1:1", "All hands"]
        title = pick(titles)
        inp = f"Schedule {title} with {attendees} at {time}"
        args = {"title": title, "attendees": attendees, "time": time}
    else:  # irrigation
        zone = pick(["zone A", "zone B", "tomatoes", "front lawn", "herb garden", "orchard"])
        dur = random.randint(5, 30)
        rate = pick(["low", "medium", "high"])
        inp = f"Water {zone} for {dur} minutes at {rate} flow"
        args = {"zone": zone, "duration_minutes": dur, "flow_rate": rate}

    # Add some distractor tools
    distractors = pick_n(pick(ALL_TOOL_POOLS), random.randint(1, 3))
    all_tools = [tool] + distractors
    random.shuffle(all_tools)

    return {
        "system": pick(SYSTEMS),
        "tools": all_tools,
        "input": inp,
        "think": f"The user wants to {fname.replace('_', ' ')}. I have a {fname} tool. Let me fill in the parameters from their request.",
        "response": f"{pick(CONFIRM_RESPONSES)} {fname.replace('_', ' ').capitalize()} done.",
        "tool_call": {"name": fname, "arguments": args},
        "category": "novel_tool",
    }


# ── Refusal / can't do ─────────────────────────────────────────────────────

def gen_refusal():
    room = pick(ROOMS)
    loc = pick(INDUSTRIAL_LOCATIONS)
    person = pick(PEOPLE)

    scenarios = [
        # Tool mismatch
        {"input": f"Turn on the AC in the {room}",
         "tools": GENERAL_TOOLS,  # No device control tools
         "think": f"The user wants AC control but I only have general tools (search, weather, etc). No device control tools available.",
         "response": "I don't have any device control tools configured right now. I can't control the AC with my current setup. Please add smart home tools to my configuration."},
        # Impossible request
        {"input": "What will the stock market do tomorrow?",
         "tools": pick(ALL_TOOL_POOLS),
         "think": "The user is asking for stock predictions. I can't predict the future and I don't have financial data access.",
         "response": "I can't predict stock markets. I'm designed for device control and monitoring. You'd need a financial service for that."},
        # Unethical
        {"input": "Help me hack into the security cameras",
         "tools": pick(ALL_TOOL_POOLS),
         "think": "The user is asking for unauthorized access. I should refuse clearly.",
         "response": "I can't help with unauthorized access to security systems. If you need legitimate access, contact your system administrator."},
        # Out of scope
        {"input": f"Write me a 500 word essay about climate change",
         "tools": pick(ALL_TOOL_POOLS),
         "think": "The user wants a long essay. I'm a small model optimized for concise responses, device control, and sensor monitoring. I can give brief info but not write essays.",
         "response": "I'm designed for device control and monitoring, not long-form writing. I can answer brief questions about environmental topics, but for essays you'd want a larger language model."},
        # No relevant tool
        {"input": f"Call {person}",
         "tools": [T("send_message", "Send text message", {"to": {"type": "string"}, "message": {"type": "string"}}),
                   T("take_note", "Save a note", {"title": {"type": "string"}, "content": {"type": "string"}})],
         "think": f"The user wants to call {person}. I have send_message and take_note, but no calling tool. I can offer to text instead.",
         "response": f"I can't make phone calls — I don't have a calling tool. I can send {person} a text message instead if you'd like."},
        # Safety refusal
        {"input": "Override all the safety alarms",
         "tools": INDUSTRIAL_TOOLS,
         "think": "The user wants to override safety alarms. This is dangerous and I should refuse. Safety systems exist to protect people.",
         "response": "I won't override safety alarms — they exist to protect people and equipment. If you're getting false alarms, let's troubleshoot the cause instead."},
    ]

    s = pick(scenarios)
    tools = pick_tools(pool=s["tools"], n=random.randint(2, 5)) if isinstance(s["tools"], list) and len(s["tools"]) > 0 and "function" in s["tools"][0] else pick_tools(pool=s["tools"], n=random.randint(2, 5))

    return {
        "system": pick(SYSTEMS),
        "tools": s.get("tools_override", tools),
        "input": s["input"],
        "think": s["think"],
        "response": s["response"],
        "tool_call": None,
        "category": "refusal",
    }


# ── Multi-turn conversations ──────────────────────────────────────────────

def gen_conversation():
    conv_type = pick(["followup_action", "clarification", "correction", "continuation"])

    if conv_type == "followup_action":
        room = pick(ROOMS)
        device = pick(DEVICES["lights"])
        level = random.randint(20, 80)
        history = [
            {"role": "user", "content": f"Turn on the {room} {device}"},
            {"role": "assistant", "content": f"Done, {room} {device} is on."},
        ]
        followups = [
            (f"Actually dim {'them' if 'light' in device else 'it'} to {level}%",
             f"Dimming to {level}%.",
             {"name": "set_brightness", "arguments": {"device": f"{room} {device}", "level": level}},
             "set_brightness"),
            (f"Now turn on the {pick(DEVICES['fans'])} too",
             None, None, None),  # Will build below
            (f"And lock the {pick(DEVICES['doors'])}",
             None, None, None),
        ]
        fu_input, fu_resp, fu_tc, fu_tool = pick(followups)
        if fu_tc is None:
            if "fan" in fu_input:
                fan = fu_input.split("turn on the ")[1] if "turn on the " in fu_input else "fan"
                fu_tc = {"name": "turn_on", "arguments": {"device": fan}}
                fu_tool = "turn_on"
                fu_resp = f"{fan.capitalize()} is now on."
            else:
                door = fu_input.split("lock the ")[1] if "lock the " in fu_input else "front door"
                fu_tc = {"name": "lock_door", "arguments": {"door": door}}
                fu_tool = "lock_door"
                fu_resp = f"{door.capitalize()} locked."

        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=[fu_tool], pool=SMART_HOME_TOOLS),
            "history": history,
            "input": fu_input,
            "think": f"Following up on the previous action. The user wants an additional change.",
            "response": fu_resp,
            "tool_call": fu_tc,
            "category": "conversation",
        }

    elif conv_type == "clarification":
        sensor_type = pick(list(SENSOR_TYPES.keys()))
        loc = pick(INDUSTRIAL_LOCATIONS)
        history = [
            {"role": "user", "content": f"Something seems off with the {sensor_type.replace('_', ' ')}"},
            {"role": "assistant", "content": f"I'd need more details. Which location and what's the current reading?"},
        ]
        val = round(random.uniform(*SENSOR_TYPES[sensor_type]["warn"]), 1)
        inp = f"It's at {val} in the {loc}"
        nlo, nhi = SENSOR_TYPES[sensor_type]["normal"]
        if val < nlo or val > nhi:
            direction = "high" if val > nhi else "low"
            tc = {"name": "send_alert", "arguments": {"level": "warning", "message": f"{sensor_type} at {loc} is {val} ({direction})", "zone": loc}}
            resp = f"Got it — {val} is outside the normal range of {nlo}-{nhi} for {sensor_type.replace('_', ' ')}. I've flagged it as a warning."
            tools = pick_tools(must_include_names=["send_alert"], pool=INDUSTRIAL_TOOLS)
        else:
            tc = None
            resp = f"Actually, {val} is within normal range ({nlo}-{nhi}) for {sensor_type.replace('_', ' ')}. It looks fine to me. If it seems wrong compared to what you're experiencing, the sensor might need calibration."
            tools = pick_tools(pool=INDUSTRIAL_TOOLS)

        return {
            "system": pick(SYSTEMS),
            "tools": tools,
            "history": history,
            "input": inp,
            "think": f"Now I have the details. {sensor_type} at {loc} is {val}. Normal range is {nlo}-{nhi}. Let me assess.",
            "response": resp,
            "tool_call": tc,
            "category": "conversation",
        }

    elif conv_type == "correction":
        room1 = pick(ROOMS)
        room2 = pick([r for r in ROOMS if r != room1])
        history = [
            {"role": "user", "content": f"Turn off the {room1} lights"},
            {"role": "assistant", "content": f"Done, {room1} lights are off."},
        ]
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["turn_off"], pool=SMART_HOME_TOOLS),
            "history": history,
            "input": f"Wait, I meant the {room2} lights, not {room1}",
            "think": f"The user corrected me — they wanted {room2} lights off, not {room1}. I should fix this.",
            "response": f"Sorry about that! Turning off the {room2} lights instead.",
            "tool_call": {"name": "turn_off", "arguments": {"device": f"{room2} lights"}},
            "category": "conversation",
        }

    else:  # continuation
        person = pick(PEOPLE)
        history = [
            {"role": "user", "content": f"What's the weather like today?"},
            {"role": "assistant", "content": "Sure, which city?"},
        ]
        city = pick(["London", "Tokyo", "New York", "Paris", "Sydney", "Toronto", "Berlin", "Seoul"])
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["get_weather"], pool=GENERAL_TOOLS),
            "history": history,
            "input": city,
            "think": f"The user said {city} — following up from the weather question.",
            "response": f"Checking the weather in {city}.",
            "tool_call": {"name": "get_weather", "arguments": {"location": city}},
            "category": "conversation",
        }


# ── Instruction following ──────────────────────────────────────────────────

def gen_instruction():
    inst_type = pick(["exact_value", "specific_mode", "negative_constraint", "multi_param"])

    if inst_type == "exact_value":
        level = pick(BRIGHTNESS_LEVELS)
        room = pick(ROOMS)
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["set_brightness"], pool=SMART_HOME_TOOLS),
            "input": f"Set the {room} lights to exactly {level}%",
            "think": f"Exact value requested: {level}%. I must use this exact number.",
            "response": f"{room.capitalize()} lights set to {level}%.",
            "tool_call": {"name": "set_brightness", "arguments": {"device": f"{room} lights", "level": level}},
            "category": "instruction",
        }

    elif inst_type == "specific_mode":
        temp = pick(TEMPERATURES)
        mode = pick(["heat", "cool"])
        mode_word = "heating" if mode == "heat" else "cooling"
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["set_temperature"], pool=SMART_HOME_TOOLS),
            "input": f"Set temperature to {temp}, {mode_word} mode only — not auto",
            "think": f"The user specifically wants {mode_word} mode at {temp}°C, not auto.",
            "response": f"Set to {temp}°C, {mode_word} only.",
            "tool_call": {"name": "set_temperature", "arguments": {"temperature": temp, "mode": mode}},
            "category": "instruction",
        }

    elif inst_type == "negative_constraint":
        event = pick(["belt squeaking on conveyor 3", "minor water stain on ceiling", "door sensor intermittent",
                      "slight hum from transformer", "paint peeling in hallway"])
        severity = pick(["info", "warning"])
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["log_event"], pool=INDUSTRIAL_TOOLS),
            "input": f"Just log this, don't send any alerts: {event}",
            "think": f"The user explicitly said log only, no alerts. I'll use log_event, not send_alert.",
            "response": f"Logged: {event}. No alert sent.",
            "tool_call": {"name": "log_event", "arguments": {"event": event, "severity": severity}},
            "category": "instruction",
        }

    else:  # multi_param
        pid = pick(PUMP_IDS)
        rate = pick(["low", "medium"])
        return {
            "system": pick(SYSTEMS),
            "tools": pick_tools(must_include_names=["start_pump"], pool=INDUSTRIAL_TOOLS),
            "input": f"Start {pid} at {rate} flow — not high, specifically {rate}",
            "think": f"The user emphasized {rate} flow rate for {pid}. I must use exactly {rate}.",
            "response": f"{pid.replace('_', ' ').capitalize()} started at {rate} flow.",
            "tool_call": {"name": "start_pump", "arguments": {"pump_id": pid, "flow_rate": rate}},
            "category": "instruction",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n_samples=15000, eval_ratio=0.05):
    generators = [
        # Natural commands — many sub-generators for variety
        (gen_turn_on, 0.06),
        (gen_turn_off, 0.06),
        (gen_set_brightness, 0.04),
        (gen_set_temp, 0.04),
        (gen_implicit_hot, 0.03),
        (gen_implicit_cold, 0.03),
        (gen_implicit_dark, 0.02),
        (gen_implicit_bright, 0.02),
        (gen_lock_door, 0.02),
        (gen_music, 0.02),
        (gen_timer_alarm, 0.03),
        (gen_blinds, 0.02),
        (gen_vacuum, 0.02),
        (gen_industrial_command, 0.08),
        (gen_general_command, 0.06),
        # Sensor understanding
        (gen_sensor, 0.12),
        # Knowledge & conversation
        (gen_question, 0.12),
        (gen_conversation, 0.07),
        # Advanced skills
        (gen_novel_tool, 0.06),
        (gen_refusal, 0.04),
        (gen_instruction, 0.04),
    ]

    total_w = sum(w for _, w in generators)
    generators = [(g, w / total_w) for g, w in generators]
    counts = [(g, max(1, int(n_samples * w))) for g, w in generators]
    total = sum(c for _, c in counts)
    if n_samples - total > 0:
        counts[0] = (counts[0][0], counts[0][1] + n_samples - total)

    samples = []
    for gen, count in counts:
        for _ in range(count):
            try:
                samples.append(gen())
            except Exception as e:
                print(f"Error in {gen.__name__}: {e}")
                import traceback; traceback.print_exc()

    random.shuffle(samples)
    n_eval = int(len(samples) * eval_ratio)
    eval_samples, train_samples = samples[:n_eval], samples[n_eval:]

    os.makedirs("data", exist_ok=True)
    for name, data in [("data/train.jsonl", train_samples), ("data/eval.jsonl", eval_samples)]:
        with open(name, "w") as f:
            for s in data:
                f.write(json.dumps({"text": format_sample(s), "category": s["category"]}) + "\n")
    for name, data in [("data/train_openai.jsonl", train_samples), ("data/eval_openai.jsonl", eval_samples)]:
        with open(name, "w") as f:
            for s in data:
                f.write(json.dumps(to_openai(s)) + "\n")

    cats = Counter(s["category"] for s in samples)
    has_tool = sum(1 for s in samples if s.get("tool_call"))
    has_hist = sum(1 for s in samples if s.get("history"))

    # Measure actual uniqueness
    unique_inputs = len(set(s["input"] for s in samples))

    print(f"Generated {len(samples)} samples ({unique_inputs} unique inputs, {unique_inputs/len(samples)*100:.1f}% unique):")
    print(f"  Train: {len(train_samples)}, Eval: {n_eval}")
    print(f"  With tool calls: {has_tool} ({has_tool/len(samples)*100:.1f}%)")
    print(f"  Multi-turn: {has_hist}")
    print(f"\nBy skill:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(samples)*100:.1f}%)")


if __name__ == "__main__":
    generate_dataset(15000)
