# prompt_rules.py

import re

DEFAULT_DURATION = 20
DEFAULT_SHOT_LEN_MIN = 1.6
DEFAULT_SHOT_LEN_MAX = 2.6

KEYWORDS = {
    "themes": {
        "jumanji": ["jumanji", "jungle", "adventure", "jumanji theme", "jungle theme", "jumanji zone"],
        "ghostbusters": ["ghostbusters", "ghost", "spooky", "mysterious", "ghostbusters zone", "ghost theme"],
        "zombieland": ["zombieland", "zombie", "horror", "scary", "zombieland zone", "zombie theme"],
        "water": ["water", "splash", "pool", "wave pool", "playing in water"],
        "slide": ["waterslide", "slide", "tube", "slider"],
        "family": ["family", "kids", "children", "parents", "hotel transylvania"],
        "surf": ["surf", "flowrider", "surfing", "surfer", "wave"],
        "food": ["food", "eat", "drink", "restaurant", "burger", "pizza"],
    },
    "vibe": {
        "upbeat": ["upbeat", "energetic", "fun", "excited", "playful", "happy", "party"],
        "cinematic": ["cinematic", "epic", "dramatic"],
        "relax": ["relaxed", "chill", "slow", "calm", "lifestyle"],
        "spooky": ["spooky", "mysterious", "eerie", "creepy"],
        "thrilling": ["thrilling", "intense", "adrenaline", "scary", "horror"],
    }
}

def parse_prompt(prompt: str):
    p = prompt.lower()

    # duration เช่น "20 วินาที" หรือ "20s"
    duration = DEFAULT_DURATION
    m = re.search(r"(\d+)\s*(s|sec|secs|second|seconds|วินาที)", p)
    if m:
        duration = int(m.group(1))
    else:
        # ถ้ามีตัวเลขลอย ๆ ให้เอาตัวแรก
        m2 = re.search(r"\b(\d+)\b", p)
        if m2:
            duration = int(m2.group(1))

    themes = []
    for k, words in KEYWORDS["themes"].items():
        if any(w in p for w in words):
            themes.append(k)

    vibe = None
    for k, words in KEYWORDS["vibe"].items():
        if any(w in p for w in words):
            vibe = k
            break

    return {
        "duration": duration,
        "themes": themes if themes else ["water"],
        "vibe": vibe or "upbeat",
        "shot_len_min": DEFAULT_SHOT_LEN_MIN,
        "shot_len_max": DEFAULT_SHOT_LEN_MAX,
    }
