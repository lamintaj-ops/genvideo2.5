# clip_selector.py (Zone-aware version)

import random
import pandas as pd

# Zone/folder mapping for Canto
ZONE_KEYWORDS = {
    "jumanji": ["jumanji", "zone d"],
    "ghostbusters": ["ghostbusters", "zone b", "ghost"],
    "zombieland": ["zombieland", "zone c", "zombie"],
    "family": ["hotel transylvania", "kids zone"],
    "surf": ["flowrider", "surf"],
    "food": ["food", "beverage", "restaurant"],
}

# -----------------------------------------
# Score by zone matching
# -----------------------------------------
def score_by_zone(folder_path: str, themes: list):
    """
    Give bonus points if clip is from the right zone/folder
    """
    folder = (folder_path or "").lower()
    score = 0
    
    for theme in themes:
        if theme in ZONE_KEYWORDS:
            for keyword in ZONE_KEYWORDS[theme]:
                if keyword in folder:
                    score += 5  # Big bonus for zone match
                    break
    
    return score

# -----------------------------------------
# Score จาก prompt
# -----------------------------------------
def score_by_prompt(top_tags: str, prompt_info: dict):
    tags = (top_tags or "").lower()
    score = 0

    for t in prompt_info["themes"]:
        if t in tags:
            score += 4

    if prompt_info["vibe"] in tags:
        score += 1

    # Penalize logo/text heavy clips (watermarks, title cards, posters)
    if any(bad in tags for bad in ["logo", "watermark", "text", "title", "caption", "poster"]):
        score -= 8

    # Penalize extremely static scenes (we still want smooth, but not frozen)
    if any(bad in tags for bad in ["static", "still", "freeze"]):
        score -= 3

    return score

# -----------------------------------------
# Rank โดยไม่ใช้ mood
# -----------------------------------------
def rank_clips(df: pd.DataFrame, prompt_info: dict):

    df["prompt_score"] = df["top_tags"].apply(
        lambda t: score_by_prompt(t, prompt_info)
    )
    
    # Add zone bonus score
    if "folder_path" in df.columns:
        df["zone_score"] = df["folder_path"].apply(
            lambda f: score_by_zone(f, prompt_info["themes"])
        )
    else:
        df["zone_score"] = 0

    # ไม่ใช้ mood → ใช้เฉพาะ prompt score + zone score
    if "mood_motion" in df.columns:
        mm = df["mood_motion"].fillna(df["mood_motion"].median())
        mm_min, mm_max = float(mm.min()), float(mm.max())
        if mm_max > mm_min:
            mm_norm = (mm - mm_min) / (mm_max - mm_min)
        else:
            mm_norm = mm * 0 + 0.5

        vibe = prompt_info.get("vibe") or "upbeat"
        target = 0.40 if vibe == "relax" else 0.55
        df["motion_score"] = 1.0 - (mm_norm - target).abs()
        df["overall"] = df["prompt_score"] * 2 + df["zone_score"] * 3 + df["motion_score"] * 1.5
    else:
        df["overall"] = df["prompt_score"] * 2 + df["zone_score"] * 3

    return df.sort_values("overall", ascending=False)


# -----------------------------------------
# Simple Story Structure (ไม่ใช้ mood)
# -----------------------------------------
def build_story(df_ranked, n_shots=6):

    # 1) HOOK = คลิปอันดับ 1
    hook = df_ranked.head(1)

    # 2) END = คลิปอันดับ 2
    end = df_ranked.iloc[1:2]

    # 3) BODY = ที่เหลือ
    body = df_ranked.iloc[2 : 2 + (n_shots - 2)]

    story = pd.concat([hook, body, end], ignore_index=True)
    return story


# -----------------------------------------
# MAIN SELECTOR (เรียกจาก generate_edit.py)
# -----------------------------------------
def select_clips(df: pd.DataFrame, prompt_info: dict, n_shots: int = 6):
    df_ranked = rank_clips(df.copy(), prompt_info)
    return df_ranked.head(n_shots)

