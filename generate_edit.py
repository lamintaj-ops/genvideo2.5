# generate_edit.py - Fix Repetitive Scenes
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import json
import shutil

from ffmpeg_utils import (
    cut_random_segment,
    concat_videos_smooth,
    get_resolution,
    trim_music,
    apply_lut,
    export_ratios,
    mix_video_music
)

from prompt_rules import parse_prompt
from clip_selector import select_clips
from downloader import download_file

# ======================
# PATH SETTINGS
# ======================
TAGS_CSV = "canto_clip_tags_with_urls.csv"
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp_edit")
BGM_DIR = Path("bgm")

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ======================
# HELPER: Pick Random BGM (Download from GitHub for HuggingFace)
# ======================
def pick_bgm(vibe="upbeat"):
    """Download BGM from GitHub if not exists locally"""
    import requests
    
    # BGM URLs from GitHub repo (public)
    GITHUB_BGM_URLS = [
        "https://raw.githubusercontent.com/lamintaj-ops/genvideo/main/bgm/RM-Ride%20the%20Sky-02.mp3",
        "https://raw.githubusercontent.com/lamintaj-ops/genvideo/main/bgm/RM-Jump%20In-02.mp3",
        "https://raw.githubusercontent.com/lamintaj-ops/genvideo/main/bgm/RM-No%20Rules%20Here-02.mp3",
        "https://raw.githubusercontent.com/lamintaj-ops/genvideo/main/bgm/RM-Magic%20at%20Aquaverse-01.mp3",
        "https://raw.githubusercontent.com/lamintaj-ops/genvideo/main/bgm/JT-Ready%20for%20the%20Ride%20%28Remix%29.mp3",
    ]
    
    # Try local files first
    candidates = list(BGM_DIR.glob("*.mp3"))
    if candidates:
        return random.choice(candidates)
    
    # Download from GitHub if no local files
    BGM_DIR.mkdir(exist_ok=True)
    try:
        bgm_url = random.choice(GITHUB_BGM_URLS)
        filename = bgm_url.split('/')[-1].replace('%20', ' ').replace('%28', '(').replace('%29', ')')
        bgm_filename = BGM_DIR / filename
        
        if not bgm_filename.exists():
            print(f"⬇️ Downloading BGM from GitHub: {filename}")
            response = requests.get(bgm_url, timeout=30)
            response.raise_for_status()
            bgm_filename.write_bytes(response.content)
            print(f"✅ BGM downloaded: {bgm_filename.name}")
        
        return bgm_filename
    except Exception as e:
        print(f"⚠️ Failed to download BGM: {e}")
        return None

# ======================
# MAIN WORKFLOW
# ======================
def main(prompt: str):
    print("\n=== Aquaverse Auto Editing Engine (Anti-Stutter & Diverse) ===\n")
    print("Prompt:", prompt)

    prompt_info = parse_prompt(prompt)
    duration = prompt_info["duration"]
    shot_min = prompt_info["shot_len_min"]
    shot_max = prompt_info["shot_len_max"]

    # 1. Load Data
    df = pd.read_csv(TAGS_CSV)
    df = df[df["status"] == "ok"].copy()
    
    # Patch for missing columns
    if "download_url" not in df.columns:
        raise RuntimeError("Missing 'download_url' column.")
    if "mood_motion" not in df.columns:
        df["mood_motion"] = 0.5 

    # 2. Select Clips (เพิ่มความหลากหลาย)
    # คำนวณจำนวนช็อตที่ต้องใช้ (no transitions - modern hard cuts)
    avg_shot_len = (shot_min + shot_max) / 2
    n_shots = max(6, int(duration / avg_shot_len))
    
    print(f"Target duration: {duration}s")
    print(f"Average shot length: {avg_shot_len}s")
    print(f"Number of shots: {n_shots}")
    
    # 🔥 Trick: ขอคลิปมาเยอะๆ ก่อน (3 เท่า) แล้วค่อยมาสุ่มเลือกทีหลัง กันซ้ำ
    pool_size = n_shots * 3
    candidates = select_clips(df, prompt_info, n_shots=pool_size)
    
    # 🔥 Logic: สุ่มเลือกจาก Candidates และตัดตัวซ้ำ (Asset ID)
    if "asset_id" in candidates.columns:
        # ลบ Duplicate Asset ID เพื่อไม่ให้เอาไฟล์เดิมมาใช้ซ้ำ
        candidates = candidates.drop_duplicates(subset=["asset_id"])
    
    # ถ้าเหลือคลิปน้อยกว่าที่ต้องการ ก็ช่วยไม่ได้ แต่ถ้าเหลือเยอะ ให้สุ่มมา
    if len(candidates) > n_shots:
        chosen = candidates.sample(n=n_shots) # สุ่มมาตามจำนวนที่ต้องการ
    else:
        chosen = candidates # เอาเท่าที่มี

    print(f"Selected {len(chosen)} unique clips from pool.")

    log = {"prompt": prompt, "selected": []}
    cut_files = []

    # 3. Download & Cut
    print("\n--- Downloading & Cutting ---")
    for i, r in tqdm(chosen.iterrows(), total=len(chosen)):
        asset_id = str(r.get("asset_id"))
        url = r.get("download_url")

        local_video = TEMP_DIR / f"{asset_id}.mp4"
        cut_video = TEMP_DIR / f"cut_{i:03d}.mp4"

        if not local_video.exists():
            download_file(url, local_video)

        seg_len = random.uniform(shot_min, shot_max)
        
        # เรียกใช้ฟังก์ชันตัดแบบใหม่ (FPS=30)
        start, full_dur = cut_random_segment(local_video, cut_video, seg_len=seg_len)

        cut_files.append(cut_video)
        log["selected"].append({"asset_id": asset_id, "seg_len": seg_len})

    # 4. Concat (Modern hard cuts - no transitions)
    print("\n--- Concatenating (Modern Hard Cuts) ---")
    concat_out = OUTPUT_DIR / "concat_raw.mp4"
    concat_videos_smooth(cut_files, concat_out)

    # 5. Color Grade
    print("\n--- Applying Color Grade ---")
    graded = OUTPUT_DIR / "graded.mp4"
    apply_lut(concat_out, graded)

    # 6. Music Processing
    print("\n--- Processing Music ---")
    bgm = pick_bgm(prompt_info["vibe"])
    final_master = OUTPUT_DIR / "master_with_music.mp4"
    trimmed_bgm_wav = None

    if bgm:
        print(f"BGM Selected: {bgm.name}")
        temp_bgm_path = OUTPUT_DIR / "bgm_temp" 
        trimmed_bgm_wav = trim_music(bgm, temp_bgm_path, duration, fade=1.0)
        
        print("Mixing Video + Music...")
        mix_video_music(graded, trimmed_bgm_wav, final_master)
    else:
        print("⚠️ No BGM available. Video without background music.")
        shutil.copy(graded, final_master)

    # 7. Export Ratios
    print("\n--- Exporting Formats (9:16 & 16:9) ---")

    # =============================
    #   OUTRO (Logo + BG color)
    # =============================
    print("\n--- Adding Outro (Logo + Background) ---")

    from ffmpeg_utils import create_outro_sized, overlay_logo_scaled, concat_with_outro

    logo_path = "assets/brand_logo.png"

    # Create logo if not exists (for HuggingFace deployment)
    if not Path(logo_path).exists():
        from PIL import Image, ImageDraw, ImageFont
        Path("assets").mkdir(exist_ok=True)
        img = Image.new('RGBA', (800, 300), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('arial.ttf', 80)
        except:
            font = ImageFont.load_default()
        draw.text((50, 100), 'AQUAVERSE', fill=(43, 73, 126, 255), font=font)
        img.save(logo_path)
        print(f"✓ Created {logo_path}")
    outro_bg = OUTPUT_DIR / "outro_bg.mp4"
    outro_logo = OUTPUT_DIR / "outro_logo.mp4"
    final_with_outro = OUTPUT_DIR / "final_with_outro.mp4"

    # 1) สร้างพื้นหลังสี #2b497e ความยาว 2 วินาที
    create_outro_sized("#2b497e", duration=2, out_path=outro_bg, size="1920x1080")

    # 2) ใส่โลโก้ลงตรงกลางพื้นหลัง
    overlay_logo_scaled(outro_bg, logo_path, outro_logo, logo_scale=0.35)

    # 3) เอา outro ต่อท้ายคลิปสุดท้ายที่ใส่เพลงแล้ว
    # NOTE: outro concat is handled per-aspect-ratio below (prevents 9:16 outro scaling/cropping)

    # 4) ตั้งให้ final_master เป็นไฟล์ใหม่เพื่อนำไป export 16:9 / 9:16 ต่อ
    outputs = export_ratios(final_master, OUTPUT_DIR)

    # 9:16: rebuild from vertical-only sources (overwrite cropped 9:16 export)
    print("\n--- Rebuilding 9:16 (vertical sources only) ---")
    candidates_9x16 = select_clips(df, prompt_info, n_shots=n_shots * 10)
    if "asset_id" in candidates_9x16.columns:
        candidates_9x16 = candidates_9x16.drop_duplicates(subset=["asset_id"])

    cut_files_9x16 = []
    chosen_9x16 = 0
    for _, r in tqdm(candidates_9x16.iterrows(), total=len(candidates_9x16)):
        if chosen_9x16 >= n_shots:
            break

        asset_id = str(r.get("asset_id"))
        url = r.get("download_url")
        local_video = TEMP_DIR / f"{asset_id}.mp4"
        if not local_video.exists():
            download_file(url, local_video)

        w, h = get_resolution(local_video)
        if w <= 0 or h <= 0 or h <= w:
            continue

        seg_len = random.uniform(shot_min, shot_max)
        cut_video = TEMP_DIR / f"9x16_cut_{chosen_9x16:03d}.mp4"
        cut_random_segment(local_video, cut_video, seg_len=seg_len, target_w=1080, target_h=1920)
        cut_files_9x16.append(cut_video)
        chosen_9x16 += 1

    if chosen_9x16 < n_shots:
        print(f"Warning: only found {chosen_9x16}/{n_shots} vertical clips for 9:16.")

    if cut_files_9x16:
        concat_9x16 = OUTPUT_DIR / "concat_raw_9x16_vertical.mp4"
        graded_9x16 = OUTPUT_DIR / "graded_9x16_vertical.mp4"
        master_9x16 = OUTPUT_DIR / "master_with_music_9x16_vertical.mp4"
        final_9x16 = OUTPUT_DIR / "final_9x16.mp4"

        concat_videos_smooth(cut_files_9x16, concat_9x16)
        apply_lut(concat_9x16, graded_9x16)

        if trimmed_bgm_wav:
            mix_video_music(graded_9x16, trimmed_bgm_wav, master_9x16)
            shutil.copy(master_9x16, final_9x16)
        else:
            shutil.copy(graded_9x16, final_9x16)

        outputs["9:16"] = str(final_9x16)

    def append_outro(video_path: Path, outro_path: Path) -> Path:
        # Write to a new file to avoid Windows file-lock issues when the target is open in a player.
        out_final = OUTPUT_DIR / f"{video_path.stem}_with_outro.mp4"
        concat_with_outro(video_path, outro_path, out_final)
        return out_final

    outro_bg_9x16 = OUTPUT_DIR / "outro_bg_9x16.mp4"
    outro_logo_9x16 = OUTPUT_DIR / "outro_logo_9x16.mp4"
    create_outro_sized("#2b497e", duration=2, out_path=outro_bg_9x16, size="1080x1920")
    overlay_logo_scaled(outro_bg_9x16, logo_path, outro_logo_9x16, logo_scale=0.60)

    final_16x9_out = append_outro(OUTPUT_DIR / "final_16x9.mp4", outro_logo)
    final_9x16_out = append_outro(OUTPUT_DIR / "final_9x16.mp4", outro_logo_9x16)
    outputs["16:9"] = str(final_16x9_out)
    outputs["9:16"] = str(final_9x16_out)
    

    # Log & Done
    with open(OUTPUT_DIR / "edit_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("\n===== DONE! =====")
    for k, v in outputs.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    import sys
    
    # Support both interactive and non-interactive modes
    if len(sys.argv) > 1:
        # Command-line argument mode (for HuggingFace/automated calls)
        prompt = " ".join(sys.argv[1:]).strip()
    elif not sys.stdin.isatty():
        # Piped/redirected input mode (for subprocess)
        prompt = sys.stdin.read().strip()
    else:
        # Interactive mode (original behavior)
        prompt = input("ใส่ prompt: ").strip()
    
    if prompt:
        main(prompt)
    else:
        print("❌ No prompt provided!")
