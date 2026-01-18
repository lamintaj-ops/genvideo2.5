import subprocess
from pathlib import Path
import random
import cv2
import os

def run(cmd):
    # capture_output=True  Error
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"\n FFmpeg Failed! Command: {' '.join(cmd[:5])}...")
        print("="*40)
        print(process.stderr)
        print("="*40)
        # If NVENC is listed but not usable (missing driver/GPU/session), retry once with libx264.
        if "h264_nvenc" in cmd:
            try:
                cmd2 = list(cmd)
                for i in range(len(cmd2) - 1):
                    if cmd2[i] == "-c:v" and cmd2[i + 1] == "h264_nvenc":
                        cmd2[i + 1] = "libx264"
                        break

                def _drop_flag(flag: str, takes_value: bool = True):
                    nonlocal cmd2
                    while flag in cmd2:
                        j = cmd2.index(flag)
                        del cmd2[j]
                        if takes_value and j < len(cmd2):
                            del cmd2[j]

                _drop_flag("-cq", True)
                _drop_flag("-rc", True)
                _drop_flag("-b:v", True)
                _drop_flag("-maxrate", True)
                _drop_flag("-bufsize", True)
                _drop_flag("-profile:v", True)

                if "-preset" in cmd2:
                    p = cmd2.index("-preset")
                    if p + 1 < len(cmd2) and str(cmd2[p + 1]).startswith("p"):
                        cmd2[p + 1] = "veryfast"
                else:
                    cmd2.extend(["-preset", "veryfast"])

                if "-crf" not in cmd2:
                    cmd2.extend(["-crf", "20"])

                process2 = subprocess.run(cmd2, capture_output=True, text=True)
                if process2.returncode == 0:
                    return
            except Exception:
                pass

        raise subprocess.CalledProcessError(process.returncode, cmd, output=process.stdout, stderr=process.stderr)

def safe(p):
    return str(p).replace("\\", "/")

def low_resource_mode():
    v = os.environ.get("VIDEOSCORE_LOW_RESOURCE", "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False

    # Auto-enable on Railway-like environments unless explicitly disabled.
    for k in (
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_PUBLIC_DOMAIN",
        "RAILWAY_STATIC_URL",
    ):
        if os.environ.get(k):
            return True

    return False

def ffmpeg_resource_args():
    # Limit threads to reduce RAM spikes / CPU contention in small containers.
    if not low_resource_mode():
        return []
    return ["-threads", "1", "-filter_threads", "1"]

# ---------------------------
# 1) Read video duration
# ---------------------------
def get_duration(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frames / fps if frames > 0 else 0.0

def get_resolution(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h

def find_smooth_start(
    video_path: Path,
    seg_len: float,
    sample_rate_fps: float = 6.0,
    edge_seconds: float = 2.0,
    min_motion: float = 0.8,
    target_quantile: float = 0.35,
    candidates: int = 18,
    max_samples_per_candidate: int = 18,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    window_frames = max(1, int(round(seg_len * fps)))
    max_start_frame = total_frames - window_frames
    if max_start_frame <= 1:
        cap.release()
        return 0.0

    edge_frames = int(round(edge_seconds * fps))
    lo = max(0, edge_frames)
    hi = max(lo, max_start_frame - edge_frames)
    if hi <= lo:
        lo = 0
        hi = max_start_frame

    stride = max(1, int(round(fps / max(1.0, sample_rate_fps))))
    samples = max(6, int(round(seg_len * max(1.0, sample_rate_fps))))
    samples = min(samples, int(max_samples_per_candidate))

    def motion_stats(start_frame: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev_small = None
        diffs = []

        for _ in range(samples):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
            if prev_small is not None:
                diff = cv2.absdiff(small, prev_small)
                diffs.append(float(cv2.mean(diff)[0]))
            prev_small = small

            for _ in range(stride - 1):
                cap.grab()

        if len(diffs) < 3:
            return None
        avg = sum(diffs) / len(diffs)
        var = sum((d - avg) ** 2 for d in diffs) / len(diffs)
        return avg, var

    results = []
    for _ in range(max(1, int(candidates))):
        start_frame = random.randint(lo, hi)
        st = motion_stats(start_frame)
        if st is None:
            continue
        avg, var = st
        results.append((start_frame, avg, var))

    cap.release()
    if not results:
        return None

    avgs = [avg for _, avg, _ in results]
    eligible = [a for a in avgs if a >= float(min_motion)]
    if eligible:
        eligible_sorted = sorted(eligible)
        q = min(0.95, max(0.05, float(target_quantile)))
        target = eligible_sorted[int(round((len(eligible_sorted) - 1) * q))]
    else:
        target = sorted(avgs)[int(round((len(avgs) - 1) * 0.25))]

    best_key = None
    best_frame = None
    for start_frame, avg, var in results:
        key = (abs(avg - target), var)
        if best_key is None or key < best_key:
            best_key = key
            best_frame = start_frame

    start_time = best_frame / float(fps)
    return max(0.0, min(start_time, max_start_frame / float(fps)))

# ---------------------------
# 2) Cut & Normalize (Strict Mode)
# ---------------------------
def cut_random_segment(input_path: Path, out_path: Path, seg_len=2.0, target_w: int = 1920, target_h: int = 1080):
    dur = get_duration(input_path)
    if dur <= seg_len + 0.1:
        start = 0.0
    else:
        smooth = find_smooth_start(input_path, seg_len)
        if smooth is None:
            start = random.uniform(0, dur - seg_len)
        else:
            jitter = random.uniform(-0.15, 0.15)
            start = max(0.0, min(smooth + jitter, dur - seg_len))

    #  FIX:   Concat 
    # - scale=1920:1080 (HD)
    # - setsar=1 (Pixel Aspect Ratio )
    # - fps=30 ( Frame rate)
    vf = f"scale={int(target_w)}:{int(target_h)}:force_original_aspect_ratio=increase,crop={int(target_w)}:{int(target_h)},fps=30,format=yuv420p,setsar=1"
    cmd = [
        "ffmpeg", "-y",
        *ffmpeg_resource_args(),
        "-ignore_editlist", "1",
        "-ss", f"{start:.3f}",
        "-i", safe(input_path),
        "-t", f"{seg_len:.3f}",
        "-vf", vf,
        *h264_video_args(quality="cut"),
        "-pix_fmt", "yuv420p",
        "-an",               # 
        "-movflags", "+faststart",
        safe(out_path)
    ]
    run(cmd)
    return start, dur

# ---------------------------
# 3) Concatenate (Demuxer Method - Stable)
# ---------------------------
def concat_videos(file_list, output):
    #  list.txt
    list_path = Path("concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for fp in file_list:
            #  path  absolute  relative 
            f.write(f"file '{safe(fp.resolve())}'\n")

    #  -f concat  filter_complex 
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", safe(list_path),
        "-c", "copy",  # Copy stream   Normalize  ()
        safe(output)
    ]
    run(cmd)

def concat_videos_smooth(file_list, output, transition_duration=0.0):
    """
    Concatenate videos with modern hard cuts (no transitions)
    Applies consistent color grading and frame rate for smooth playback
    """
    if len(file_list) == 0:
        raise ValueError("No files to concatenate")
    
    if len(file_list) == 1:
        # Single file - process with consistent settings
        cmd = [
            "ffmpeg", "-y",
            *ffmpeg_resource_args(),
            "-i", safe(file_list[0]),
            "-vf", "fps=30,setsar=1,eq=contrast=1.02:brightness=0.01:saturation=1.05",
            *h264_video_args(quality="concat"),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            safe(output)
        ]
        run(cmd)
        return
    
    # Modern approach: Hard cuts with filter_complex for consistent processing
    inputs = []
    filter_parts = []
    
    for i, fp in enumerate(file_list):
        inputs.extend(["-i", safe(fp)])
        # Normalize each clip: fps, aspect ratio, color correction
        filter_parts.append(
            f"[{i}:v]fps=30,setsar=1,eq=contrast=1.02:brightness=0.01:saturation=1.05[v{i}]"
        )
    
    # Concat all normalized clips
    concat_inputs = "".join(f"[v{i}]" for i in range(len(file_list)))
    filter_parts.append(f"{concat_inputs}concat=n={len(file_list)}:v=1:a=0[out]")
    
    filter_complex = ";".join(filter_parts)
    
    cmd = [
        "ffmpeg",
        "-y",
        *ffmpeg_resource_args(),
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        *h264_video_args(quality="concat"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        safe(output),
    ]
    run(cmd)

# ---------------------------
# 4) Trim music (Stereo WAV)
# ---------------------------
def trim_music(music_path: Path, out_path: Path, target_duration: float, fade=0.8, start=0.0):
    end_fade = max(0.0, target_duration - fade)
    out_wav = out_path.with_suffix(".wav")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", safe(music_path),
        "-t", f"{target_duration:.2f}",
        "-vn", 
        "-af", f"volume=1.0,afade=t=in:ss=0:d={fade},afade=t=out:st={end_fade:.2f}:d={fade}",
        "-ac", "2", 
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        safe(out_wav)
    ]
    run(cmd)
    return out_wav

# ---------------------------
# 5) Mix Video + Music
# ---------------------------
def mix_video_music(video_in, audio_in, video_out):
    cmd = [
        "ffmpeg", "-y",
        "-i", safe(video_in),
        "-i", safe(audio_in),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        "-shortest",
        safe(video_out)
    ]
    run(cmd)

def estimate_luma_clipping(video_path: Path, samples: int = 6):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return 0.0, 0.0

    step = max(1, total_frames // max(1, samples))
    hi = 0.0
    lo = 0.0
    got = 0

    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            continue
        y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
        px = float(y.size)
        hi += float(cv2.countNonZero((y >= 250).astype("uint8"))) / px
        lo += float(cv2.countNonZero((y <= 5).astype("uint8"))) / px
        got += 1

    cap.release()
    if got == 0:
        return 0.0, 0.0
    return hi / got, lo / got

# ---------------------------
# 6) Apply LUT
# ---------------------------
def apply_lut(video_in, video_out, lut_file="lut/aquaverse_fun.cube"):
    if not Path(lut_file).exists():
        print(f"Warning: LUT file not found at {lut_file}, skipping grading.")
        import shutil
        shutil.copy(video_in, video_out)
        return

    hi_clip, lo_clip = estimate_luma_clipping(Path(video_in))
    # Default: vivid
    contrast = 1.06
    saturation = 1.18
    gamma = 1.02
    # If a lot of highlights/shadows already clipped, back off to avoid ""
    if hi_clip > 0.020 or lo_clip > 0.020:
        contrast = 1.03
        saturation = 1.12
        gamma = 1.00
    if hi_clip > 0.050 or lo_clip > 0.050:
        contrast = 1.02
        saturation = 1.08
        gamma = 1.00

    vf = f"lut3d={safe(lut_file)},eq=contrast={contrast}:saturation={saturation}:gamma={gamma}"

    cmd = [
        "ffmpeg",
        "-y",
        *ffmpeg_resource_args(),
        "-i",
        safe(video_in),
        "-vf",
        vf,
        *h264_video_args(quality="grade"),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        safe(video_out),
    ]
    run(cmd)

def create_outro(bg_color="#2b497e", duration=2, out_path="outro.mp4"):
    # convert hex (#2b497e)  0x2b497e
    hex_color = "0x" + bg_color.strip("#")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={hex_color}:s=1920x1080:d={duration}:r=30",
        *h264_video_args(quality="final"),
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    run(cmd)
    return out_path

def overlay_logo(bg_video, logo_path, out_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", safe(bg_video),
        "-i", safe(logo_path),
        "-filter_complex",
        "overlay=(W-w)/2:(H-h)/2:format=auto",  # center logo
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    run(cmd)
    return out_path

def concat_with_outro(video, outro, out_final):
    with open("concat_outro.txt", "w") as f:
        f.write(f"file '{safe(video)}'\n")
        f.write(f"file '{safe(outro)}'\n")

    cmd = [
        "ffmpeg", "-y",
        *ffmpeg_resource_args(),
        "-f", "concat",
        "-safe", "0",
        "-i", "concat_outro.txt",
        "-vf", "setsar=1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-movflags", "+faststart",
        safe(out_final)
    ]
    run(cmd)
    return out_final

def create_outro_sized(bg_color="#2b497e", duration=2, out_path="outro.mp4", size="1920x1080", fps=30):
    hex_color = "0x" + bg_color.strip("#")

    cmd = [
        "ffmpeg",
        "-y",
        *ffmpeg_resource_args(),
        "-f",
        "lavfi",
        "-i",
        f"color=c={hex_color}:s={size}:r={fps}",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t",
        str(duration),
        "-shortest",
        *h264_video_args(quality="final"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        safe(out_path),
    ]
    run(cmd)
    return out_path

def overlay_logo_scaled(bg_video, logo_path, out_path, logo_scale=0.6):
    bg_w, bg_h = get_resolution(Path(bg_video))
    target_w = max(2, int(round(bg_w * float(logo_scale)))) if bg_w > 0 else 0

    if target_w <= 0:
        raise RuntimeError(f"Could not read bg resolution for: {bg_video}")

    cmd = [
        "ffmpeg",
        "-y",
        *ffmpeg_resource_args(),
        "-i",
        safe(bg_video),
        "-i",
        safe(logo_path),
        "-filter_complex",
        f"[0:v]setsar=1[bg];"
        f"[1:v]setsar=1,scale={target_w}:-1:flags=lanczos[logo];"
        "[bg][logo]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        *h264_video_args(quality="final"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        safe(out_path),
    ]
    run(cmd)
    return out_path
 
# ---------------------------
# 7) Export Ratios
# ---------------------------
def export_ratios(input_video, out_dir: Path):
    outputs = {}
    
    base_args = [
        *h264_video_args(quality="final"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
    ]

    # 9:16
    o916 = out_dir / "final_9x16.mp4"
    cmd916 = [
        "ffmpeg", "-y",
        *ffmpeg_resource_args(),
        "-i", safe(input_video),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
        "-map", "0:v",
        "-map", "0:a?",
    ] + base_args + [safe(o916)]
    run(cmd916)
    outputs["9:16"] = safe(o916)

    # 16:9
    o169 = out_dir / "final_16x9.mp4"
    cmd169 = [
        "ffmpeg", "-y",
        *ffmpeg_resource_args(),
        "-i", safe(input_video),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1",
        "-map", "0:v",
        "-map", "0:a?",
    ] + base_args + [safe(o169)]
    run(cmd169)
    outputs["16:9"] = safe(o169)

    return outputs


# ---------------------------
# Encoder selection (NVENC)
# ---------------------------
_H264_ENCODER = None

def get_h264_encoder():
    global _H264_ENCODER
    if _H264_ENCODER is not None:
        return _H264_ENCODER

    if os.environ.get("VIDEOSCORE_NO_NVENC") == "1":
        _H264_ENCODER = "libx264"
        return _H264_ENCODER

    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
        )
        hay = (p.stdout or "") + "\n" + (p.stderr or "")
        if "h264_nvenc" in hay:
            # Runtime self-test: ffmpeg may list nvenc even if driver/GPU is not usable.
            test = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=128x72:rate=30",
                    "-t",
                    "0.1",
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p3",
                    "-cq",
                    "23",
                    "-b:v",
                    "0",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
            )
            _H264_ENCODER = "h264_nvenc" if test.returncode == 0 else "libx264"
        else:
            _H264_ENCODER = "libx264"
    except Exception:
        _H264_ENCODER = "libx264"
    return _H264_ENCODER

def h264_video_args(*, quality: str = "normal"):
    """
    quality:
      - "cut": fastest intermediate segments
      - "concat": smooth concat master
      - "grade": color-graded master (prefer higher quality)
      - "final": final exports/outros
    """
    enc = get_h264_encoder()
    if enc == "h264_nvenc":
        if low_resource_mode():
            preset = {"cut": "p2", "concat": "p3", "grade": "p4", "final": "p4"}.get(quality, "p3")
            cq = {"cut": "26", "concat": "25", "grade": "23", "final": "24"}.get(quality, "25")
        else:
            preset = {"cut": "p3", "concat": "p4", "grade": "p5", "final": "p5"}.get(quality, "p4")
            cq = {"cut": "23", "concat": "22", "grade": "19", "final": "20"}.get(quality, "22")
        return ["-c:v", "h264_nvenc", "-preset", preset, "-cq", cq, "-b:v", "0"]

    if low_resource_mode():
        preset = {"cut": "ultrafast", "concat": "ultrafast", "grade": "veryfast", "final": "veryfast"}.get(quality, "veryfast")
        crf = {"cut": "26", "concat": "26", "grade": "23", "final": "24"}.get(quality, "24")
    else:
        preset = {"cut": "veryfast", "concat": "veryfast", "grade": "medium", "final": "veryfast"}.get(quality, "veryfast")
        crf = {"cut": "20", "concat": "20", "grade": "18", "final": "20"}.get(quality, "20")
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf]

