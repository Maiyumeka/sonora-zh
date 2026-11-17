"""Local, one-shot separation bridge. No server and no audio upload."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = "htdemucs.yaml"
LABELS = {
    "mix": "原曲", "vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "其他乐器",
    "instrumental": "伴奏", "guitar": "吉他", "piano": "钢琴", "strings": "弦乐",
    "lead vocals": "主唱", "backing vocals": "和声",
}


def cached_tracks(target, manifest):
    tracks = manifest.get("tracks")
    if tracks is None:
        # Upgrade existing completed caches without running inference again.
        files = sorted(target.glob("*.wav"), key=lambda p: (p.name != "mix.wav", p.name))
        tracks = [{"file": p.name, "label": LABELS.get(p.stem.lower(), p.stem)}
                  for p in files if not p.name.endswith(".tmp.wav")]
    if not tracks or tracks[0].get("file") != "mix.wav" or len(tracks) < 2:
        return None
    if not all(Path(t["file"]).name == t["file"] and (target / t["file"]).is_file() for t in tracks):
        return None
    return tracks


def status(message):
    print(f"SONORA_STATUS {message}", flush=True)


def cache_directory(digest: str, model: str):
    return ROOT / "songs" / (digest[:24] + "-" + Path(model).stem + "-v1")


def run(source: Path, device: str, model: str = MODEL):
    started = time.monotonic()
    for key, folder in [("TEMP", "tmp"), ("TMP", "tmp"), ("TORCH_HOME", "cache/torch"), ("HF_HOME", "cache/huggingface"), ("CUDA_CACHE_PATH", "cache/cuda-kernels"), ("NUMBA_CACHE_DIR", "cache/numba"), ("MPLCONFIGDIR", "cache/matplotlib")]:
        directory = ROOT / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    if not source.is_file():
        raise ValueError("找不到选中的音频文件")
    status("检查歌曲缓存…")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    target = cache_directory(digest.hexdigest(), model)
    ready = target / "ready.json"
    if ready.is_file():
        manifest = json.loads(ready.read_text(encoding="utf-8"))
        tracks = cached_tracks(target, manifest)
        if tracks and manifest.get("model") == model:
            if manifest.get("tracks") != tracks:
                manifest["tracks"] = tracks
                temporary = target / "ready.tmp.json"
                temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(ready)
            status("找到已有分轨，直接载入")
            print(f"SONORA_DONE {target}", flush=True)
            return

    import imageio_ffmpeg
    import numpy as np
    import soundfile as sf
    import torch
    # Keep CPU use bounded, including when inference runs on the GPU.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    use_gpu = device != "cpu" and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_per_process_memory_fraction(0.5)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    bin_dir = ROOT / "bin"
    bin_dir.mkdir(exist_ok=True)
    ffmpeg = bin_dir / "ffmpeg.exe"
    if not ffmpeg.is_file():
        shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg)
    os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = str(ROOT / "models")
    target.mkdir(parents=True, exist_ok=True)
    status("解码歌曲（仅在本机处理）…")
    # Bound decoding so a long recording cannot fill RAM or disk by accident.
    subprocess.run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-t", "601", "-vn", "-ar", "44100", "-ac", "2",
        "-c:a", "pcm_s16le", str(target / "mix.wav"),
    ], check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    info = sf.info(target / "mix.wav")
    if info.duration > 600:
        raise ValueError("省资源模式目前支持最长 10 分钟的单首歌曲，请先截取片段")
    if info.duration <= 0:
        raise ValueError("音频文件没有可读取的声音")

    from audio_separator.separator import Separator
    models = ROOT / "models"
    mode_label = "六轨" if model == "htdemucs_6s.yaml" else "四轨"
    status("加载" + mode_label + "模型（首次需要下载），使用" + ("显卡加速" if use_gpu else "CPU 省资源模式") + "…")
    separator = Separator(
        output_dir=str(target), model_file_dir=str(models), output_format="WAV",
        use_soundfile=True, sample_rate=44100, normalization_threshold=1.0,
        demucs_params={"segment_size": 6, "shifts": 0, "overlap": 0.25, "segments_enabled": True},
        log_level=logging.WARNING,
    )
    separator.load_model(model_filename=model)
    status("正在运行" + mode_label + "分离，请稍候…")
    infer_started = time.monotonic()
    outputs = separator.separate(str(target / "mix.wav"), custom_output_names={s: s for s in LABELS if s != "mix"})
    inference_seconds = time.monotonic() - infer_started
    status("整理分轨并校验长度…")
    tracks = [{"file": "mix.wav", "label": "原曲"}]
    if not outputs:
        raise RuntimeError("模型没有返回分轨文件")
    import re
    for output in outputs:
        candidate = Path(output) if Path(output).is_absolute() else target / output
        if not candidate.is_file() or candidate.parent.resolve() != target.resolve() or candidate.name == "mix.wav":
            raise RuntimeError("模型返回了无效的分轨路径")
        match = re.search(r"_\(([^)]+)\)_", candidate.stem)
        stem = match.group(1) if match else candidate.stem
        samples, rate = sf.read(candidate, dtype="float32", always_2d=True)
        if rate != 44100 or samples.shape[1] != 2:
            raise RuntimeError("分轨采样格式不一致")
        # Preserve alignment; remove only model padding, never shift samples.
        if abs(len(samples) - info.frames) > 44100:
            raise RuntimeError("分轨长度与原曲不一致")
        samples = samples[:info.frames]
        if len(samples) < info.frames:
            samples = np.pad(samples, ((0, info.frames - len(samples)), (0, 0)))
        destination = candidate
        temporary = candidate.with_name(candidate.stem + ".tmp.wav")
        sf.write(temporary, samples, 44100, subtype="PCM_16")
        temporary.replace(destination)
        tracks.append({"file": destination.name, "label": LABELS.get(stem.lower(), stem)})
    metrics = {
        "tracks": tracks,
        "source": source.name, "model": model, "seconds": info.duration,
        "inference_seconds": round(inference_seconds, 2),
        "reserved_gpu_mb": round(torch.cuda.max_memory_reserved() / 1024**2) if use_gpu else 0,
        "elapsed_seconds": round(time.monotonic() - started, 2), "gpu": use_gpu,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024**2) if use_gpu else 0,
    }
    ready = target / "ready.json"
    temporary = target / "ready.tmp.json"
    temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(ready)
    print("SONORA_METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"SONORA_DONE {target}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--model", choices=["htdemucs.yaml", "htdemucs_6s.yaml"], default=MODEL)
    args = parser.parse_args()
    try:
        run(args.source.resolve(), args.device, args.model)
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"SONORA_ERROR 拆轨未完成：{error}", flush=True)
        sys.exit(1)
