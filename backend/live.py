"""Bounded, persistent GPU separation for visualisation only. No audio output/files."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
WINDOW_SECONDS = 6
OUTPUT_SECONDS = 2
RATE = 44100
PRE_ROLL = 4096  # One FFT window before the displayed segment prevents periodic blank curves.
LABELS = {'mix': '原声', 'drums': '鼓', 'bass': '贝斯', 'other': '其他乐器',
          'vocals': '人声', 'guitar': '吉他', 'piano': '钢琴'}

def event(**value):
    sys.stdout.buffer.write((json.dumps(value, ensure_ascii=False) + '\n').encode('utf-8'))
    sys.stdout.buffer.flush()

def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise EOFError('实时输入已关闭')
        data.extend(block)
    return data

def run(model_name, input_rate):
    event(type='status', text='加载实时模型并预热显卡…')
    for key, folder in [('TORCH_HOME', 'cache/torch'), ('CUDA_CACHE_PATH', 'cache/cuda-kernels'),
                        ('NUMBA_CACHE_DIR', 'cache/numba'), ('TEMP', 'tmp'), ('TMP', 'tmp')]:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from math import gcd
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('实时模式需要兼容的 NVIDIA 显卡；这台电脑请使用离线拆轨。')
    torch.cuda.set_per_process_memory_fraction(0.5)
    vendored = Path(importlib.util.find_spec('audio_separator').origin).parent / 'separator/uvr_lib_v5'
    sys.path.insert(0, str(vendored))
    from demucs.pretrained import get_model
    from demucs.apply import apply_model, demucs_segments
    # Local repo only: this path never downloads weights or resolves a network URL.
    model = demucs_segments('6', get_model(Path(model_name).stem, repo=ROOT / 'models')).eval().to('cuda')
    names = ['原声'] + [LABELS.get(s, s) for s in model.sources]
    with torch.inference_mode():
        # Warm the complete normalisation/inference/transfer path, not just a zero forward.
        for _ in range(2):
            warm = torch.randn(2, RATE * WINDOW_SECONDS, device='cuda') * 0.03
            reference = warm.mean(0)
            mean, std = reference.mean(), reference.std().clamp_min(1e-7)
            warm_result = apply_model(model, ((warm - mean) / std)[None], shifts=0, split=False, device='cuda')[0]
            warm_cpu = (warm_result[..., 2 * RATE:4 * RATE] * std + mean).cpu().numpy()
            torch.cuda.synchronize()
            del warm, warm_result, warm_cpu
        event(type='ready', names=names, rate=RATE, frames=RATE * OUTPUT_SECONDS + PRE_ROLL, pre_roll=PRE_ROLL,
              reserved_mb=round(torch.cuda.memory_reserved() / 1024**2))
        input_frames = input_rate * WINDOW_SECONDS
        while True:
            line = sys.stdin.buffer.readline(4097)
            if not line:
                return
            if len(line) > 4096:
                raise ValueError('实时输入头过长')
            header = json.loads(line)
            if header.get('frames') != input_frames or header.get('rate') != input_rate:
                raise ValueError('实时输入格式改变，请重新开启实时模式')
            raw = read_exact(sys.stdin.buffer, input_frames * 2 * 4)
            samples = np.frombuffer(raw, dtype='<f4').reshape(-1, 2).T.copy()
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            if input_rate != RATE:
                divisor = gcd(input_rate, RATE)
                samples = resample_poly(samples, RATE // divisor, input_rate // divisor, axis=-1).astype(np.float32)
            samples = samples[:, :RATE * WINDOW_SECONDS]
            if samples.shape[-1] != RATE * WINDOW_SECONDS:
                raise ValueError('实时重采样长度不一致')
            start = 2 * RATE - PRE_ROLL
            end = 4 * RATE
            began = time.perf_counter()
            if np.max(np.abs(samples)) < 1e-6:
                stems = np.zeros((len(names) - 1, 2, OUTPUT_SECONDS * RATE + PRE_ROLL), dtype=np.float32)
            else:
                mix = torch.from_numpy(samples).to('cuda')
                reference = mix.mean(0)
                mean, std = reference.mean(), reference.std()
                std = std.clamp_min(1e-7)
                normal = (mix - mean) / std
                output = apply_model(model, normal[None], shifts=0, split=False, device='cuda')[0]
                stems = (output[..., start:end] * std + mean).cpu().numpy()
                del mix, normal, output
            duration = time.perf_counter() - began
            all_tracks = np.concatenate([samples[None, :, start:end], stems], axis=0)
            # Same fixed PCM16 scale as offline playback; never normalise individual stems.
            pcm = (np.clip(all_tracks.transpose(0, 2, 1), -1.0, 32767 / 32768) * 32768).astype('<i2')
            if pcm.shape != (len(names), OUTPUT_SECONDS * RATE + PRE_ROLL, 2):
                raise ValueError('实时分轨尺寸不一致')
            event(type='result', seq=header['seq'], frames=OUTPUT_SECONDS * RATE + PRE_ROLL, pre_roll=PRE_ROLL,
                  tracks=len(names), rate=RATE, seconds=round(duration, 4),
                  reserved_mb=round(torch.cuda.memory_reserved() / 1024**2))
            sys.stdout.buffer.write(pcm.tobytes(order='C'))
            sys.stdout.buffer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['htdemucs.yaml', 'htdemucs_6s.yaml'], required=True)
    parser.add_argument('--rate', type=int, required=True)
    args = parser.parse_args()
    try:
        if not 8000 <= args.rate <= 192000:
            raise ValueError('实时模式支持 8–192 kHz 音源')
        run(args.model, args.rate)
    except (BrokenPipeError, EOFError):
        pass
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        event(type='error', text=str(error))
        sys.exit(1)
