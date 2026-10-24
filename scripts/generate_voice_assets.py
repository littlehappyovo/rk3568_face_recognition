#!/usr/bin/env python3
"""Generate the five fixed prompts on a development PC, not during recognition.

Install sherpa-onnx==2.0.0 and numpy in a dedicated virtual environment first.
The vits-melo-tts-zh_en model is downloaded separately; see audio/README.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import sherpa_onnx

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "audio")
args = parser.parse_args()
model = args.model_dir.resolve()
args.output_dir.mkdir(parents=True, exist_ok=True)
engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(model / "model.onnx"), tokens=str(model / "tokens.txt"),
            lexicon=str(model / "lexicon.txt"), dict_dir=str(model / "dict")),
        num_threads=2, provider="cpu")))
prompts = {"morning": "早上好。", "noon": "中午好。", "afternoon": "下午好。",
           "evening": "晚上好。", "denied": "无权限，请联系管理员。"}
manifest = {"engine": "sherpa-onnx", "engine_version": sherpa_onnx.__version__,
            "model": "vits-melo-tts-zh_en", "model_sha256": hashlib.sha256(
                (model / "model.onnx").read_bytes()).hexdigest(), "files": []}
for key, text in prompts.items():
    audio = engine.generate(text, sid=0, speed=1.0)
    samples = np.asarray(audio.samples)
    duration = len(samples) / audio.sample_rate
    if not 0.25 < duration < 15 or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.001:
        raise RuntimeError(f"Invalid audio generated for {key}")
    # Normalize the model's very quiet raw output, keeping headroom and equal
    # speech loudness across phrases. Pad for codec/amp startup and shutdown.
    gain = min(0.8 / float(np.max(np.abs(samples))),
               0.14 / max(1e-8, float(np.sqrt(np.mean(samples * samples)))))
    normalized = np.clip(samples * gain, -0.8, 0.8)
    # RK809 on the target board was audibly verified using 48kHz stereo PCM.
    sample_rate = 48000
    count = round(len(normalized) * sample_rate / audio.sample_rate)
    resampled = np.interp(np.arange(count) * audio.sample_rate / sample_rate,
                          np.arange(len(normalized)), normalized)
    padded = np.concatenate((np.zeros(round(sample_rate * 0.2)), resampled,
                             np.zeros(round(sample_rate * 0.1))))
    pcm = (np.repeat(padded[:, None], 2, axis=1) * 32767).astype("<i2")
    target = args.output_dir / (key + ".wav")
    with wave.open(str(target), "wb") as output:
        output.setnchannels(2); output.setsampwidth(2); output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    manifest["files"].append({"file": target.name, "text": text,
                              "sample_rate": sample_rate, "channels": 2,
                              "duration_seconds": round(len(padded)/sample_rate, 3),
                              "speech_rms": round(float(np.sqrt(np.mean(normalized * normalized))), 4),
                              "peak": round(float(np.max(np.abs(normalized))), 4),
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    print(target.name, f"{duration:.2f}s", flush=True)
(args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
