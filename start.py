#!/usr/bin/env python3
"""
MOOCs Auto-Generation System — Quick Launcher
Run: python start.py
"""
import sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent


def banner():
    print("═" * 58)
    print("  🎓 Automatic MOOCs Generation System Based on Speech")
    print("  Tamkang University CSIE · Ya-Xin Guo")
    print("  Dr. Chih-Yung Chang · Dr. Shi-Hjung Wu")
    print("═" * 58)


def check_env():
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required (found {sys.version.split()[0]})")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")

    required = [
        ("fastapi",              "FastAPI"),
        ("uvicorn",              "Uvicorn"),
        ("torch",                "PyTorch"),
        ("whisper",              "Whisper"),
        ("cv2",                  "OpenCV"),
        ("pptx",                 "python-pptx"),
        ("transformers",         "Transformers"),
        ("sentence_transformers","Sentence-BERT"),
        ("requests",             "requests"),
    ]
    optional = [
        ("mediapipe",   "MediaPipe (pose extraction)"),
        ("noisereduce", "noisereduce (noise reduction)"),
        ("opencc",      "OpenCC (traditional Chinese)"),
        ("keybert",     "KeyBERT (keywords)"),
        ("clip",        "CLIP (image alignment)"),
        ("f5_tts",      "F5-TTS (real voice synthesis)"),
    ]
    missing = []
    for pkg, name in required:
        try:    __import__(pkg); print(f"  ✅ {name}")
        except ImportError: print(f"  ❌ {name}"); missing.append(pkg)

    print("\n  Optional:")
    for pkg, name in optional:
        try:    __import__(pkg); print(f"  ✅ {name}")
        except ImportError: print(f"  ⚠️  {name}")

    # wav2lip
    wl = BASE / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth"
    print(f"  {'✅' if wl.exists() else '⚠️ '} wav2lip_gan.pth {'found' if wl.exists() else '(not found — real mode disabled)'}")

    # ffmpeg
    ok = subprocess.run(["ffmpeg","-version"], capture_output=True).returncode == 0
    print(f"  {'✅' if ok else '⚠️ '} FFmpeg {'available' if ok else '(not found — real mode disabled)'}")

    # Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models",[])]
            print(f"\n  ✅ Ollama: {models or ['(no models)']}")
            if not any("llama3" in m for m in models):
                print("     ⚠️  LLaMA3 not found. Run: ollama pull llama3")
        else:
            print("\n  ⚠️  Ollama responding but unhealthy")
    except Exception:
        print("\n  ⚠️  Ollama not running — start: ollama serve")

    # .env setup
    from pathlib import Path
    env_file = BASE / ".env"
    env_example = BASE / ".env.example"
    if not env_file.exists() and env_example.exists():
        print(f"\n  ⚠️  No .env found — copy from .env.example:")
        print(f"     cp .env.example .env")
    elif env_file.exists():
        print(f"\n  ✅ .env found")

    # CPU performance note
    import torch
    if not torch.cuda.is_available():
        print(f"\n  ℹ️  CPU-only mode:")
        print(f"     • Whisper: base model (auto-selected, ~140 MB)")
        print(f"     • Silero-VAD: CPU-native ✓")
        print(f"     • MediaPipe: CPU-optimized ✓")
        print(f"     • Sentence-BERT: CPU-compatible ✓")
        print(f"     • CLIP: CPU-compatible (~1s/image) ✓")
        print(f"     • BLIP-2: very slow on CPU — CLIP recommended")
        print(f"     • Step 6 real mode: demo mode recommended on CPU")
    else:
        device_name = torch.cuda.get_device_name(0)
        print(f"\n  ✅ GPU detected: {device_name}")
        print(f"     • Whisper: large-v3 enabled")

    # Demo video
    demo = BASE / "output" / "完整moocs影片產出.mp4"
    print(f"\n  {'✅' if demo.exists() else '⚠️ '} Demo video: {'found' if demo.exists() else 'missing — place at output/完整moocs影片産出.mp4'}")

    if missing:
        print(f"\n❌ Missing required packages: {missing}")
        print("   Fix: pip install -r requirements.txt")
        sys.exit(1)
    print()


def main():
    banner()
    check_env()
    print("🚀 Starting FastAPI…")
    print("   Simple UI : http://localhost:8000")
    print("   Pro UI    : http://localhost:8000/pro")
    print("   API Docs  : http://localhost:8000/docs")
    print("   Press Ctrl+C to stop\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload",
        "--reload-dir", str(BASE / "backend"),
    ], cwd=str(BASE))


if __name__ == "__main__":
    main()
