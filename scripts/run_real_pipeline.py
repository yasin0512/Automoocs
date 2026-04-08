#!/usr/bin/env python3
"""
run_real_pipeline.py — Standalone Real-Mode Video Generation
============================================================
Use this script when you want to run the FULL pipeline outside the browser UI,
especially for batch processing or when GPU resources are available.

Prerequisites:
  pip install f5-tts
  git clone https://github.com/Rudrabha/Wav2Lip
  pip install -r Wav2Lip/requirements.txt
  Download wav2lip_gan.pth → Wav2Lip/checkpoints/
  ffmpeg on PATH

Usage:
  python scripts/run_real_pipeline.py \\
    --audio  path/to/lecture.wav \\
    --ref    path/to/teacher_ref.wav \\  # 5-10 sec reference for F5-TTS
    --face   path/to/teacher_face.mp4 \\ # face video for wav2lip
    --script "讲义文字内容…"             # or use --script_file

Optional:
  --syllabus  path/to/syllabus.txt
  --images    path/to/images_dir/
  --title     "Course Title"
  --theme     "Modern Blue"
  --out       path/to/output_dir/
"""

import argparse, sys, os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def main():
    ap = argparse.ArgumentParser(description="MOOCs Real-Mode Pipeline")
    ap.add_argument("--audio",       required=True,  help="Lecture audio file (wav/mp3/m4a)")
    ap.add_argument("--ref",         default="",     help="Teacher reference audio (wav, 5-10s)")
    ap.add_argument("--face",        default="",     help="Teacher face video (mp4)")
    ap.add_argument("--script",      default="",     help="Lecture script text")
    ap.add_argument("--script_file", default="",     help="Lecture script text file")
    ap.add_argument("--syllabus",    default="",     help="Syllabus text file")
    ap.add_argument("--images",      default="",     help="Directory with image materials")
    ap.add_argument("--title",       default="Lecture", help="Course title")
    ap.add_argument("--theme",       default="Modern Blue")
    ap.add_argument("--out",         default=str(ROOT/"output"), help="Output directory")
    ap.add_argument("--wav2lip_dir", default=str(ROOT/"Wav2Lip"), help="Path to Wav2Lip repo")
    ap.add_argument("--no_clean",    action="store_true", help="Skip LLM text cleaning")
    ap.add_argument("--no_align",    action="store_true", help="Skip syllabus alignment")
    ap.add_argument("--no_ppt",      action="store_true", help="Skip PPTX generation")
    ap.add_argument("--no_script",   action="store_true", help="Skip script generation")
    args = ap.parse_args()

    print("═"*58)
    print("  🎓 MOOCs Real-Mode Pipeline")
    print("  All steps: Audio → Clean → Align → PPT → Script → Video")
    print("═"*58)

    # Check real-mode tools
    try:
        import f5_tts; print("✅ F5-TTS available")
    except ImportError:
        print("❌ F5-TTS not installed — pip install f5-tts"); sys.exit(1)
    import subprocess
    if subprocess.run(["ffmpeg","-version"], capture_output=True).returncode != 0:
        print("❌ ffmpeg not found on PATH"); sys.exit(1)
    print("✅ ffmpeg available")
    ckpt = Path(args.wav2lip_dir) / "checkpoints" / "wav2lip_gan.pth"
    if not ckpt.exists():
        print(f"❌ wav2lip_gan.pth not found at {ckpt}"); sys.exit(1)
    print("✅ wav2lip_gan.pth found")

    from modules.pipeline import MOOCsPipeline
    syllabus = ""
    if args.syllabus and Path(args.syllabus).exists():
        syllabus = Path(args.syllabus).read_text(encoding="utf-8")
    images = []
    if args.images and Path(args.images).is_dir():
        images = [str(p) for p in Path(args.images).glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".webp")]
        print(f"✅ Found {len(images)} image(s)")
    script = args.script
    if args.script_file and Path(args.script_file).exists():
        script = Path(args.script_file).read_text(encoding="utf-8")

    pipeline = MOOCsPipeline({
        "output_dir":   args.out,
        "ppt_theme":    args.theme,
        "wav2lip_dir":  args.wav2lip_dir,
    })

    def progress(p, m):
        bar = "█" * int(p*30) + "░" * (30-int(p*30))
        print(f"\r[{bar}] {int(p*100):3d}% {m[:60]:<60}", end="", flush=True)

    print(f"\n🎯 Starting real-mode pipeline on: {args.audio}")
    result = pipeline.run(
        audio_path=args.audio,
        syllabus_text=syllabus,
        course_title=args.title,
        image_paths=images or None,
        ref_audio=args.ref or None,
        face_video=args.face or None,
        clean_method="ollama",
        video_mode="real",
        progress_cb=progress,
    )
    print()

    if result["success"]:
        print("\n✅ Pipeline complete!")
        print(f"   Video  : {result['output'].get('video_path','')}")
        print(f"   PPTX   : {result['output'].get('pptx_path','')}")
        print(f"   Script : {result['output'].get('script','')[:60]}…")
        print(f"   Time   : {result['elapsed_sec']}s")
    else:
        print("\n❌ Pipeline completed with errors:")
        for e in result["errors"]:
            print(f"   • {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
