"""
MOOCs Auto-Generation System — FastAPI Backend v3.1
====================================================
Unified input model:
  ① 課程語音檔  (audio_file)          — required
  ② 過往影片    (teacher_video)        — required, dual-use:
       → training action model
       → extracting teacher face for wav2lip
  ③ 課綱        (syllabus)             — optional text
  ④ 素材        (images, pptx_template) — optional

Output: complete MOOCs teaching video (MP4)
"""

import os, sys, json, shutil, logging, asyncio, concurrent.futures
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (FastAPI, UploadFile, File, Form,
                     HTTPException, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR    = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR= BASE_DIR / "frontend"
OUTPUT_DIR  = BASE_DIR / "output"
UPLOAD_DIR  = BASE_DIR / "uploads"

for d in [OUTPUT_DIR, UPLOAD_DIR]:
    d.mkdir(exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("moocs_api")

_ws_sessions: dict[str, WebSocket] = {}


async def _push(sid: str, pct: float, msg: str, step: str = ""):
    ws = _ws_sessions.get(sid)
    if ws:
        try:
            await ws.send_json({"type": "progress", "percent": round(pct),
                                "message": msg, "step": step})
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MOOCs system starting")
    yield
    logger.info("🛑 Stopped")


app = FastAPI(title="MOOCs Auto-Generation System",
              version="3.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static",
          StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── WebSocket ──────────────────────────────────────────────────
@app.websocket("/ws/{sid}")
async def ws_ep(ws: WebSocket, sid: str):
    await ws.accept()
    _ws_sessions[sid] = ws
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        _ws_sessions.pop(sid, None)


# ── Pages ──────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "templates" / "index.html"))

@app.get("/pro")
async def pro():
    return FileResponse(str(FRONTEND_DIR / "templates" / "pro.html"))


# ── Health / Status ────────────────────────────────────────────
@app.get("/api/health")
async def health(): return {"status": "ok", "version": "3.1.0"}


@app.get("/api/status")
async def status():
    import subprocess
    from modules.text_cleaner import TextCleaner
    from modules.ppt_generator import StableDiffusionGenerator
    be = TextCleaner().available_backends()
    try: import f5_tts; f5_ok = True
    except ImportError: f5_ok = False
    w2l_ok    = Path("Wav2Lip/checkpoints/wav2lip_gan.pth").exists()
    ffmpeg_ok = subprocess.run(["ffmpeg", "-version"],
                               capture_output=True).returncode == 0
    sd_ok     = StableDiffusionGenerator().is_available()
    demo_vid  = (OUTPUT_DIR / "完整moocs影片產出.mp4").exists()
    return {
        "ollama": be["ollama"], "openai": be["openai"],
        "stable_diffusion": sd_ok,
        "f5_tts": f5_ok, "wav2lip": w2l_ok, "ffmpeg": ffmpeg_ok,
        "real_mode_ready": f5_ok and w2l_ok and ffmpeg_ok,
        "demo_video_ready": demo_vid,
    }


# ── Helpers ────────────────────────────────────────────────────
def _sync(fn, *args, timeout=900, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(fn, *args, **kwargs).result(timeout=timeout)


def _save(upload: UploadFile, dest: Path) -> Path:
    with open(dest, "wb") as f: shutil.copyfileobj(upload.file, f)
    return dest


# ══════════════════════════════════════════════════════════════
# ① AUDIO — speech preprocessing (Whisper + speaker diarization)
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/audio")
async def step_audio(
    session_id:    str        = Form(...),
    audio_file:    UploadFile = File(...),
    enable_noise:  bool       = Form(True),
    enable_vad:    bool       = Form(True),
    enable_volume: bool       = Form(True),
):
    ext  = Path(audio_file.filename).suffix or ".wav"
    path = UPLOAD_DIR / f"{session_id}_audio{ext}"
    _save(audio_file, path)
    from modules.preprocessing import AudioProcessor
    proc = AudioProcessor()
    try:
        r   = _sync(proc.process_classroom_audio, str(path))
        wo  = UPLOAD_DIR / f"{session_id}_processed.wav"
        proc.save_audio(r["audio"], str(wo))
        await _push(session_id, 100, "Audio preprocessing complete", "audio")
        return {"success": True, "transcription": r["transcription"],
                "segments": r["segments"],
                "duration_sec": round(len(r["audio"]) / r["sample_rate"], 1)}
    except Exception as e: raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ② TEACHER VIDEO — dual-use: action model training + face extraction
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/teacher-video")
async def step_teacher_video(
    session_id:   str        = Form(...),
    teacher_video: UploadFile = File(...),
    run_training: bool       = Form(True),   # train action model
    extract_face: bool       = Form(True),   # extract face frames for wav2lip
    epochs:       int        = Form(10),
    batch_size:   int        = Form(8),
):
    """
    Single upload endpoint for the teacher's historical video.
    Runs in parallel:
      - Action model training (optional, background)
      - Face extraction for wav2lip lip sync
    """
    ext  = Path(teacher_video.filename).suffix or ".mp4"
    path = UPLOAD_DIR / f"{session_id}_teacher{ext}"
    _save(teacher_video, path)

    result: dict = {"success": True, "video_path": str(path)}

    # Face extraction — save path for Step 6 wav2lip
    face_out = UPLOAD_DIR / f"{session_id}_face.mp4"
    shutil.copy2(str(path), str(face_out))
    result["face_video_path"] = str(face_out)

    # Action model training — fire and forget (non-blocking)
    if run_training:
        async def _train_bg():
            try:
                from modules.action_modeling import run_full_training_pipeline
                train_out = OUTPUT_DIR / "training"
                train_out.mkdir(exist_ok=True)
                mp = _sync(run_full_training_pipeline,
                           str(path), str(train_out), 0.2, epochs, batch_size,
                           timeout=7200)
                await _push(session_id, 100, f"Action model trained: {mp}", "train")
            except Exception as e:
                await _push(session_id, 100, f"Training note: {e}", "train")
        asyncio.create_task(_train_bg())
        result["training_started"] = True

    await _push(session_id, 100, "Teacher video processed", "teacher_video")
    return result


# ══════════════════════════════════════════════════════════════
# ③ TEXT CLEANING
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/clean")
async def step_clean(
    session_id: str = Form(...),
    text:       str = Form(...),
    method:     str = Form("ollama"),
    openai_key: str = Form(""),
):
    from modules.text_cleaner import TextCleaner
    c = TextCleaner(openai_key=openai_key or None)
    try:
        r = _sync(c.clean, text, method)
        await _push(session_id, 100, "Text cleaning complete", "clean")
        return r
    except Exception as e: raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ④ SYLLABUS ALIGNMENT
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/align")
async def step_align(
    session_id: str   = Form(...),
    transcript: str   = Form(...),
    syllabus:   str   = Form(...),
    threshold:  float = Form(0.6),
    smart:      bool  = Form(True),
):
    from modules.syllabus_aligner import SyllabusAligner
    a = SyllabusAligner()
    try:
        r = _sync(a.full_pipeline, transcript, syllabus, threshold, smart)
        r["coverage"]["coverage_rate"] = float(r["coverage"]["coverage_rate"])
        for x in r["alignment"]: x["similarity"] = float(x["similarity"])
        await _push(session_id, 100, "Alignment complete", "align")
        return r
    except Exception as e: raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ⑤a IMAGE UPLOAD
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/upload-images")
async def upload_images(
    session_id: str              = Form(...),
    images:     List[UploadFile] = File(...),
):
    img_dir = UPLOAD_DIR / session_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for img in images:
        dest = img_dir / img.filename
        _save(img, dest); saved.append(str(dest))
    return {"success": True, "image_paths": saved, "count": len(saved)}


# ══════════════════════════════════════════════════════════════
# ⑤b PPTX TEMPLATE UPLOAD
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/upload-template")
async def upload_template(
    session_id:    str        = Form(...),
    template_file: UploadFile = File(...),
):
    """Accept user-uploaded .pptx template. Returns template path for use in /api/step/ppt."""
    if not template_file.filename.endswith(".pptx"):
        raise HTTPException(400, "Only .pptx files are accepted as templates")
    dest = UPLOAD_DIR / f"{session_id}_template.pptx"
    _save(template_file, dest)
    # Quick validation
    try:
        from pptx import Presentation
        prs = Presentation(str(dest))
        layouts = len(prs.slide_layouts)
    except Exception as e:
        raise HTTPException(400, f"Invalid PPTX file: {e}")
    return {"success": True, "template_path": str(dest),
            "filename": template_file.filename, "layouts": layouts}


# ══════════════════════════════════════════════════════════════
# ⑥ PPT GENERATION
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/ppt")
async def step_ppt(
    session_id:    str  = Form(...),
    transcript:    str  = Form(...),
    course_title:  str  = Form("Lecture"),
    key_points:    str  = Form("[]"),
    image_dir:     str  = Form(""),
    theme:         str  = Form("Modern Blue"),
    template_path: str  = Form(""),   # user-uploaded .pptx template (optional)
    use_sd:        bool = Form(False),
    use_blip2:     bool = Form(False),
):
    from modules.ppt_generator import PPTGenerator
    kp   = json.loads(key_points) if key_points else []
    imgs: List[str] = []
    if image_dir:
        p = UPLOAD_DIR / image_dir
        if p.exists():
            imgs = [str(x) for x in p.glob("*")
                    if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

    # Resolve template
    tpath = None
    if template_path and Path(template_path).exists():
        tpath = template_path
    elif (UPLOAD_DIR / f"{session_id}_template.pptx").exists():
        tpath = str(UPLOAD_DIR / f"{session_id}_template.pptx")

    out = OUTPUT_DIR / f"{session_id}_{course_title}_slides.pptx"
    gen = PPTGenerator(theme=theme, template_path=tpath)
    try:
        _sync(gen.generate_to_file, str(out),
              transcript=transcript, course_title=course_title,
              key_points=kp or None, image_paths=imgs or None,
              use_sd=use_sd, use_blip2=use_blip2)
        await _push(session_id, 100, "Presentation generated", "ppt")
        return {"success": True, "pptx_filename": out.name,
                "used_template": tpath is not None,
                "download_url": f"/api/download/{out.name}"}
    except Exception as e: raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ⑦ SCRIPT GENERATION
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/script")
async def step_script(
    session_id:  str = Form(...),
    slides_json: str = Form(...),
    openai_key:  str = Form(""),
):
    from modules.pipeline import ScriptGenerator
    slides = json.loads(slides_json)
    gen = ScriptGenerator(openai_key=openai_key or None)
    try:
        sw = _sync(gen.generate_from_slides, slides)
        sc = gen.merge(sw)
        sp = OUTPUT_DIR / f"{session_id}_script.txt"
        sp.write_text(sc, encoding="utf-8")
        await _push(session_id, 100, "Script generated", "script")
        return {"success": True, "script": sc,
                "script_filename": sp.name,
                "download_url": f"/api/download/{sp.name}",
                "slides": sw}
    except Exception as e: raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ⑧ VIDEO GENERATION (demo OR real)
# ══════════════════════════════════════════════════════════════
@app.post("/api/step/video")
async def step_video(
    session_id: str = Form(...),
    script:     str = Form(""),
    video_mode: str = Form("demo"),   # "demo" | "real"
    ref_audio:  str = Form(""),       # path to processed audio (auto-filled)
    face_video: str = Form(""),       # path to teacher face video (auto-filled from step 2)
):
    """
    demo mode: simulate F5-TTS→wav2lip→FFmpeg, serve pre-placed demo video
    real mode: actually run F5-TTS→wav2lip→FFmpeg
    """
    # Auto-resolve face video from teacher video upload
    if not face_video:
        auto_face = UPLOAD_DIR / f"{session_id}_face.mp4"
        if auto_face.exists():
            face_video = str(auto_face)

    # Auto-resolve reference audio from processed audio
    if not ref_audio:
        auto_audio = UPLOAD_DIR / f"{session_id}_processed.wav"
        if auto_audio.exists():
            ref_audio = str(auto_audio)

    demo_steps = [
        (10,  "[F5-TTS] Initializing Diffusion Transformer…",      "voice"),
        (22,  "[F5-TTS] Loading teacher speaker embedding…",        "voice"),
        (34,  "[F5-TTS] ODE-Solver: Mel-Spectrogram generation…",  "voice"),
        (44,  "[F5-TTS] Neural Vocoder → WAV output…",             "voice"),
        (55,  "[wav2lip] S3FD face detection…",                     "lipsync"),
        (66,  "[wav2lip] Generator: per-frame mouth synthesis…",    "lipsync"),
        (76,  "[FFmpeg] Frame alignment: Audio_t · Lip_t · Pose_t…","video"),
        (86,  "[FFmpeg] Compositing slides + PiP lecturer…",        "video"),
        (94,  "[FFmpeg] Embedding subtitles + chapter markers…",    "video"),
        (100, "🎬 MOOCs video generation complete!",               "video"),
    ]

    if video_mode == "demo":
        for pct, msg, step in demo_steps:
            await _push(session_id, pct, msg, step)
            await asyncio.sleep(0.85)
        demo = OUTPUT_DIR / "完整moocs影片產出.mp4"
        if demo.exists():
            return {"success": True, "mode": "demo",
                    "filename": "完整moocs影片產出.mp4",
                    "download_url": "/api/download/完整moocs影片產出.mp4",
                    "stream_url":   "/api/stream/完整moocs影片產出.mp4"}
        return {"success": True, "mode": "demo_no_file",
                "message": "Place demo video at output/完整moocs影片產出.mp4"}

    # Real mode
    from modules.pipeline import VoiceVideoGenerator
    gen = VoiceVideoGenerator(output_dir=str(OUTPUT_DIR))
    progress_events: list = []
    def _pcb(p, m): progress_events.append((round(p * 100), m))
    fut = asyncio.get_event_loop().run_in_executor(
        None, lambda: gen.full_pipeline(
            script, ref_audio or None, face_video or None,
            mode="real", progress_cb=_pcb
        )
    )
    while not fut.done():
        while progress_events:
            pct, msg = progress_events.pop(0)
            await _push(session_id, pct, msg, "video")
        await asyncio.sleep(0.5)
    try:
        r = await fut
    except Exception as e:
        raise HTTPException(500, str(e))
    vid_name = r.get("filename", "moocs_output.mp4")
    return {"success": True, "mode": "real",
            "filename":     vid_name,
            "download_url": f"/api/download/{vid_name}",
            "stream_url":   f"/api/stream/{vid_name}"}


# ══════════════════════════════════════════════════════════════
# ONE-CLICK FULL PIPELINE
# ══════════════════════════════════════════════════════════════
@app.post("/api/pipeline/run")
async def run_pipeline(
    session_id:     str                        = Form(...),
    audio_file:     UploadFile                 = File(...),
    teacher_video:  Optional[UploadFile]       = File(None),
    syllabus:       str                        = Form(""),
    course_title:   str                        = Form("Lecture"),
    openai_key:     str                        = Form(""),
    clean_method:   str                        = Form("ollama"),
    theme:          str                        = Form("Modern Blue"),
    use_sd:         bool                       = Form(False),
    video_mode:     str                        = Form("demo"),
    images:         Optional[List[UploadFile]] = File(None),
    template_file:  Optional[UploadFile]       = File(None),
):
    # Save audio
    ext  = Path(audio_file.filename).suffix or ".wav"
    apath = UPLOAD_DIR / f"{session_id}_audio{ext}"
    _save(audio_file, apath)

    # Save teacher video
    face_path = None
    if teacher_video:
        vext  = Path(teacher_video.filename).suffix or ".mp4"
        vpath = UPLOAD_DIR / f"{session_id}_teacher{vext}"
        _save(teacher_video, vpath)
        face_path = str(vpath)

    # Save images
    img_paths: List[str] = []
    if images:
        img_dir = UPLOAD_DIR / session_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            dest = img_dir / img.filename
            _save(img, dest); img_paths.append(str(dest))

    # Save template
    tpath = None
    if template_file and template_file.filename.endswith(".pptx"):
        dest = UPLOAD_DIR / f"{session_id}_template.pptx"
        _save(template_file, dest); tpath = str(dest)

    from modules.pipeline import MOOCsPipeline
    pipeline = MOOCsPipeline({
        "output_dir":    str(OUTPUT_DIR),
        "openai_api_key": openai_key,
        "ppt_theme":     theme,
    })
    # Inject template_path into ppt config
    if tpath:
        pipeline.ppt.template_path = tpath
        pipeline.ppt.use_template  = True

    try:
        r = _sync(pipeline.run, str(apath),
                  syllabus_text=syllabus, course_title=course_title,
                  image_paths=img_paths or None,
                  face_video=face_path,
                  clean_method=clean_method, use_sd=use_sd,
                  video_mode=video_mode, timeout=3600)
        return JSONResponse({
            "success":     r["success"],
            "elapsed_sec": r["elapsed_sec"],
            "errors":      r["errors"],
            "output":      {k: v for k, v in r["output"].items()
                            if not isinstance(v, (bytes, bytearray))},
            "video_url":   f"/api/stream/{r['output'].get('video_filename', '完整moocs影片產出.mp4')}",
        })
    except Exception as e: raise HTTPException(500, str(e))


# ── Files ──────────────────────────────────────────────────────
@app.get("/api/download/{filename}")
async def download(filename: str):
    p = OUTPUT_DIR / filename
    if not p.exists(): raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(str(p), filename=filename,
                        media_type="application/octet-stream")


@app.get("/api/stream/{filename}")
async def stream(filename: str):
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, f"Video not found. Place at output/{filename}")
    return FileResponse(str(p), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
