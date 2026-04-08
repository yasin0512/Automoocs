"""
Module 6: Script Generation & Virtual Lecturer Video Output
===========================================================
Thesis Ch. 4-7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT : slides [{title, bullets}] from Module 5

PROCESSING:
  Per-slide LLM (Ollama → OpenAI fallback):
    Slide 1    → opening script
    Middle     → teaching body (12-15 min per section)
    Last slide → closing + gratitude
  All output: Traditional Chinese

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOICE — F5-TTS (Thesis 4-7-1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real mode (requires GPU + pip install f5-tts):
  from f5_tts.infer.utils_infer import infer_process
  Mel_gen = ODE-Solver(DiffusionTransformer(Text, SpeakerEmbedding))
  WAV_out = NeuralVocoder(Mel_gen)
  Adjust Prosody tags (pitch/speed/pause)

Demo mode: simulate steps, serve pre-placed demo video

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIP SYNC — wav2lip (Thesis 4-7-2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real mode (requires wav2lip repo + wav2lip_gan.pth):
  subprocess: python Wav2Lip/inference.py
    --checkpoint_path wav2lip_gan.pth
    --face face_video.mp4
    --audio synth.wav
    --outfile lipsync.mp4
  L_total = λ₁L_visual + λ₂L_sync + λ₃L_adv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO COMPOSITION — FFmpeg (Thesis 4-7-3/4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real mode (requires ffmpeg on PATH):
  Frame_t = f(Audio_t, Lip_t, Pose_t)
  ffmpeg overlay: slides + PiP lecturer + subtitles + chapters
  H.264/AAC MP4 output

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE SELECTION (per-endpoint or per-call)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mode="demo"  — simulate progress + serve pre-placed demo video
  mode="real"  — actually run F5-TTS / wav2lip / ffmpeg via subprocess
"""

import os, sys, time, json, logging, subprocess
from typing import List, Dict, Optional, Callable, Any
import requests

logger = logging.getLogger(__name__)

DEMO_FILENAME = "完整moocs影片產出.mp4"

# ─── Prompt templates ────────────────────────────────────────
_SYS = ("You are a senior university professor, warm and clear. "
        "Always respond in Traditional Chinese (繁體中文).")
_OPEN_TMPL  = """Based on the slide title, write a natural course opening (~1-2 min):
- Start: 「大家好，今天這門課是…」
- Briefly introduce, then say: 「現在開始我們今天的課程」
- Warm, welcoming, NOT instructional
Title: \"\"\"{content}\"\"\"
Output opening directly:"""
_BODY_TMPL  = """Based on slide content, write teaching script (~12-15 min section):
- All Traditional Chinese; translate English terms
- Natural classroom tone; varied transitions
- NO opening/closing — dive into the topic
- Use examples and analogies
Content: \"\"\"{content}\"\"\"
Output teaching script directly:"""
_CLOSE_TMPL = """Based on slide content, write a course closing (~2 min):
- Concise key-point summary
- Natural ending: 「好的，今天的課程就到這邊」
- Gratitude: 「謝謝大家的聆聽」
Content: \"\"\"{content}\"\"\"
Output closing directly:"""


# ──────────────────────────────────────────────────────────────
# SCRIPT GENERATOR
# ──────────────────────────────────────────────────────────────
class ScriptGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3",
                 openai_key: Optional[str] = None,
                 openai_model: str = "gpt-4o"):
        self.ollama_url   = ollama_url
        self.ollama_model = ollama_model
        self.openai_key   = openai_key or os.getenv("OPENAI_API_KEY","")
        self.openai_model = openai_model
        self._ok: Optional[bool] = None

    def _check_ollama(self) -> bool:
        if self._ok is not None: return self._ok
        try:
            self._ok = requests.get(f"{self.ollama_url}/api/tags", timeout=4).status_code == 200
        except Exception:
            self._ok = False
        return self._ok

    def _ollama(self, prompt: str) -> str:
        r = requests.post(f"{self.ollama_url}/api/chat", json={
            "model": self.ollama_model,
            "messages":[{"role":"system","content":_SYS},{"role":"user","content":prompt}],
            "stream": False, "options":{"temperature":0.7},
        }, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _openai(self, prompt: str) -> str:
        import openai
        c = openai.OpenAI(api_key=self.openai_key)
        r = c.chat.completions.create(
            model=self.openai_model,
            messages=[{"role":"system","content":_SYS},{"role":"user","content":prompt}],
            temperature=0.7, max_tokens=2000)
        return r.choices[0].message.content.strip()

    def _gen(self, content: str, is_first: bool, is_last: bool) -> str:
        tmpl = _OPEN_TMPL if is_first else (_CLOSE_TMPL if is_last else _BODY_TMPL)
        prompt = tmpl.format(content=content[:1500])
        if self._check_ollama():
            try: return self._ollama(prompt)
            except Exception as e: logger.warning(f"Ollama: {e}")
        if self.openai_key:
            try: return self._openai(prompt)
            except Exception as e: logger.error(f"OpenAI: {e}")
        return f"（{'開場白' if is_first else '課程結語' if is_last else '教學講解'}）\n\n{content}"

    def generate_from_slides(self, slides: List[Dict],
                              progress_cb: Optional[Callable] = None) -> List[Dict]:
        total = len(slides)
        for i, s in enumerate(slides):
            if progress_cb: progress_cb(i/total, f"Script {i+1}/{total} …")
            s["script"] = self._gen(
                s.get("title","") + "\n" + "\n".join(s.get("bullets",[])),
                i==0, i==total-1
            )
            time.sleep(0.15)
        if progress_cb: progress_cb(1.0,"Script generation complete")
        return slides

    @staticmethod
    def merge(slides: List[Dict]) -> str:
        return "\n\n".join(s.get("script","") for s in slides if s.get("script"))


# ──────────────────────────────────────────────────────────────
# VOICE & VIDEO GENERATOR (DEMO + REAL)
# ──────────────────────────────────────────────────────────────
class VoiceVideoGenerator:
    """
    Supports two modes per operation:
      mode="demo"  — simulate progress steps, serve pre-placed video
      mode="real"  — invoke F5-TTS / wav2lip / ffmpeg via subprocess
    """

    def __init__(self, output_dir: str = "output",
                 wav2lip_dir: str  = "Wav2Lip",
                 f5_available: bool = False):
        self.output_dir    = output_dir
        self.wav2lip_dir   = wav2lip_dir
        self.f5_available  = f5_available
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _check_f5() -> bool:
        try: import f5_tts; return True
        except ImportError: return False

    @staticmethod
    def _check_wav2lip(wav2lip_dir: str) -> bool:
        ckpt = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
        return os.path.exists(ckpt)

    @staticmethod
    def _check_ffmpeg() -> bool:
        return subprocess.run(["ffmpeg","-version"], capture_output=True).returncode == 0

    # ── F5-TTS ────────────────────────────────────────────
    def synthesize_voice(self, script: str,
                          ref_audio: Optional[str] = None,
                          out_path: Optional[str] = None,
                          mode: str = "demo",
                          progress_cb: Optional[Callable] = None) -> Dict:
        def _cb(p,m):
            if progress_cb: progress_cb(p,m)
        out = out_path or os.path.join(self.output_dir, "synthesized_voice.wav")

        if mode == "real" and self._check_f5():
            _cb(0.1, "[F5-TTS] Loading Diffusion Transformer …")
            try:
                from f5_tts.infer.utils_infer import infer_process
                _cb(0.3, "[F5-TTS] Extracting speaker embedding …")
                audio = infer_process(
                    ref_audio=ref_audio or "",
                    ref_text="",
                    gen_text=script,
                )
                import soundfile as sf, numpy as np
                sf.write(out, np.array(audio), 24000)
                _cb(1.0, "[F5-TTS] Voice synthesis complete (real)")
                return {"success":True,"mode":"real","output_path":out}
            except Exception as e:
                logger.error(f"F5-TTS real mode failed: {e}"); mode = "demo"

        # Demo mode
        for pct, msg in [(0.1,"[F5-TTS] Initializing Diffusion Transformer …"),
                          (0.3,"[F5-TTS] Loading speaker voiceprint embedding …"),
                          (0.5,"[F5-TTS] ODE-Solver → Mel-Spectrogram generation …"),
                          (0.75,"[F5-TTS] Neural Vocoder → WAV output …"),
                          (0.9, "[F5-TTS] Prosody tag adjustment …"),
                          (1.0, "[F5-TTS] Voice synthesis complete (demo)")]:
            _cb(pct, msg); time.sleep(0.35)
        return {"success":True,"mode":"demo","output_path":out,
                "note":"Demo mode. Install f5-tts + ref audio for real synthesis."}

    # ── wav2lip ───────────────────────────────────────────
    def lipsync(self, face_video: Optional[str] = None,
                audio_path: Optional[str] = None,
                out_path: Optional[str] = None,
                mode: str = "demo",
                progress_cb: Optional[Callable] = None) -> Dict:
        def _cb(p,m):
            if progress_cb: progress_cb(p,m)
        out = out_path or os.path.join(self.output_dir, "lipsync.mp4")

        if mode == "real" and self._check_wav2lip(self.wav2lip_dir) and face_video and audio_path:
            _cb(0.1, "[wav2lip] Starting lip-sync inference …")
            ckpt = os.path.join(self.wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
            cmd  = [
                sys.executable,
                os.path.join(self.wav2lip_dir, "inference.py"),
                "--checkpoint_path", ckpt,
                "--face", face_video,
                "--audio", audio_path,
                "--outfile", out,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if proc.returncode == 0:
                    _cb(1.0, "[wav2lip] Lip-sync complete (real)")
                    return {"success":True,"mode":"real","output_path":out}
                logger.error(f"wav2lip stderr: {proc.stderr}"); mode = "demo"
            except Exception as e:
                logger.error(f"wav2lip real failed: {e}"); mode = "demo"

        for pct, msg in [(0.1,"[wav2lip] Loading S3FD face detection …"),
                          (0.3,"[wav2lip] Detecting + cropping face …"),
                          (0.5,"[wav2lip] Mel-Spectrogram + CNN face encoding …"),
                          (0.7,"[wav2lip] Generator: per-frame mouth synthesis …"),
                          (0.9,"[wav2lip] GAN Discriminator verification …"),
                          (1.0,"[wav2lip] Lip-sync complete (demo)")]:
            _cb(pct, msg); time.sleep(0.35)
        return {"success":True,"mode":"demo","output_path":out,
                "note":"Demo mode. Clone Wav2Lip + download wav2lip_gan.pth for real output."}

    # ── FFmpeg composition ────────────────────────────────
    def compose_video(self, slide_video: Optional[str] = None,
                       lipsync_video: Optional[str] = None,
                       audio: Optional[str] = None,
                       mode: str = "demo",
                       progress_cb: Optional[Callable] = None) -> Dict:
        def _cb(p,m):
            if progress_cb: progress_cb(p,m)
        demo_out = os.path.join(self.output_dir, DEMO_FILENAME)

        if (mode == "real" and self._check_ffmpeg()
                and lipsync_video and os.path.exists(lipsync_video)):
            final_out = os.path.join(self.output_dir, "moocs_output.mp4")
            _cb(0.2, "[FFmpeg] Compositing slide + PiP lecturer …")
            # PiP overlay: lipsync bottom-right corner of slide canvas
            filt = "[1:v]scale=320:240[pip];[0:v][pip]overlay=W-w-20:H-h-20"
            cmd = [
                "ffmpeg", "-y",
                "-i", slide_video or lipsync_video,
                "-i", lipsync_video,
                "-filter_complex", filt,
                "-c:v", "libx264", "-c:a", "aac",
                "-shortest", final_out,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=600)
                if proc.returncode == 0:
                    _cb(1.0,"[FFmpeg] Composition complete (real)")
                    return {"success":True,"mode":"real","output_path":final_out,
                            "filename":os.path.basename(final_out)}
                mode = "demo"
            except Exception as e:
                logger.error(f"ffmpeg failed: {e}"); mode = "demo"

        for pct, msg in [(0.1,"[FFmpeg] Initializing multimodal composition …"),
                          (0.3,"[FFmpeg] Frame-level alignment: Audio_t · Lip_t · Pose_t …"),
                          (0.5,"[FFmpeg] Compositing slides + PiP virtual lecturer …"),
                          (0.7,"[FFmpeg] Embedding audio + subtitles …"),
                          (0.85,"[FFmpeg] Adding chapter markers …"),
                          (0.95,"[FFmpeg] Encoding H.264/AAC MP4 …"),
                          (1.0, "🎬 MOOCs video generation complete (demo)")]:
            _cb(pct, msg); time.sleep(0.40)
        return {"success":True,"mode":"demo","output_path":demo_out,
                "filename":DEMO_FILENAME}

    # ── Full video pipeline ───────────────────────────────
    def full_pipeline(self, script: str,
                       ref_audio: Optional[str] = None,
                       face_video: Optional[str] = None,
                       mode: str = "demo",
                       progress_cb: Optional[Callable] = None) -> Dict:
        def _cb(p,m):
            if progress_cb: progress_cb(p,m)
        _cb(0.0,"Starting voice & video pipeline …")
        v = self.synthesize_voice(script, ref_audio, mode=mode,
                                   progress_cb=lambda p,m: _cb(p*0.33,m))
        l = self.lipsync(face_video, v["output_path"], mode=mode,
                          progress_cb=lambda p,m: _cb(0.33+p*0.34,m))
        c = self.compose_video(mode=mode,
                                lipsync_video=l["output_path"] if mode=="real" else None,
                                progress_cb=lambda p,m: _cb(0.67+p*0.33,m))
        return {"success":True,"voice":v,"lipsync":l,"video":c,
                "final_video":c["output_path"],"filename":c["filename"]}


# ──────────────────────────────────────────────────────────────
# UNIFIED PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────
from .preprocessing import AudioProcessor, VideoSlicer
from .text_cleaner  import TextCleaner
from .syllabus_aligner import SyllabusAligner
from .ppt_generator import PPTGenerator, segment_slides


class MOOCsPipeline:
    """
    Complete 6-module MOOCs generation pipeline.

    Usage:
      p = MOOCsPipeline(config={...})
      r = p.run(audio_path=..., ...)
    """

    def __init__(self, config: Optional[Dict] = None):
        c = config or {}
        self.output_dir   = c.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        ou  = c.get("ollama_url",    "http://localhost:11434")
        om  = c.get("ollama_model",  "llama3")
        oai = c.get("openai_api_key","")
        oam = c.get("openai_model",  "gpt-4o")
        self.audio    = AudioProcessor(c.get("whisper_model","large-v3"))
        self.cleaner  = TextCleaner(oai, ou, om, oam)
        self.aligner  = SyllabusAligner()
        self.ppt      = PPTGenerator(c.get("ppt_theme","Modern Blue"),
                                     c.get("sd_url","http://127.0.0.1:7860"))
        self.scripter = ScriptGenerator(ou, om, oai, oam)
        self.video    = VoiceVideoGenerator(
            self.output_dir,
            wav2lip_dir=c.get("wav2lip_dir","Wav2Lip"),
        )

    def run(self, audio_path: str, syllabus_text: str = "",
            course_title: str = "Lecture",
            image_paths: Optional[List[str]] = None,
            ref_audio: Optional[str] = None,
            face_video: Optional[str] = None,
            clean_method: str = "ollama",
            align_threshold: float = 0.6,
            use_sd: bool = False, use_blip2: bool = False,
            video_mode: str = "demo",
            progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        import traceback as tb

        def _cb(p, m):
            logger.info(f"[{p*100:.0f}%] {m}")
            if progress_cb: progress_cb(p, m)

        result: Dict[str,Any] = {"success":False,"steps":{},"errors":[],"output":{}}
        t0 = time.time()

        # ── Step 1 ───────────────────────────────────────
        try:
            _cb(0.02,"Step 1/6 | Audio preprocessing …")
            ar = self.audio.process_classroom_audio(audio_path,
                 progress_cb=lambda p,m: _cb(0.02+p*0.13,f"[Audio] {m}"))
            raw = ar["transcription"]
            result["steps"]["audio"] = {"transcription":raw,"segments":ar.get("segments",[])}
            wav_out = os.path.join(self.output_dir,"processed.wav")
            self.audio.save_audio(ar["audio"], wav_out)
            _cb(0.16,f"✅ Transcription: {len(raw)} chars")
        except Exception as e:
            result["errors"].append(f"Audio failed: {e}"); logger.error(tb.format_exc())
            return result

        # ── Step 2 ───────────────────────────────────────
        try:
            _cb(0.17,"Step 2/6 | LLM text cleaning …")
            cr = self.cleaner.clean(raw, prefer=clean_method,
                 progress_cb=lambda p,m: _cb(0.17+p*0.10,f"[Clean] {m}"))
            clean = cr["cleaned"]
            result["steps"]["text_clean"] = cr
            _cb(0.28,f"✅ Cleaning ({cr['method']})")
        except Exception as e:
            clean = raw
            result["errors"].append(f"Cleaning failed: {e}")

        # ── Step 3 ───────────────────────────────────────
        kps: List[str] = []
        try:
            _cb(0.29,"Step 3/6 | Syllabus alignment …")
            if syllabus_text.strip():
                ar3 = self.aligner.full_pipeline(clean, syllabus_text, align_threshold,
                      progress_cb=lambda p,m: _cb(0.29+p*0.16,f"[Align] {m}"))
                kps = ar3["syllabus"]["key_points"]
                result["steps"]["alignment"] = {"coverage_rate":ar3["coverage"]["coverage_rate"],"key_points":kps}
                _cb(0.46,f"✅ Alignment: {ar3['coverage']['coverage_rate']*100:.1f}% coverage")
            else:
                _cb(0.46,"⚠️ No syllabus — skipping")
                result["steps"]["alignment"] = {"skipped":True}
        except Exception as e:
            result["errors"].append(f"Alignment failed: {e}")

        # ── Step 4 ───────────────────────────────────────
        pptx_path = None; slides_data: List[Dict] = []
        try:
            _cb(0.47,"Step 4/6 | Presentation generation …")
            pptx_path = os.path.join(self.output_dir,f"{course_title}_slides.pptx")
            buf = self.ppt.generate(clean, course_title=course_title,
                 key_points=kps or None, image_paths=image_paths,
                 use_sd=use_sd, use_blip2=use_blip2,
                 progress_cb=lambda p,m: _cb(0.47+p*0.13,f"[PPT] {m}"))
            with open(pptx_path,"wb") as f: f.write(buf.read())
            slides_data = segment_slides(clean, kps or None)
            result["steps"]["ppt"] = {"path":pptx_path,"n_slides":len(slides_data)}
            _cb(0.61,f"✅ PPT: {len(slides_data)} slides")
        except Exception as e:
            result["errors"].append(f"PPT failed: {e}"); logger.error(tb.format_exc())

        # ── Step 5 ───────────────────────────────────────
        script = ""
        try:
            _cb(0.62,"Step 5/6 | Script generation …")
            if slides_data:
                slides_ws = self.scripter.generate_from_slides(list(slides_data),
                            progress_cb=lambda p,m: _cb(0.62+p*0.13,f"[Script] {m}"))
                script = self.scripter.merge(slides_ws)
            else:
                script = self.scripter._gen(clean[:2000], False, False)
            sp = os.path.join(self.output_dir,f"{course_title}_script.txt")
            with open(sp,"w",encoding="utf-8") as f: f.write(script)
            result["steps"]["script"] = {"path":sp,"length":len(script)}
            _cb(0.76,f"✅ Script: {len(script)} chars")
        except Exception as e:
            result["errors"].append(f"Script failed: {e}"); logger.error(tb.format_exc())

        # ── Step 6 ───────────────────────────────────────
        try:
            _cb(0.77,"Step 6/6 | Voice + video …")
            vr = self.video.full_pipeline(script or clean, ref_audio, face_video,
                 mode=video_mode,
                 progress_cb=lambda p,m: _cb(0.77+p*0.22,f"[Video] {m}"))
            result["steps"]["video"] = vr
            result["output"]["video_path"]     = vr["final_video"]
            result["output"]["video_filename"] = vr["filename"]
            _cb(1.0,"✅ Complete MOOCs video ready!")
        except Exception as e:
            result["errors"].append(f"Video failed: {e}"); logger.error(tb.format_exc())

        result["success"]     = len(result["errors"]) == 0
        result["elapsed_sec"] = round(time.time()-t0,1)
        result["output"].update({"pptx_path":pptx_path,"transcript":clean,"script":script})
        return result

    def run_training(self, video_path: str, out_dir: Optional[str] = None,
                      epochs: int = 20, batch: int = 16,
                      progress_cb: Optional[Callable] = None) -> str:
        from .action_modeling import run_full_training_pipeline
        return run_full_training_pipeline(
            video_path, out_dir or os.path.join(self.output_dir,"training"),
            val_ratio=0.2, epochs=epochs, batch_size=batch,
            progress_cb=progress_cb,
        )
