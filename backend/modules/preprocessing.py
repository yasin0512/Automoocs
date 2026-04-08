"""
Module 1 + Module (4-3): Data Collection, Preprocessing & Speaker Analysis
============================================================================
Thesis Sections:
  Ch. 4-2    Resource Collection & Preprocessing
  Ch. 4-3-1  Speech-to-Text (Whisper ASR)
  Ch. 4-3-2  Speaker Diarization & Speaker Embedding (ECAPA-TDNN)
  Ch. 4-3-3  Time Alignment & Multimodal Annotation Integration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4-2-1  AUDIO PREPROCESSING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT : WAV / MP3 / M4A / FLAC lecture recording

PROCESSING:
  1. librosa.load             — any sample rate, mono
  2. noisereduce              — non-stationary noise suppression, prop_decrease=0.8
  3. librosa.resample         — → 16 kHz
  4. Silero-VAD               — threshold=0.5, min_speech_ms=100, min_silence_ms=100
  5. RMS normalization        — target −20 dBFS, peak clip @ 0.95
  6. Whisper large-v3 / base  — Chinese ASR with timestamped segments
  7. OpenCC s2tw              — Simplified → Traditional Chinese

OUTPUT:
  {audio: np.ndarray, sample_rate: int,
   transcription: str, segments: [{start, end, text}]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4-2-2  VIDEO PREPROCESSING PIPELINE (VideoSlicer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT : Historical teacher lecture video (MP4/AVI/MOV)

PROCESSING per 5-second clip:
  Frame enhancement: CLAHE → brightness (α=1.2, β=10) → Gaussian blur (3×3)
  Resize: letterbox → 224×224 JPEG
  Audio: extract WAV → transcribe per clip

OUTPUT:
  clips/{name}/frame_NNNN.jpg   — enhanced frames
  wav/{name}.wav                — 16 kHz mono
  transcripts/{name}.json       — {text, segments, start, end}
  clip_metadata.json            — full manifest

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4-3-2  SPEAKER DIARIZATION (ECAPA-TDNN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROCESSING:
  ECAPA-TDNN speaker embedding model:
    → fixed-dim speaker embedding vector (voiceprint) per segment
    → loss: weighted AM-Softmax:
      L_spk = −(1/N) Σ log[ e^{s(cos(θ_{y_i,i})−m)} /
                           (e^{s(cos(θ_{y_i,i})−m)} + Σ_{j≠y_i} e^{s·cos(θ_{j,i})}) ]
  Clustering: cosine similarity + AHC (Agglomerative Hierarchical Clustering)
    → unique speaker ID per cluster

OUTPUT per segment:
  {start, end, speaker_id, embedding, text, confidence}
  Builds cumulative teacher voice database for F5-TTS clone

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4-3-3  TIME ALIGNMENT & MULTIMODAL ANNOTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT unified annotation record per segment:
  {start, end, speaker_id, text, confidence,
   pose_features (if available), segment_id}
  → JSON database for downstream alignment, PPT, video synthesis

CPU DEPLOYMENT NOTES:
  - Whisper: auto-downgrades large-v3 → base on CPU (no GPU detected)
  - Silero-VAD: runs on CPU via torch (no CUDA required)
  - ECAPA-TDNN: speechbrain runs on CPU (slow but functional)
  - All torch operations: device auto-selected (cuda if available, else cpu)
  - BLIP-2: skipped on CPU if not installed (CLIP still works CPU-side)
"""

import os
import gc
import cv2
import json
import time
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# IMAGE ENHANCEMENT (Thesis 4-2-2)
# ──────────────────────────────────────────────────────────────
def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE on LAB L-channel — contrast-limited adaptive histogram equalization."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def adjust_brightness(frame: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """Linear contrast + brightness: output = α·input + β"""
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Full frame pipeline: CLAHE → brightness → Gaussian blur (3×3)"""
    return cv2.GaussianBlur(adjust_brightness(apply_clahe(frame)), (3, 3), 0)


def letterbox(frame: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Aspect-ratio-preserving resize with black border padding → 224×224"""
    h, w = frame.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ──────────────────────────────────────────────────────────────
# AUDIO PROCESSOR (Thesis 4-2-1 + 4-3-1)
# ──────────────────────────────────────────────────────────────
class AudioProcessor:
    """
    Full audio preprocessing + Whisper ASR.
    CPU-compatible: auto-selects Whisper 'base' on CPU, 'large-v3' on GPU.
    Silero-VAD runs on CPU natively via PyTorch.
    """

    def __init__(self, whisper_model_size: str = "large-v3"):
        self._wsize   = whisper_model_size
        self._whisper = None
        self._vad_model = self._vad_utils = None
        self._cc      = None
        self._device  = "cpu"
        self._ready   = False

    def _load(self):
        if self._ready:
            return
        import torch
        import whisper

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU auto-downgrade: large-v3 is 1.5 GB and very slow on CPU
        size = self._wsize if self._device == "cuda" else "base"
        logger.info(f"AudioProcessor: device={self._device}, Whisper={size}")
        self._whisper = whisper.load_model(size, device=self._device)

        # Silero-VAD — runs on CPU, downloads ~few MB automatically
        logger.info("Loading Silero-VAD (CPU-compatible)…")
        self._vad_model, self._vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            force_reload=False, onnx=False
        )
        self._vad_model = self._vad_model.to(self._device)

        # OpenCC simplified → traditional Chinese
        try:
            import opencc
            self._cc = opencc.OpenCC("s2tw")
        except ImportError:
            logger.warning("opencc not installed — skipping traditional Chinese conversion")
        self._ready = True

    # ── Extract WAV from video ────────────────────────────────
    def extract_wav(self, video_path: str, out_wav: str) -> bool:
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    return False
                clip.audio.write_audiofile(out_wav, fps=16000, nbytes=2,
                                           codec="pcm_s16le", logger=None)
            return True
        except Exception as e:
            logger.warning(f"moviepy failed ({e}), trying ffmpeg…")
            return os.system(
                f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 "{out_wav}" -loglevel quiet'
            ) == 0

    # ── Noise reduction ───────────────────────────────────────
    @staticmethod
    def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
        """noisereduce non-stationary suppression, prop_decrease=0.8"""
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)

    # ── Silero-VAD ────────────────────────────────────────────
    def vad_remove_silence(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Voice Activity Detection — removes silence, keeps speech segments."""
        self._load()
        import torch, librosa
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        tensor = torch.FloatTensor(audio).to(self._device)
        (get_ts, *_) = self._vad_utils
        segs = get_ts(tensor, self._vad_model, sampling_rate=16000,
                      threshold=0.5, min_speech_duration_ms=100,
                      min_silence_duration_ms=100)
        if not segs:
            return audio
        return np.concatenate([audio[s["start"]:s["end"]] for s in segs])

    # ── Volume normalization ──────────────────────────────────
    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """RMS normalization to −20 dBFS; peak clip at 0.95."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        audio = audio * (10 ** (target_db / 20) / rms)
        peak = np.abs(audio).max()
        if peak > 0.95:
            audio *= 0.95 / peak
        return audio

    # ── Whisper transcription ─────────────────────────────────
    def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe 16 kHz float32 array with Whisper + OpenCC s2tw."""
        self._load()
        import soundfile as sf
        ts  = str(int(time.time() * 1e6))
        tmp = os.path.join(tempfile.gettempdir(), f"w_{ts}.wav")
        try:
            sf.write(tmp, audio.astype(np.float32), 16000)
            r = self._whisper.transcribe(tmp, language="zh",
                                          task="transcribe", verbose=False)
            text = r["text"].strip()
            segs = []
            for seg in r.get("segments", []):
                t = seg["text"].strip()
                if self._cc:
                    t = self._cc.convert(t)
                segs.append({"start": seg["start"], "end": seg["end"],
                              "text": t, "confidence": seg.get("avg_logprob", 0.0)})
            if self._cc:
                text = self._cc.convert(text)
            return {"text": text, "segments": segs}
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            gc.collect()

    # ── Full classroom audio pipeline (4-2-1 + 4-3-1) ────────
    def process_classroom_audio(self, audio_path: str,
                                 progress_cb=None) -> Dict[str, Any]:
        """
        Full audio pipeline for current lecture recording.

        Steps: Load → Denoise → Resample 16kHz → VAD → Normalize → Whisper

        Returns:
          {audio, sample_rate, transcription, segments}
        """
        def _cb(p, m):
            if progress_cb:
                progress_cb(p, m)

        import librosa

        _cb(0.05, "Loading audio file…")
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        logger.info(f"Loaded {len(audio)/sr:.1f}s @ {sr} Hz")

        _cb(0.15, "Noise reduction (noisereduce non-stationary)…")
        audio = self.denoise(audio, sr)

        _cb(0.30, "Resampling to 16 kHz…")
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        _cb(0.45, "Silero-VAD: removing silence segments…")
        audio = self.vad_remove_silence(audio, sr)

        _cb(0.60, "Volume normalization → −20 dBFS…")
        audio = self.normalize(audio)

        _cb(0.72, "Whisper ASR + OpenCC (Traditional Chinese)…")
        result = self.transcribe(audio)

        _cb(1.0, "Audio preprocessing complete")
        return {
            "audio":         audio,
            "sample_rate":   sr,
            "transcription": result["text"],
            "segments":      result["segments"],
        }

    def save_audio(self, audio: np.ndarray, path: str):
        import soundfile as sf
        sf.write(path, audio.astype(np.float32), 16000)


# ──────────────────────────────────────────────────────────────
# SPEAKER DIARIZATION (Thesis 4-3-2) — ECAPA-TDNN
# ──────────────────────────────────────────────────────────────
class SpeakerDiarizer:
    """
    Speaker diarization using ECAPA-TDNN (Thesis 4-3-2).

    Technical flow:
      1. ECAPA-TDNN → speaker embedding vector per segment
         Loss: Weighted AM-Softmax
           L_spk = -(1/N) Σ log [ e^{s(cos(θ_{y_i,i})-m)} /
                                  (e^{s(cos(θ_{y_i,i})-m)} + Σ_{j≠y_i} e^{s·cos(θ_{j,i})}) ]
      2. Cosine similarity + AHC clustering → unique speaker IDs
      3. Build teacher voice database for downstream F5-TTS cloning

    CPU deployment:
      speechbrain runs on CPU — slower but fully functional.
      Install: pip install speechbrain

    Fallback:
      If speechbrain not installed, returns simple single-speaker annotation.
    """

    def __init__(self):
        self._model  = None
        self._device = "cpu"

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from speechbrain.pretrained import EncoderClassifier
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading ECAPA-TDNN (SpeechBrain) on {self._device}…")
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self._device}
            )
        except Exception as e:
            logger.warning(f"ECAPA-TDNN unavailable ({e}). "
                           "Install: pip install speechbrain. "
                           "Falling back to single-speaker mode.")

    def _embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Compute speaker embedding for audio segment."""
        if self._model is None:
            return None
        try:
            import torch
            wav = torch.FloatTensor(audio).unsqueeze(0).to(self._device)
            with torch.no_grad():
                emb = self._model.encode_batch(wav)
            return emb.squeeze().cpu().numpy()
        except Exception:
            return None

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _ahc_cluster(self, embeddings: List[np.ndarray],
                     threshold: float = 0.75) -> List[int]:
        """
        Agglomerative Hierarchical Clustering (AHC) on speaker embeddings.
        Returns speaker ID list.
        """
        n = len(embeddings)
        ids = list(range(n))
        # Greedy AHC: merge if cosine similarity >= threshold
        centroids = [e.copy() for e in embeddings]
        counts    = [1] * n
        for i in range(n):
            best_sim = -1.0
            best_j   = -1
            for j in range(i):
                sim = self._cosine_sim(centroids[i], centroids[j])
                if sim > best_sim:
                    best_sim, best_j = sim, j
            if best_j >= 0 and best_sim >= threshold:
                # Merge i into best_j
                root_j = ids[best_j]
                for k in range(n):
                    if ids[k] == ids[i]:
                        ids[k] = root_j
                # Update centroid (running average)
                c = counts[root_j]
                centroids[root_j] = (centroids[root_j] * c + centroids[i]) / (c + 1)
                counts[root_j] += 1
        # Remap to 0-based speaker IDs
        uid_map = {}
        out = []
        for sid in ids:
            if sid not in uid_map:
                uid_map[sid] = len(uid_map)
            out.append(uid_map[sid])
        return out

    def diarize(self, audio: np.ndarray, sr: int,
                whisper_segments: List[Dict],
                threshold: float = 0.75) -> List[Dict]:
        """
        Run speaker diarization on Whisper segments.

        Returns annotated segments:
          [{start, end, speaker_id, text, confidence, embedding}]

        Builds teacher voice database for F5-TTS speaker cloning.
        """
        self._load()

        if not whisper_segments:
            return []

        if self._model is None:
            # Fallback: single teacher speaker
            return [{**seg, "speaker_id": "TEACHER_00", "embedding": None}
                    for seg in whisper_segments]

        embeddings = []
        for seg in whisper_segments:
            s = int(seg["start"] * sr)
            e = int(seg["end"]   * sr)
            chunk = audio[s:e] if s < len(audio) else np.zeros(sr, np.float32)
            emb   = self._embed(chunk)
            embeddings.append(emb if emb is not None else np.zeros(192))

        speaker_ids = self._ahc_cluster(embeddings, threshold)
        n_speakers  = len(set(speaker_ids))
        logger.info(f"Diarization: {len(whisper_segments)} segments → {n_speakers} speaker(s)")

        result = []
        for i, (seg, sid, emb) in enumerate(
                zip(whisper_segments, speaker_ids, embeddings)):
            result.append({
                "start":      seg["start"],
                "end":        seg["end"],
                "speaker_id": f"SPEAKER_{sid:02d}",
                "text":       seg.get("text", ""),
                "confidence": seg.get("confidence", 0.0),
                "embedding":  emb.tolist() if emb is not None else None,
                "segment_id": i,
            })
        return result

    def build_teacher_voice_db(self, diarized_segments: List[Dict],
                                teacher_speaker_id: str = "SPEAKER_00",
                                out_path: str = "output/teacher_voice_db.json"):
        """
        Build cumulative teacher voiceprint database (Thesis 4-3-2).
        Used by F5-TTS for personalized voice cloning.
        """
        teacher_segs = [s for s in diarized_segments
                        if s["speaker_id"] == teacher_speaker_id]
        db = {
            "speaker_id": teacher_speaker_id,
            "n_segments":  len(teacher_segs),
            "embeddings":  [s["embedding"] for s in teacher_segs if s["embedding"]],
            "segments":    [{"start": s["start"], "end": s["end"], "text": s["text"]}
                            for s in teacher_segs],
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        logger.info(f"Teacher voice DB saved: {out_path} ({len(teacher_segs)} segments)")
        return db


# ──────────────────────────────────────────────────────────────
# TIME ALIGNMENT & MULTIMODAL ANNOTATION (Thesis 4-3-3)
# ──────────────────────────────────────────────────────────────
class TimeAligner:
    """
    4-3-3: Integrate ASR timestamps + speaker IDs + pose features into
    unified multimodal annotation records.

    Output JSON schema per segment:
      {segment_id, start, end, speaker_id, text, confidence,
       pose_features: {joint_variance, hand_speed, centroid_shift} | null}

    Supports downstream:
      - Syllabus knowledge-point alignment (Ch. 4-4)
      - Slide generation (Ch. 4-6)
      - Video synthesis (Ch. 4-7)
    """

    @staticmethod
    def align(diarized_segments: List[Dict],
              clip_metadata: Optional[List[Dict]] = None,
              out_path: Optional[str] = None) -> List[Dict]:
        """
        Merge diarized speech segments with pose features from clip metadata.

        Args:
            diarized_segments : output of SpeakerDiarizer.diarize()
            clip_metadata     : output of VideoSlicer.slice() (optional)
            out_path          : if set, saves JSON to this path

        Returns: unified annotation list
        """
        # Build clip lookup by time range
        clip_by_time: Dict[Tuple, Dict] = {}
        if clip_metadata:
            for clip in clip_metadata:
                clip_by_time[(clip["start"], clip["end"])] = clip

        records = []
        for seg in diarized_segments:
            t_mid = (seg["start"] + seg["end"]) / 2
            pose  = None
            for (cs, ce), clip in clip_by_time.items():
                if cs <= t_mid <= ce:
                    pose = {
                        "joint_variance":  clip.get("joint_variance"),
                        "hand_speed":      clip.get("hand_speed"),
                        "centroid_shift":  clip.get("centroid_shift"),
                    }
                    break

            records.append({
                "segment_id":    seg.get("segment_id", len(records)),
                "start":         seg["start"],
                "end":           seg["end"],
                "speaker_id":    seg.get("speaker_id", "SPEAKER_00"),
                "text":          seg.get("text", ""),
                "confidence":    seg.get("confidence", 0.0),
                "pose_features": pose,
            })

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"Multimodal annotation saved: {out_path}")

        return records

    @staticmethod
    def filter_teacher(records: List[Dict],
                       teacher_speaker_id: str = "SPEAKER_00") -> List[Dict]:
        """Return only teacher segments — filters out student questions etc."""
        return [r for r in records if r["speaker_id"] == teacher_speaker_id]


# ──────────────────────────────────────────────────────────────
# VIDEO SLICER (Thesis 4-2-2)
# ──────────────────────────────────────────────────────────────
class VideoSlicer:
    """
    Slice historical teacher lecture video into 5-second clips.

    Per clip:
      • Frame enhancement: CLAHE → brightness → Gaussian blur → letterbox 224×224
      • Audio extraction + Whisper transcription
      • Speaker diarization (optional)

    OUTPUT layout:
      clips/{name}/frame_NNNN.jpg
      wav/{name}.wav
      transcripts/{name}_transcript.json
      clip_metadata.json
    """

    def __init__(self, clip_duration: float = 5.0,
                 fps_sample: int = 10,
                 whisper_size: str = "base"):
        self.clip_duration = clip_duration
        self.fps_sample    = fps_sample
        self.audio_proc    = AudioProcessor(whisper_model_size=whisper_size)
        self.diarizer      = SpeakerDiarizer()

    def slice(self, video_path: str, output_dir: str,
              enhance: bool = True, transcribe: bool = True,
              run_diarization: bool = False,
              progress_cb=None) -> List[Dict]:
        """
        Slice video → clips with frames + wav + transcript.
        Returns: clip metadata list.
        """
        import soundfile as sf

        clips_dir = os.path.join(output_dir, "clips")
        wav_dir   = os.path.join(output_dir, "wav")
        trans_dir = os.path.join(output_dir, "transcripts")
        for d in [clips_dir, wav_dir, trans_dir]:
            os.makedirs(d, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_sec    = total_frames / fps_orig
        n_clips      = max(1, int(total_sec / self.clip_duration))
        base         = Path(video_path).stem

        # Extract full audio once
        full_wav = os.path.join(wav_dir, f"{base}_full.wav")
        self.audio_proc.extract_wav(video_path, full_wav)
        try:
            audio_full, sr_full = sf.read(full_wav)
            if audio_full.ndim > 1:
                audio_full = audio_full[:, 0]
            audio_full = audio_full.astype(np.float32)
        except Exception:
            audio_full = np.zeros(int(total_sec * 16000), np.float32)
            sr_full    = 16000

        sample_interval = max(1, int(fps_orig / self.fps_sample))
        metadata: List[Dict] = []

        for i in range(n_clips):
            t_start  = i * self.clip_duration
            t_end    = t_start + self.clip_duration
            name     = f"{base}_clip_{i:04d}"

            if progress_cb:
                progress_cb(i / n_clips, f"Slicing clip {i+1}/{n_clips}")

            # Frames
            fdir = os.path.join(clips_dir, name)
            os.makedirs(fdir, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps_orig))
            saved = 0
            for fnum in range(int(self.clip_duration * fps_orig)):
                ret, frame = cap.read()
                if not ret:
                    break
                if fnum % sample_interval == 0:
                    if enhance:
                        frame = enhance_frame(frame)
                    cv2.imwrite(
                        os.path.join(fdir, f"frame_{saved:04d}.jpg"),
                        letterbox(frame, (224, 224))
                    )
                    saved += 1

            # Audio segment
            s = int(t_start * 16000)
            e = int(t_end   * 16000)
            clip_audio = (audio_full[s:e] if s < len(audio_full)
                          else np.zeros(int(self.clip_duration * 16000), np.float32))
            clip_wav = os.path.join(wav_dir, f"{name}.wav")
            sf.write(clip_wav, clip_audio, 16000)

            # Transcript
            td: Dict[str, Any] = {
                "text": "", "segments": [],
                "clip_name": name, "start": t_start, "end": t_end
            }
            if transcribe and len(clip_audio) > 100:
                try:
                    res = self.audio_proc.transcribe(clip_audio)
                    td.update(res)
                    # Speaker diarization per clip (optional — slow on CPU)
                    if run_diarization and res["segments"]:
                        diarized = self.diarizer.diarize(
                            clip_audio, 16000, res["segments"]
                        )
                        td["diarized_segments"] = diarized
                except Exception as exc:
                    logger.warning(f"Transcription failed for {name}: {exc}")

            trans_path = os.path.join(trans_dir, f"{name}_transcript.json")
            with open(trans_path, "w", encoding="utf-8") as f:
                json.dump(td, f, ensure_ascii=False, indent=2)

            metadata.append({
                "clip_name":       name,
                "start":           t_start,
                "end":             t_end,
                "frames_dir":      fdir,
                "wav_path":        clip_wav,
                "transcript_path": trans_path,
                "transcript_text": td["text"],
                "frame_count":     saved,
            })

        cap.release()
        manifest = os.path.join(output_dir, "clip_metadata.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if progress_cb:
            progress_cb(1.0, f"Done — {len(metadata)} clips")
        return metadata
