"""
Module 2: Multimodal Action Analysis & Teacher Behavior Modeling
================================================================
Thesis Ch. 4-5

INPUTS (training):
  clips/{name}/frame_*.jpg   — enhanced 224×224 frames (from Module 1)
  wav/{name}.wav             — 16 kHz mono (from Module 1)
  transcripts/{name}.json    — Whisper text (from Module 1)

STEP 2a — PoseAnalyzer (MediaPipe):
  Per clip, process each JPEG frame through MediaPipe Pose.
  33 landmarks (x, y, z) per frame → (T, 33, 3) tensor.
  Compute:
    joint_variance  = mean(var(poses, axis=time))
    hand_speed      = mean(‖hand_joints[t] - hand_joints[t-1]‖)  — landmarks 15-20
    centroid_shift  = mean(‖trunk_centroid[t] - trunk_centroid[t-1]‖) — landmarks 11,12,23,24

STEP 2b — SpeechRateCalc:
  token_count  = total characters in transcript
  speech_rate  = token_count / total_duration  (chars/sec)

STEP 2c — ActionLabeler (Q1/Q3 percentile auto-annotation):
  Collect joint_variance and hand_speed across ALL clips.
  Q1 = 25th percentile, Q3 = 75th percentile.
  strong : joint_variance ≥ Q3  AND  hand_speed ≥ Q3
  weak   : joint_variance ≤ Q1  AND  hand_speed ≤ Q1  AND  token_count < 6
  medium : all others

STEP 2d — MultimodalActionModel (Thesis 4-5-2):
  Audio branch:
    • Compute 128-band Mel-Spectrogram from 16 kHz WAV
    • MelCNNEncoder: Conv1D stack → 768-dim embedding S^E
  Text branch:
    • BERT-base-chinese tokenization → [CLS] pooler output → 768-dim T^CLS
  Cross-modal fusion (Multi-Head Attention):
    • Q = T^CLS, K = V = S^E   →   attn_output  (768-dim)
    • combined = concat(attn_output, T^CLS)  →  1536-dim
  Classification head:
    • DNN: 1536→512→128→3  (BatchNorm + ReLU + Dropout 0.3)
    • Output: softmax over {weak=0, medium=1, strong=2}
  Loss: Weighted Cross-Entropy (Thesis eq.6)
    Loss = −Σ y_i · log(ŷ_i)
  Optimizer: Adam lr=2e-5, ReduceLROnPlateau, EarlyStopping(patience=7)

STEP 2e — ActionInference:
  Load best_action_model.pth → predict label + probabilities for new clips

OUTPUT:
  features/{name}_features.json  — joint_variance, hand_speed, centroid_shift
  labels/{name}_label.json       — label: weak/medium/strong
  models/best_action_model.pth
  models/training_history.json
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 2a. POSE ANALYZER
# ──────────────────────────────────────────────────────────────
class PoseAnalyzer:
    """
    Extract skeleton features per 5-second clip using MediaPipe Pose.
    Hand landmarks: 15,16,17,18,19,20
    Trunk landmarks: 11,12,23,24 (shoulders + hips)
    """

    HAND_IDX  = [15, 16, 17, 18, 19, 20]
    TRUNK_IDX = [11, 12, 23, 24]

    def __init__(self, min_det: float = 0.5, min_trk: float = 0.5):
        self.min_det = min_det
        self.min_trk = min_trk

    def extract_from_frames_dir(self, frames_dir: str) -> Dict[str, Any]:
        try:
            import mediapipe as mp
            import cv2
        except ImportError:
            logger.warning("mediapipe not installed; returning zeros")
            return self._zeros(frames_dir)

        mp_pose = mp.solutions.pose
        frames  = sorted(Path(frames_dir).glob("*.jpg"))
        if not frames:
            return self._zeros(frames_dir)

        all_lm = []
        with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_det,
            min_tracking_confidence=self.min_trk,
        ) as pose:
            import cv2
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lm = np.array(
                        [[p.x, p.y, p.z] for p in res.pose_landmarks.landmark],
                        dtype=np.float32
                    )
                    all_lm.append(lm)

        if not all_lm:
            return self._zeros(frames_dir)

        poses = np.stack(all_lm)                                 # (T, 33, 3)
        joint_variance  = float(np.mean(np.var(poses, axis=0)))

        hand  = poses[:, self.HAND_IDX, :]                       # (T, 6, 3)
        hand_speed = float(np.mean(np.linalg.norm(
            hand[1:] - hand[:-1], axis=-1
        ))) if len(hand) > 1 else 0.0

        trunk     = poses[:, self.TRUNK_IDX, :2]                 # (T, 4, 2)
        centroids = trunk.mean(axis=1)                           # (T, 2)
        centroid_shift = float(np.mean(np.linalg.norm(
            centroids[1:] - centroids[:-1], axis=-1
        ))) if len(centroids) > 1 else 0.0

        return {
            "frames_dir":     frames_dir,
            "n_frames":       len(all_lm),
            "joint_variance": round(joint_variance,  6),
            "hand_speed":     round(hand_speed,       6),
            "centroid_shift": round(centroid_shift,   6),
        }

    @staticmethod
    def _zeros(frames_dir: str) -> Dict:
        return {"frames_dir": frames_dir, "n_frames": 0,
                "joint_variance": 0.0, "hand_speed": 0.0, "centroid_shift": 0.0}


# ──────────────────────────────────────────────────────────────
# 2b. SPEECH RATE
# ──────────────────────────────────────────────────────────────
class SpeechRateCalc:
    @staticmethod
    def calc(transcript_path: str) -> Dict[str, Any]:
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"speech_rate": 0.0, "token_count": 0, "duration": 0.0}
        segs  = data.get("segments", [])
        chars = sum(len(s["text"]) for s in segs)
        dur   = sum(s["end"] - s["start"] for s in segs)
        return {
            "speech_rate": round(chars / dur, 2) if dur > 0 else 0.0,
            "token_count": chars,
            "duration":    round(dur, 2),
        }


# ──────────────────────────────────────────────────────────────
# 2c. ACTION LABELER
# ──────────────────────────────────────────────────────────────
class ActionLabeler:
    """Q1/Q3 percentile auto-annotation — Thesis 4-5-1."""

    def label(self, features: List[Dict]) -> List[Dict]:
        jv = np.array([f.get("joint_variance", 0.0) for f in features])
        hs = np.array([f.get("hand_speed",     0.0) for f in features])
        q1_jv, q3_jv = float(np.percentile(jv, 25)), float(np.percentile(jv, 75))
        q1_hs, q3_hs = float(np.percentile(hs, 25)), float(np.percentile(hs, 75))
        logger.info(f"joint_variance Q1={q1_jv:.5f} Q3={q3_jv:.5f}")
        logger.info(f"hand_speed     Q1={q1_hs:.5f} Q3={q3_hs:.5f}")

        labeled = []
        for f in features:
            f = dict(f)
            tc = f.get("token_count", 999)
            if   f["joint_variance"] >= q3_jv and f["hand_speed"] >= q3_hs:
                f["label"] = "strong"
            elif f["joint_variance"] <= q1_jv and f["hand_speed"] <= q1_hs and tc < 6:
                f["label"] = "weak"
            else:
                f["label"] = "medium"
            labeled.append(f)

        dist = {k: sum(1 for x in labeled if x["label"] == k)
                for k in ("strong", "medium", "weak")}
        logger.info(f"Label distribution: {dist}")
        return labeled


# ──────────────────────────────────────────────────────────────
# 2d. MULTIMODAL MODEL
# ──────────────────────────────────────────────────────────────
class MelCNNEncoder(nn.Module):
    """
    1D-CNN over 128-band Mel-Spectrogram.
    Input  : (B, 128, T)
    Output : (B, 768)
    """
    def __init__(self, out: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(512, out,  3, padding=1), nn.BatchNorm1d(out),  nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultimodalActionModel(nn.Module):
    """
    Thesis 4-5-2 architecture:
      Audio: MelCNNEncoder → 768-dim S^E
      Text : BERT-base-chinese [CLS] → 768-dim T^CLS
      Fusion: MHA(Q=T^CLS, K=V=S^E) → concat(attn, T^CLS) → DNN → 3-class
    """
    def __init__(self, num_classes: int = 3, freeze_bert: int = 8):
        super().__init__()
        self.audio_enc = MelCNNEncoder(768)
        self.text_enc  = BertModel.from_pretrained("bert-base-chinese")
        for layer in self.text_enc.encoder.layer[:freeze_bert]:
            for p in layer.parameters():
                p.requires_grad = False
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8,
                                                  batch_first=True, dropout=0.1)
        self.cls_head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),                             nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, mel, input_ids, attention_mask):
        audio_feat = self.audio_enc(mel)                          # (B, 768)
        text_feat  = self.text_enc(input_ids=input_ids,
                                    attention_mask=attention_mask).pooler_output  # (B, 768)
        attn, _ = self.cross_attn(
            query=text_feat.unsqueeze(1),
            key=audio_feat.unsqueeze(1),
            value=audio_feat.unsqueeze(1),
        )
        return self.cls_head(torch.cat([attn.squeeze(1), text_feat], dim=1))


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class ClipDataset(Dataset):
    LABEL_MAP = {"weak": 0, "medium": 1, "strong": 2}

    def __init__(self, meta: List[Dict], max_len: int = 64,
                 n_mels: int = 128, mframes: int = 128):
        self.meta    = meta
        self.maxlen  = max_len
        self.n_mels  = n_mels
        self.mframes = mframes
        self.tok     = BertTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        item  = self.meta[idx]
        label = self.LABEL_MAP.get(item.get("label", "medium"), 1)
        mel   = self._mel(item.get("wav_path", ""))
        text  = item.get("transcript_text", "")
        enc   = self.tok(text, max_length=self.maxlen, padding="max_length",
                         truncation=True, return_tensors="pt")
        return {
            "mel":            mel,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }

    def _mel(self, path: str) -> torch.Tensor:
        try:
            import librosa, soundfile as sf
            y, sr = sf.read(path)
            if y.ndim > 1: y = y[:, 0]
            y = y.astype(np.float32)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel = librosa.power_to_db(mel, ref=np.max)
            if mel.shape[1] < self.mframes:
                mel = np.pad(mel, ((0,0),(0, self.mframes - mel.shape[1])))
            else:
                mel = mel[:, :self.mframes]
            return torch.FloatTensor(mel)
        except Exception:
            return torch.zeros(self.n_mels, self.mframes)


# ──────────────────────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────────────────────
class ActionModelTrainer:
    """Train MultimodalActionModel. Returns best checkpoint path."""

    def __init__(self, save_dir: str = "output/models", device: Optional[str] = None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def train(self, train_meta: List[Dict], val_meta: List[Dict],
              epochs: int = 20, batch_size: int = 16,
              lr: float = 2e-5, progress_cb=None) -> str:

        def _cb(p, m):
            if progress_cb: progress_cb(p, m)

        dl_train = DataLoader(ClipDataset(train_meta), batch_size=batch_size,
                              shuffle=True, num_workers=0)
        dl_val   = DataLoader(ClipDataset(val_meta),   batch_size=batch_size,
                              shuffle=False, num_workers=0)
        model = MultimodalActionModel(3).to(self.device)

        counts = [max(sum(1 for m in train_meta if m.get("label") == k), 1)
                  for k in ("weak", "medium", "strong")]
        total  = sum(counts)
        wt     = torch.FloatTensor([total / c for c in counts]).to(self.device)
        crit   = nn.CrossEntropyLoss(weight=wt)
        opt    = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

        best_acc = 0.0
        best_path = os.path.join(self.save_dir, "best_action_model.pth")
        history: Dict[str, list] = {"train_loss": [], "train_acc": [],
                                     "val_loss":   [], "val_acc":   []}
        patience = 0
        STOP = 7

        for ep in range(epochs):
            model.train()
            tl, tc, tt = 0.0, 0, 0
            for b in dl_train:
                mel  = b["mel"].to(self.device)
                ids  = b["input_ids"].to(self.device)
                mask = b["attention_mask"].to(self.device)
                lbl  = b["label"].to(self.device)
                opt.zero_grad()
                out  = model(mel, ids, mask)
                loss = crit(out, lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item() * lbl.size(0)
                tc += (out.argmax(1) == lbl).sum().item()
                tt += lbl.size(0)

            model.eval()
            vl, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for b in dl_val:
                    mel  = b["mel"].to(self.device)
                    ids  = b["input_ids"].to(self.device)
                    mask = b["attention_mask"].to(self.device)
                    lbl  = b["label"].to(self.device)
                    out  = model(mel, ids, mask)
                    loss = crit(out, lbl)
                    vl += loss.item() * lbl.size(0)
                    vc += (out.argmax(1) == lbl).sum().item()
                    vt += lbl.size(0)

            t_loss, t_acc = tl/max(tt,1), tc/max(tt,1)
            v_loss, v_acc = vl/max(vt,1), vc/max(vt,1)
            sched.step(v_loss)

            for k, v in zip(("train_loss","train_acc","val_loss","val_acc"),
                             (t_loss, t_acc, v_loss, v_acc)):
                history[k].append(round(float(v), 4))

            msg = (f"Epoch {ep+1}/{epochs} | "
                   f"train_loss={t_loss:.4f} acc={t_acc:.3f} | "
                   f"val_loss={v_loss:.4f} acc={v_acc:.3f}")
            logger.info(msg)
            _cb((ep+1)/epochs, msg)

            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), best_path)
                patience = 0
            else:
                patience += 1
                if patience >= STOP:
                    logger.info("Early stopping")
                    break

        with open(os.path.join(self.save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        _cb(1.0, f"Training complete — best val_acc={best_acc:.3f}")
        return best_path


# ──────────────────────────────────────────────────────────────
# 2e. INFERENCE
# ──────────────────────────────────────────────────────────────
class ActionInference:
    LABELS = {0: "weak", 1: "medium", 2: "strong"}

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = MultimodalActionModel(3)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.tok = BertTokenizer.from_pretrained("bert-base-chinese")

    def predict(self, wav_path: str, text: str) -> Dict[str, Any]:
        ds   = ClipDataset([{"wav_path": wav_path, "transcript_text": text, "label": "medium"}])
        item = ds[0]
        mel  = item["mel"].unsqueeze(0).to(self.device)
        ids  = item["input_ids"].unsqueeze(0).to(self.device)
        mask = item["attention_mask"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(mel, ids, mask), dim=1).squeeze(0)
            pred  = int(probs.argmax())
        return {
            "label":     self.LABELS[pred],
            "label_idx": pred,
            "probs":     {self.LABELS[i]: round(float(probs[i]), 4) for i in range(3)},
        }


# ──────────────────────────────────────────────────────────────
# FULL TRAINING PIPELINE (entry point)
# ──────────────────────────────────────────────────────────────
def run_full_training_pipeline(
    video_path: str,
    output_dir: str = "output",
    val_ratio:  float = 0.2,
    epochs:     int   = 20,
    batch_size: int   = 16,
    progress_cb=None,
) -> str:
    """
    One-call pipeline: video → clips → pose → label → train → model path
    """
    import random
    from .preprocessing import VideoSlicer

    def _cb(p, m):
        if progress_cb: progress_cb(p, m)

    # Step 1: slice
    _cb(0.0, "Step 1/4: Slicing video …")
    slicer   = VideoSlicer()
    metadata = slicer.slice(
        video_path, output_dir,
        progress_cb=lambda p, m: _cb(p * 0.25, f"[Slice] {m}")
    )

    # Step 2: pose + speech rate
    _cb(0.25, "Step 2/4: Pose extraction …")
    analyzer = PoseAnalyzer()
    speech   = SpeechRateCalc()
    for i, item in enumerate(metadata):
        _cb(0.25 + 0.25 * i / max(len(metadata), 1),
            f"[Pose] {i+1}/{len(metadata)}")
        item.update(analyzer.extract_from_frames_dir(item["frames_dir"]))
        item.update(speech.calc(item["transcript_path"]))

    # Step 3: label
    _cb(0.50, "Step 3/4: Auto-labeling …")
    labeler  = ActionLabeler()
    metadata = labeler.label(metadata)

    # Save
    feats_dir  = os.path.join(output_dir, "features")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(feats_dir,  exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for item in metadata:
        n = item["clip_name"]
        with open(os.path.join(feats_dir,  f"{n}_features.json"), "w") as f:
            json.dump({k: item[k] for k in
                       ("joint_variance","hand_speed","centroid_shift",
                        "token_count","speech_rate") if k in item}, f)
        with open(os.path.join(labels_dir, f"{n}_label.json"),    "w") as f:
            json.dump({"label": item.get("label", "medium")}, f)

    # Step 4: train
    _cb(0.55, "Step 4/4: Training …")
    random.shuffle(metadata)
    split      = max(1, int(len(metadata) * (1 - val_ratio)))
    train_meta = metadata[:split]
    val_meta   = metadata[split:] or metadata[:max(1, split // 5)]
    trainer    = ActionModelTrainer(
        save_dir=os.path.join(output_dir, "models")
    )
    return trainer.train(
        train_meta, val_meta, epochs=epochs, batch_size=batch_size,
        progress_cb=lambda p, m: _cb(0.55 + p * 0.45, f"[Train] {m}")
    )
