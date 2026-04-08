# output/

Runtime output folder. Files here are **gitignored** (except this README).

## ⚠️ Required: Place Demo Video

| File | Purpose |
|------|---------|
| `完整moocs影片產出.mp4` | **Required** — served in Step 6 demo mode |

## Auto-generated at runtime

| Pattern | Description |
|---------|-------------|
| `{session}_{title}_slides.pptx` | Generated PPTX presentation |
| `{session}_script.txt` | Generated lecture script |
| `processed.wav` | Denoised + VAD audio |
| `moocs_output.mp4` | Real-mode video output |
| `training/best_action_model.pth` | Trained action model |
| `training/training_history.json` | Loss/accuracy history |
| `sd_images/` | SD-generated slide images |
