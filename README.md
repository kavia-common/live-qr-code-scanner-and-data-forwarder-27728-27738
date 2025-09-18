# live-qr-code-scanner-and-data-forwarder-27728-27738

This workspace contains the QR Code Processing Backend (FastAPI) that scans a webcam for QR codes, sends decoded data to an external API, and forwards responses to a webhook.

Quick start:
- cd qr_code_processing_backend
- Create .env from .env.example (optional)
- pip install -r requirements.txt
- uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs

New:
- POST /process_youtube — Provide a YouTube URL to scan frames for QR codes without a webcam.
- POST /process_video — Provide a direct video URL (e.g., .mp4) to scan frames for QR codes without a webcam. Dropbox share links are auto-rewritten to direct download (dl=1) for convenience.
- Decoder selection in /process_video:
  - decoder_backend: "opencv" | "pyzbar" | "zxing" | "auto"
  - Flags: use_opencv, use_pyzbar, use_zxing (all default true; control which backends are tried)
  - Back-compat aliases: use_pyzbar_fallback, use_zxingcpp (mapped internally)
- Optional ZXing-C++ decoder (zxing-cpp) supported. See backend README for installation notes and platform caveats.