# QR Code Processing Backend

A FastAPI backend that:
- Opens a webcam to scan QR codes in real-time using OpenCV
- Sends decoded QR data to an external API
- Forwards the external API response to a configurable webhook

Theme: Ocean Professional (blue & amber accents, modern, clean, minimalist)

## Features

- Start/stop real-time scanning via REST
- Manual processing endpoint for testing
- Configurable external API and webhook
- OpenAPI/Swagger docs with enhanced parameters
- Safe background threading

## Requirements

- Python 3.10+
- A local webcam (or virtual camera device)
- Internet access for calling external API and webhook (default: httpbin.org)

## Setup

1. Create and activate a virtual environment (optional):
   - python -m venv .venv
   - source .venv/bin/activate

2. Install dependencies:
   - pip install -r requirements.txt

3. Create a .env file (optional). See .env.example for variables.

4. Run the server:
   - uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs available at: http://localhost:8000/docs

## Environment Variables

See `.env.example`:
- EXTERNAL_API_URL: External API endpoint to receive QR data
- EXTERNAL_API_KEY: Optional bearer token for external API
- WEBHOOK_URL: Webhook to forward external response
- CAMERA_INDEX: Webcam device index (default 0)
- SCAN_INTERVAL_MS: Milliseconds between frame scans (default 250)
- SEND_UNIQUE_ONLY: "true" or "false" to suppress duplicate consecutive sends
- CORS_ALLOW_ORIGINS: Comma-separated list of origins (default "*")

## API Overview

- GET `/` — Health
- GET `/docs/websocket` — Notes (no WS in this version)
- POST `/scanner/start` — Start live scanning
  - body: { "camera_index"?: int, "scan_interval_ms"?: int }
- POST `/scanner/stop` — Stop scanning
- GET `/scanner/status` — Get running status
- POST `/process` — Manually process a QR payload
  - body: { "data": "..." }
- GET `/config` — Get current config
- PATCH `/config` — Update config fields at runtime
- POST `/process_youtube` — Process QR codes from a YouTube video URL
  - body:
    {
      "url": "https://www.youtube.com/watch?v=...",
      "max_frames": 1500,
      "frame_stride": 5,
      "stop_after_first": true
    }
  - returns: { "message": "...", "detected": ["QR1", "QR2"], "frames_scanned": 123 }
- POST `/process_video` — Process QR codes from a direct video file URL (e.g., mp4)
  - body:
    {
      "url": "https://example.com/video.mp4",
      "max_frames": 1500,
      "frame_stride": 5,
      "stop_after_first": true,
      "grayscale": false,
      "adaptive_threshold": false,
      "denoise": false
    }
  - returns: { "message": "...", "detected": ["QR1", "QR2"], "frames_scanned": 123 }
  - notes:
    - The server must be able to access the URL directly (no auth).
    - Large videos are sampled by frame_stride.
    - Optional preprocessing flags can help in noisy/low-contrast videos:
      - grayscale: convert to grayscale before detection
      - adaptive_threshold: increase contrast via adaptive thresholding (applies after grayscale)
      - denoise: light Gaussian blur to reduce noise
    - All preprocessing options default to false to preserve previous behavior.

### Processing Pipeline

1. QR decoded from webcam
2. POST to external API:
   - URL: `EXTERNAL_API_URL`
   - Body: `{ "qr_data": "<decoded>" }`
   - Authorization: `Bearer <EXTERNAL_API_KEY>` (if provided)
3. Forward external response to webhook:
   - URL: `WEBHOOK_URL`
   - Body (summary): `{ source, external_status_code, external_headers, external_body }`

## Notes

- If the webcam cannot open, you'll see a 500 error on start; ensure your device index is correct.
- For testing without a webcam, use `/process` to simulate a decoded QR value.

## License

MIT
