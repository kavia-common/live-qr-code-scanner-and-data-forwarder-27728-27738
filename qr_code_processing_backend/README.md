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
- For pyzbar decoder (optional): system library libzbar must be installed (e.g., Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y libzbar0`)
- For ZXing-C++ decoder (optional): Python bindings for zxing-cpp must be installed. Prefer `pip install zxing-cpp` (official wheels). If not available for your platform, try `pip install zxing-cpp-python`. The API will skip ZXing if bindings are missing; set `"use_zxing": false` to explicitly disable it.

### ZXing-C++ installation notes

- Preferred (when wheels available):
  - `pip install zxing-cpp`
- Alternative (if no official wheels for your OS/arch/Python version):
  - `pip install zxing-cpp-python`
- If neither provides a wheel for your environment, building from source may be required (not covered here). In such cases, keep `use_zxingcpp=false` or select a different decoder backend.
- Platform caveats:
  - Some environments (e.g., ARM, Alpine/musl, older Python) may lack prebuilt wheels. The API gracefully skips ZXing if import fails.
  - No additional system libraries are typically required for zxing-cpp wheels. If building from source, consult the zxing-cpp project for prerequisites.

Both ZXing-C++ bindings are optional at runtime. The API will silently skip ZXing if the module is not present. You can also explicitly disable ZXing for a request via `"use_zxing": false` in `/process_video`.

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
      "denoise": false,
      "decoder_backend": "auto",
      "use_opencv": true,
      "use_pyzbar": true,
      "use_zxing": true,

      // backward-compatibility aliases (optional):
      "use_pyzbar_fallback": true,
      "use_zxingcpp": true
    }
  - returns: { "message": "...", "detected": ["QR1", "QR2"], "frames_scanned": 123 }
  - notes:
    - The server must be able to access the URL directly (no auth).
    - Large videos are sampled by frame_stride.
    - Dropbox convenience: If you provide a Dropbox share link (e.g. dl=0), the server will rewrite it to force a direct download (dl=1) so OpenCV can read the bytes.
    - Optional preprocessing flags can help in noisy/low-contrast videos:
      - grayscale: convert to grayscale before detection
      - adaptive_threshold: increase contrast via adaptive thresholding (applies after grayscale)
      - denoise: light Gaussian blur to reduce noise
    - Decoder selection:
      - decoder_backend: "opencv" | "pyzbar" | "zxing" | "auto" (default "auto").
        - "opencv": uses OpenCV QRCodeDetector.
        - "pyzbar": uses pyzbar (requires system libzbar).
        - "zxing": uses ZXing-C++ (requires zxing-cpp Python bindings).
        - "auto": tries enabled backends in order: OpenCV → ZXing → pyzbar.
      - Explicit backend flags (all default true):
        - use_opencv: enable/disable OpenCV path
        - use_pyzbar: enable/disable pyzbar path
        - use_zxing: enable/disable ZXing path
      - Back-compatibility:
        - use_pyzbar_fallback maps to use_pyzbar when decoder_backend="auto"
        - use_zxingcpp maps to use_zxing
    - All preprocessing options default to false to preserve previous behavior.

Examples:
- Force OpenCV only:
  {
    "url": "https://example.com/video.mp4",
    "decoder_backend": "opencv",
    "use_opencv": true,
    "use_pyzbar": false,
    "use_zxing": false
  }

- Auto with ZXing disabled:
  {
    "url": "https://example.com/video.mp4",
    "decoder_backend": "auto",
    "use_zxing": false
  }

- Prefer ZXing explicitly:
  {
    "url": "https://example.com/video.mp4",
    "decoder_backend": "zxing",
    "use_zxing": true
  }

### ZXing-C++ caveats
- ZXing bindings may not be available for all OS/architecture/Python combos. If import fails, the server continues without ZXing and logs nothing; the "auto" path just skips that step.
- When forcing `decoder_backend="zxing"` and ZXing is not installed, no decoding will occur (returns empty). Prefer `auto` with `use_zxingcpp=true` for portability.

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
