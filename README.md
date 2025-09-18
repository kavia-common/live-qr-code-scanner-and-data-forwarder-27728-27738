# live-qr-code-scanner-and-data-forwarder-27728-27738

This workspace contains the QR Code Processing Backend (FastAPI) that scans a webcam for QR codes, sends decoded data to an external API, and forwards responses to a webhook.

Quick start:
- cd qr_code_processing_backend
- Create .env from .env.example (optional)
- pip install -r requirements.txt
- uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs