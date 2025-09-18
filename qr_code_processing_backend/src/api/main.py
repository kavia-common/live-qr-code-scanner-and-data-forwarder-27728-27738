import os
import threading
import time
from typing import Optional, Dict, Any, List, Tuple, Literal

import cv2  # type: ignore
import httpx
import tempfile
import os as _os
from pytube import YouTube  # type: ignore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
try:
    # Optional dependency; only required if pyzbar selected or as fallback
    from pyzbar.pyzbar import decode as pyzbar_decode  # type: ignore
    _PYZBAR_AVAILABLE = True
except Exception:  # pragma: no cover
    _PYZBAR_AVAILABLE = False

# ZXing-C++ optional Python bindings (prefer official if available)
# There are at least two bindings in the wild:
#  - zxing-cpp (official binding name in PyPI as of 2024/2025 on many platforms)
#  - zxing-cpp-python (community build)
# We try both imports in order. If neither is present or fails to load, we set flag False.
_ZXING_AVAILABLE = False
try:
    # Official binding attempt
    from zxingcpp import read_barcode as zxingcpp_read_barcode  # type: ignore
    _ZXING_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        # Community binding attempt
        # Some distributions expose a similar API under zxingcpp as well,
        # but others under a 'zxing' or 'zxingcpp_python' module. We try a compatible alias.
        from zxingcpp import read_barcode as zxingcpp_read_barcode  # type: ignore
        _ZXING_AVAILABLE = True
    except Exception:  # pragma: no cover
        # As a last resort, try the community package's top-level name if it exposes similar API
        try:
            # Many wheels still use 'zxingcpp' even when package name differs, so nothing else to do here.
            _ZXING_AVAILABLE = False
        except Exception:  # pragma: no cover
            _ZXING_AVAILABLE = False

# Load environment variables from .env if present
load_dotenv()

# -----------------------
# App metadata and theming
# -----------------------

app = FastAPI(
    title="QR Code Processing Backend",
    description="""
A service that:
- Opens a local webcam to scan QR codes in real-time
- Sends decoded QR data to an external API
- Forwards the external API response to a configurable webhook

Style Theme: Ocean Professional (blue & amber accents, modern, clean, minimalist)
    """.strip(),
    version="1.0.0",
    contact={
        "name": "QR Processing Service",
        "url": "https://example.com",
    },
    swagger_ui_parameters={
        # Subtle theming hints (FastAPI does not support full CSS theming out of the box)
        "syntaxHighlight": True,
        "tryItOutEnabled": True,
        "displayRequestDuration": True,
        # Layout and doc expansion
        "docExpansion": "list",
        "defaultModelsExpandDepth": 0,
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Service health and basic info",
        },
        {
            "name": "scanner",
            "description": "Control and observe the QR scanner",
        },
        {
            "name": "config",
            "description": "Manage runtime configuration such as webhook URL and external API",
        },
        {
            "name": "demo",
            "description": "Manual processing utilities for demo/testing",
        },
        {
            "name": "websocket",
            "description": "Notes on real-time usage (no ws endpoint provided in this version)",
        },
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Configuration
# -----------------------

class ServiceConfig(BaseModel):
    """Runtime configuration for the service."""
    external_api_url: HttpUrl = Field(..., description="External API endpoint to send decoded QR data to.")
    external_api_key: Optional[str] = Field(default=None, description="API key (if required) for the external API.")
    webhook_url: HttpUrl = Field(..., description="Webhook URL to forward the external API response to.")
    camera_index: int = Field(default=0, description="Index of the webcam (default 0 for the primary camera).")
    scan_interval_ms: int = Field(default=250, ge=50, le=5000, description="Delay between frame scans in milliseconds.")
    send_unique_only: bool = Field(default=True, description="If true, do not resend identical QR payloads consecutively.")

DEFAULT_EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "https://httpbin.org/post")
DEFAULT_WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://httpbin.org/post")
DEFAULT_API_KEY = os.getenv("EXTERNAL_API_KEY")

CONFIG: ServiceConfig = ServiceConfig(
    external_api_url=DEFAULT_EXTERNAL_API_URL,  # type: ignore[arg-type]
    external_api_key=DEFAULT_API_KEY,
    webhook_url=DEFAULT_WEBHOOK_URL,            # type: ignore[arg-type]
    camera_index=int(os.getenv("CAMERA_INDEX", "0")),
    scan_interval_ms=int(os.getenv("SCAN_INTERVAL_MS", "250")),
    send_unique_only=(os.getenv("SEND_UNIQUE_ONLY", "true").lower() == "true"),
)

# -----------------------
# Models
# -----------------------

class StartScanRequest(BaseModel):
    camera_index: Optional[int] = Field(default=None, description="Override the configured camera index for this session.")
    scan_interval_ms: Optional[int] = Field(default=None, ge=50, le=5000, description="Override scan interval in ms.")

class StopScanResponse(BaseModel):
    status: str = Field(..., description="Status message for stop operation.")

class ProcessQRRequest(BaseModel):
    data: str = Field(..., description="Decoded QR code content to process via external API.")

class ProcessQRResponse(BaseModel):
    message: str = Field(..., description="Human readable message.")
    forwarded_status: Optional[int] = Field(default=None, description="HTTP status from webhook forwarding.")
    external_status: Optional[int] = Field(default=None, description="HTTP status from external API call.")
    external_response_echo: Optional[Dict[str, Any]] = Field(default=None, description="Echo of parsed external response payload (best-effort).")

class ConfigResponse(BaseModel):
    config: ServiceConfig = Field(..., description="Current service configuration.")


class ProcessYouTubeRequest(BaseModel):
    """Request body for processing QR codes from a YouTube video URL."""
    url: HttpUrl = Field(..., description="YouTube video URL to analyze for QR codes.")
    max_frames: Optional[int] = Field(
        default=1500,
        ge=1,
        description="Maximum number of frames to scan before stopping (safety guard).",
    )
    frame_stride: Optional[int] = Field(
        default=5,
        ge=1,
        description="Analyze every Nth frame to reduce processing cost.",
    )
    stop_after_first: Optional[bool] = Field(
        default=True,
        description="If true, stop after detecting the first QR code.",
    )


class ProcessYouTubeResponse(BaseModel):
    """Response containing any decoded QR payloads found in the YouTube video."""
    message: str = Field(..., description="Summary message.")
    detected: List[str] = Field(default_factory=list, description="List of unique decoded QR strings detected.")
    frames_scanned: int = Field(..., description="Number of frames scanned.")


class ProcessVideoURLRequest(BaseModel):
    """Request body for processing QR codes from a direct video file URL."""
    url: HttpUrl = Field(..., description="Direct URL to a video file (e.g., .mp4) to analyze for QR codes.")
    max_frames: Optional[int] = Field(default=1500, ge=1, description="Maximum number of frames to scan before stopping.")
    frame_stride: Optional[int] = Field(default=5, ge=1, description="Analyze every Nth frame to reduce processing cost.")
    stop_after_first: Optional[bool] = Field(default=True, description="If true, stop after detecting the first QR code.")
    # New preprocessing options (all optional; defaults preserve prior behavior)
    grayscale: Optional[bool] = Field(
        default=False,
        description="If true, convert frames to grayscale before detection.",
    )
    adaptive_threshold: Optional[bool] = Field(
        default=False,
        description="If true, apply adaptive thresholding (after grayscale) to increase contrast.",
    )
    denoise: Optional[bool] = Field(
        default=False,
        description="If true, apply light denoising (Gaussian blur) before detection.",
    )
    # Decoder controls
    decoder_backend: Optional[Literal["opencv", "pyzbar", "zxing", "auto"]] = Field(
        default="auto",
        description="Select QR decoder backend. 'opencv' uses OpenCV only, 'pyzbar' uses pyzbar (requires system libzbar), 'zxing' uses ZXing-C++ (requires zxing-cpp binding), 'auto' tries OpenCV then ZXing (if enabled) then pyzbar."
    )
    use_pyzbar_fallback: Optional[bool] = Field(
        default=True,
        description="If true and decoder_backend is 'auto', attempt pyzbar if previous decoders return nothing."
    )
    use_zxingcpp: Optional[bool] = Field(
        default=True,
        description="Enable ZXing-C++ decoding in 'auto' mode when bindings are available. Ignored if ZXing bindings are not installed."
    )


class ProcessVideoURLResponse(BaseModel):
    """Response containing any decoded QR payloads found in the provided video URL."""
    message: str = Field(..., description="Summary message.")
    detected: List[str] = Field(default_factory=list, description="List of unique decoded QR strings detected.")
    frames_scanned: int = Field(..., description="Number of frames scanned.")

class UpdateConfigRequest(BaseModel):
    external_api_url: Optional[HttpUrl] = Field(default=None, description="External API endpoint.")
    external_api_key: Optional[str] = Field(default=None, description="API key for the external API.")
    webhook_url: Optional[HttpUrl] = Field(default=None, description="Webhook endpoint.")
    camera_index: Optional[int] = Field(default=None, description="Webcam index.")
    scan_interval_ms: Optional[int] = Field(default=None, ge=50, le=5000, description="Scan interval in ms.")
    send_unique_only: Optional[bool] = Field(default=None, description="Avoid resending duplicate payloads.")

# -----------------------
# Scanner Service
# -----------------------

class QRScannerService:
    """
    Encapsulates webcam capture and QR detection loop in a background thread.
    Uses OpenCV's QRCodeDetector to decode QR content in frames.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_sent_payload: Optional[str] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._running

    def start(self, camera_index: int, scan_interval_ms: int) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            kwargs={"camera_index": camera_index, "scan_interval_ms": scan_interval_ms},
            name="QRScannerThread",
            daemon=True,
        )
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._running = False

    def _run_loop(self, camera_index: int, scan_interval_ms: int) -> None:
        cap: Optional[cv2.VideoCapture] = None
        detector = cv2.QRCodeDetector()
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open camera index {camera_index}")

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.1)
                    continue

                # Detect and decode with OpenCV (live scanner uses OpenCV path)
                data, _points, _ = detector.detectAndDecode(frame)
                if data:
                    # Optionally skip duplicates
                    with self._lock:
                        if CONFIG.send_unique_only and self._last_sent_payload == data:
                            # Avoid spamming external API with identical consecutive payload
                            pass
                        else:
                            self._last_sent_payload = data
                            # Fire-and-forget processing
                            threading.Thread(
                                target=self._process_payload_safe,
                                args=(data,),
                                name="QRProcessWorker",
                                daemon=True,
                            ).start()

                # Pace the scanning loop
                time.sleep(max(0.0, scan_interval_ms / 1000.0))

        except Exception as exc:
            # In a real application, replace with proper structured logging
            print(f"[Scanner] Error: {exc}")
        finally:
            if cap is not None:
                cap.release()

    def _process_payload_safe(self, payload: str) -> None:
        try:
            # Reuse the API processing pipeline
            # Ignore response; background context
            httpx_client = httpx.Client(timeout=10)
            try:
                external_resp = self._send_to_external(httpx_client, payload)
                self._forward_to_webhook(httpx_client, external_resp)
            finally:
                httpx_client.close()
        except Exception as exc:
            print(f"[Scanner] Processing error: {exc}")

    @staticmethod
    def _send_to_external(client: httpx.Client, data: str) -> httpx.Response:
        headers = {}
        if CONFIG.external_api_key:
            headers["Authorization"] = f"Bearer {CONFIG.external_api_key}"
        # Example: send as JSON to external API
        resp = client.post(str(CONFIG.external_api_url), json={"qr_data": data}, headers=headers)
        return resp

    @staticmethod
    def _forward_to_webhook(client: httpx.Client, external_response: httpx.Response) -> httpx.Response:
        # Forward a summarized payload to the webhook
        payload: Dict[str, Any] = {
            "source": "qr-code-processing-backend",
            "external_status_code": external_response.status_code,
            "external_headers": dict(external_response.headers),
            "external_body": None,
        }
        try:
            payload["external_body"] = external_response.json()
        except Exception:
            payload["external_body"] = external_response.text

        wh_resp = client.post(str(CONFIG.webhook_url), json=payload, headers={"Content-Type": "application/json"})
        return wh_resp


SCANNER = QRScannerService()


def _detect_qr_from_video_file(
    path: str,
    max_frames: int = 1500,
    frame_stride: int = 5,
    stop_after_first: bool = True,
    grayscale: bool = False,
    adaptive_threshold: bool = False,
    denoise: bool = False,
    decoder_backend: str = "auto",
    use_pyzbar_fallback: bool = True,
    use_zxingcpp: bool = True,
) -> Dict[str, Any]:
    """
    Internal helper to iterate frames from a video file and collect decoded QR codes.

    Preprocessing (optional, all default False to preserve prior behavior):
    - grayscale: convert frame to grayscale
    - adaptive_threshold: apply adaptive thresholding (after grayscale)
    - denoise: apply light Gaussian blur (3x3)

    Returns a dict with:
    - detected: List[str] of unique decoded values
    - frames_scanned: int
    """
    detected: List[str] = []
    frames_scanned = 0
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open downloaded/streamed video with OpenCV")

    # Instantiate detector once for efficiency
    opencv_detector = cv2.QRCodeDetector()

    # Local helper: OpenCV decode
    def _decode_with_opencv(img) -> Optional[str]:
        try:
            data, _pts, _ = opencv_detector.detectAndDecode(img)
            if data:
                return data
            return None
        except Exception:
            return None

    # Local helper: pyzbar decode
    def _decode_with_pyzbar(img) -> Optional[str]:
        if not _PYZBAR_AVAILABLE:
            return None
        try:
            results = pyzbar_decode(img)  # type: ignore[name-defined]
            for r in results:
                try:
                    s = r.data.decode("utf-8", errors="ignore")
                except Exception:
                    s = str(r.data)
                if s:
                    return s
            return None
        except Exception:
            return None

    # Local helper: ZXing-C++ decode (supports QR and other symbologies; we only return first text)
    def _decode_with_zxing(img) -> Optional[str]:
        """
        Attempt to decode using ZXing-C++ Python bindings.

        Notes:
        - zxingcpp.read_barcode typically accepts a NumPy array (BGR or gray).
        - We return the 'text' of the first decoded symbol if available.
        """
        if not _ZXING_AVAILABLE or not use_zxingcpp:
            return None
        try:
            # Most bindings autodetect format; pass the frame/processed image as is.
            result = zxingcpp_read_barcode(img)  # type: ignore[name-defined]
            if result is None:
                return None
            # Some bindings return a single result or a list; normalize
            if isinstance(result, list):
                for r in result:
                    text = getattr(r, "text", None) or getattr(r, "decoded_text", None)
                    if text:
                        return str(text)
                return None
            text = getattr(result, "text", None) or getattr(result, "decoded_text", None)
            return str(text) if text else None
        except Exception:
            return None

    try:
        frame_idx = 0
        while frames_scanned < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # stride sampling
            if frame_idx % max(1, frame_stride) != 0:
                frame_idx += 1
                continue

            frames_scanned += 1
            frame_idx += 1

            # Apply simple, optional preprocessing
            processed = frame
            if grayscale:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            if denoise:
                # small Gaussian blur to reduce noise; works with gray or color images
                processed = cv2.GaussianBlur(processed, (3, 3), 0)
            if adaptive_threshold:
                # ensure grayscale before threshold
                if len(processed.shape) == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(
                    processed,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )

            # Detect and decode using selected backend(s)
            decoded_value: Optional[str] = None
            backend_choice = (decoder_backend or "auto").lower()
            if backend_choice == "opencv":
                decoded_value = _decode_with_opencv(processed)
            elif backend_choice == "pyzbar":
                decoded_value = _decode_with_pyzbar(processed)
            elif backend_choice == "zxing":
                decoded_value = _decode_with_zxing(processed)
            else:
                # auto: try OpenCV, then ZXing (if enabled and available), then optional pyzbar
                decoded_value = _decode_with_opencv(processed)
                if not decoded_value and use_zxingcpp:
                    zx = _decode_with_zxing(processed)
                    if zx:
                        decoded_value = zx
                if not decoded_value and use_pyzbar_fallback:
                    decoded_value = _decode_with_pyzbar(processed)

            if decoded_value:
                if decoded_value not in detected:
                    detected.append(decoded_value)
                if stop_after_first:
                    break
    finally:
        cap.release()

    return {"detected": detected, "frames_scanned": frames_scanned}


def _rewrite_dropbox_url_if_needed(url: str) -> str:
    """
    Normalize Dropbox share URLs to a direct-download form so OpenCV/httpx can fetch bytes.

    Dropbox share links often look like:
      - https://www.dropbox.com/s/<id>/<name>.mp4?dl=0
      - https://www.dropbox.com/scl/fi/<id>/<name>.mp4?rlkey=...&dl=0

    For direct download, Dropbox supports:
      - Setting dl=1 on the same host, or
      - Using dl.dropboxusercontent.com with the same path and query (no 'dl' needed).

    We prefer keeping the original host but force dl=1. If 'dl' param is present,
    we set it to 1; if not present, we add dl=1. If the host is 'www.dropbox.com'
    or 'dropbox.com', this rewrite is applied. All other URLs are returned unchanged.
    """
    try:
        from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.endswith("dropbox.com"):
            # Force dl=1 on Dropbox share links
            q = dict(parse_qsl(parsed.query, keep_blank_values=True))
            q["dl"] = "1"
            new_query = urlencode(q, doseq=True)
            rewritten = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
            return rewritten
        return url
    except Exception:
        # Safe fallback: return the original URL if parsing fails
        return url


def _download_video_to_temp(url: str) -> Tuple[str, tempfile.TemporaryDirectory]:
    """
    Download/stream a remote video to a temporary file using httpx streaming.

    Convenience: If the URL is a Dropbox share link (dropbox.com), it is automatically
    rewritten to force a direct download (dl=1). This helps avoid HTML landing pages
    that cannot be opened by OpenCV.

    Returns:
    - (file_path, tempdir_object) where tempdir_object should be kept alive
      until after the caller is done using the file. The caller is responsible
      for cleaning up (context manager recommended).

    Raises:
    - RuntimeError if the download fails or content-type/length is suspicious.
    """
    # Possibly rewrite Dropbox share links to force direct download
    url = _rewrite_dropbox_url_if_needed(url)

    # Create a temp directory the caller will manage
    tmpdir_ctx = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_ctx.name
    target_path = _os.path.join(tmpdir, "remote_video.mp4")

    # Stream download with chunking to avoid loading whole file into memory
    try:
        with httpx.stream("GET", url, timeout=30.0, follow_redirects=True) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code} while downloading video")
            # Not strictly required to be 'video/*', keep permissive
            with open(target_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        # Clean up and re-raise with context
        tmpdir_ctx.cleanup()
        raise RuntimeError(f"Failed to fetch video from URL: {exc}") from exc

    return target_path, tmpdir_ctx

# -----------------------
# Routes
# -----------------------

# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health Check", description="Simple health endpoint to verify the service is running.")
def health_check() -> Dict[str, str]:
    """Return service health."""
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.get("/docs/websocket", tags=["websocket"], summary="WebSocket usage notes", description="Notes for real-time docs visibility. No WebSocket endpoint is provided in this version.")
def websocket_usage_notes() -> Dict[str, str]:
    """Return notes on WebSocket usage."""
    return {
        "note": "This service currently does not expose a WebSocket endpoint. Real-time status can be inferred by polling /scanner/status."
    }

# PUBLIC_INTERFACE
@app.post(
    "/scanner/start",
    tags=["scanner"],
    summary="Start QR scanning",
    description="Start the background webcam scanning loop. Opens the camera and scans for QR codes.",
    status_code=status.HTTP_202_ACCEPTED,
)
def start_scanner(req: StartScanRequest = Body(default=None)) -> Dict[str, Any]:
    """
    Start the QR scanner.

    Parameters:
    - req.camera_index: Optional int to override configured camera index
    - req.scan_interval_ms: Optional int to override configured scan interval

    Returns:
    - dict with running status and active parameters.
    """
    if SCANNER.is_running():
        return {
            "status": "already_running",
            "camera_index": CONFIG.camera_index,
            "scan_interval_ms": CONFIG.scan_interval_ms,
        }

    cam_index = req.camera_index if req and req.camera_index is not None else CONFIG.camera_index
    interval = req.scan_interval_ms if req and req.scan_interval_ms is not None else CONFIG.scan_interval_ms

    try:
        SCANNER.start(camera_index=cam_index, scan_interval_ms=interval)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start scanner: {exc}") from exc

    return {
        "status": "started",
        "camera_index": cam_index,
        "scan_interval_ms": interval,
    }

# PUBLIC_INTERFACE
@app.post(
    "/scanner/stop",
    tags=["scanner"],
    summary="Stop QR scanning",
    description="Stop the background webcam scanning loop and release the camera.",
    response_model=StopScanResponse,
)
def stop_scanner() -> StopScanResponse:
    """Stop the QR scanner."""
    SCANNER.stop()
    return StopScanResponse(status="stopped")

# PUBLIC_INTERFACE
@app.get(
    "/scanner/status",
    tags=["scanner"],
    summary="Scanner status",
    description="Check whether the scanner is running.",
)
def scanner_status() -> Dict[str, Any]:
    """Get current scanner status."""
    return {"running": SCANNER.is_running(), "camera_index": CONFIG.camera_index, "scan_interval_ms": CONFIG.scan_interval_ms}

# PUBLIC_INTERFACE
@app.post(
    "/process",
    tags=["demo"],
    summary="Process a QR payload (manual)",
    description="Manually process QR data: send to external API and forward response to webhook. Useful for testing.",
    response_model=ProcessQRResponse,
)
def process_qr(req: ProcessQRRequest) -> ProcessQRResponse:
    """
    Process QR data by sending it to the external API and forwarding the response to the configured webhook.

    Parameters:
    - data: Decoded QR string

    Returns:
    - Information about external API and webhook forwarding status.
    """
    client = httpx.Client(timeout=10)
    external_status: Optional[int] = None
    forwarded_status: Optional[int] = None
    external_echo: Optional[Dict[str, Any]] = None

    try:
        # Send to external
        external_resp = SCANNER._send_to_external(client, req.data)
        external_status = external_resp.status_code

        # Attempt to parse response for echo
        try:
            external_echo = external_resp.json()  # best effort
        except Exception:
            external_echo = {"raw": external_resp.text}

        # Forward to webhook
        forwarded_resp = SCANNER._forward_to_webhook(client, external_resp)
        forwarded_status = forwarded_resp.status_code

        return ProcessQRResponse(
            message="Processed and forwarded",
            external_status=external_status,
            forwarded_status=forwarded_status,
            external_response_echo=external_echo,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Network error while contacting external/webhook: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        client.close()


# PUBLIC_INTERFACE
@app.post(
    "/process_youtube",
    tags=["demo"],
    summary="Process QR codes from a YouTube video URL",
    description=(
        "Accepts a YouTube video URL, downloads a stream using pytube, extracts frames with OpenCV, "
        "detects QR codes, and returns any decoded data found. "
        "Notes: Large/long videos will be sampled by frame_stride; processing is CPU intensive."
    ),
    response_model=ProcessYouTubeResponse,
)
def process_youtube_video(req: ProcessYouTubeRequest) -> ProcessYouTubeResponse:
    """
    Download or stream a YouTube video and scan frames for QR codes.

    Parameters:
    - url: YouTube video URL
    - max_frames: Max frames to scan (safety bound)
    - frame_stride: Analyze every Nth frame
    - stop_after_first: Stop when first QR is found

    Returns:
    - Message and a list of unique decoded QR strings along with frames scanned.
    """
    # Download the video to a temporary file using pytube
    try:
        yt = YouTube(str(req.url))
        # Get progressive stream with video+audio, lowest resolution is fine for QR detection
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").first()
        if stream is None:
            # Fallback: try any mp4 stream
            stream = yt.streams.filter(file_extension="mp4").first()
        if stream is None:
            raise HTTPException(status_code=400, detail="No suitable MP4 stream found for this YouTube video.")

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = _os.path.join(tmpdir, "video.mp4")
            stream.download(output_path=tmpdir, filename="video.mp4")

            result = _detect_qr_from_video_file(
                target_path,
                max_frames=req.max_frames or 1500,
                frame_stride=req.frame_stride or 5,
                stop_after_first=req.stop_after_first if req.stop_after_first is not None else True,
            )
            detected: List[str] = result["detected"]
            frames_scanned: int = result["frames_scanned"]

            return ProcessYouTubeResponse(
                message="Scan complete" if detected else "No QR codes detected",
                detected=detected,
                frames_scanned=frames_scanned,
            )
    except HTTPException:
        raise
    except Exception as exc:
        # Provide a clear error context
        raise HTTPException(status_code=500, detail=f"Failed to process YouTube video: {exc}") from exc


# PUBLIC_INTERFACE
@app.post(
    "/process_video",
    tags=["demo"],
    summary="Process QR codes from a direct video file URL",
    description=(
        "Accepts a direct video file URL (e.g., mp4), downloads/streams it to a temporary file, "
        "extracts frames with OpenCV, detects QR codes, and returns any decoded data found.\n\n"
        "Notes: Large/long videos will be sampled by frame_stride; processing is CPU intensive. "
        "Ensure the URL is directly accessible by the server and does not require authentication.\n\n"
        "Convenience: If you provide a Dropbox share link (dropbox.com with dl=0 or missing), the service "
        "automatically rewrites it to force a direct download (dl=1) so OpenCV can read the video."
    ),
    response_model=ProcessVideoURLResponse,
)
def process_video_url(req: ProcessVideoURLRequest) -> ProcessVideoURLResponse:
    """
    Download or stream a direct video file URL and scan frames for QR codes.

    Parameters:
    - url: Direct URL to a video file (e.g., .mp4)
    - max_frames: Max frames to scan (safety bound)
    - frame_stride: Analyze every Nth frame
    - stop_after_first: Stop when first QR is found

    Returns:
    - Message and a list of unique decoded QR strings along with frames scanned.
    """
    # Download to temp file using httpx
    try:
        path, tmpctx = _download_video_to_temp(str(req.url))
        try:
            result = _detect_qr_from_video_file(
                path,
                max_frames=req.max_frames or 1500,
                frame_stride=req.frame_stride or 5,
                stop_after_first=req.stop_after_first if req.stop_after_first is not None else True,
                grayscale=bool(req.grayscale) if req.grayscale is not None else False,
                adaptive_threshold=bool(req.adaptive_threshold) if req.adaptive_threshold is not None else False,
                denoise=bool(req.denoise) if req.denoise is not None else False,
                decoder_backend=(req.decoder_backend or "auto"),
                use_pyzbar_fallback=bool(req.use_pyzbar_fallback) if req.use_pyzbar_fallback is not None else True,
                use_zxingcpp=bool(req.use_zxingcpp) if getattr(req, "use_zxingcpp", None) is not None else True,
            )
            detected: List[str] = result["detected"]
            frames_scanned: int = result["frames_scanned"]
            return ProcessVideoURLResponse(
                message="Scan complete" if detected else "No QR codes detected",
                detected=detected,
                frames_scanned=frames_scanned,
            )
        finally:
            # Ensure temp files cleaned
            tmpctx.cleanup()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process video URL: {exc}") from exc

# PUBLIC_INTERFACE
@app.get(
    "/config",
    tags=["config"],
    summary="Get configuration",
    description="Fetch the current runtime configuration.",
    response_model=ConfigResponse,
)
def get_config() -> ConfigResponse:
    """Get current configuration."""
    return ConfigResponse(config=CONFIG)

# PUBLIC_INTERFACE
@app.patch(
    "/config",
    tags=["config"],
    summary="Update configuration",
    description="Update one or more configuration fields at runtime.",
    response_model=ConfigResponse,
)
def update_config(req: UpdateConfigRequest) -> ConfigResponse:
    """Update configuration safely."""
    global CONFIG
    # Apply updates
    new_values = CONFIG.model_dump()
    if req.external_api_url is not None:
        new_values["external_api_url"] = str(req.external_api_url)
    if req.external_api_key is not None:
        new_values["external_api_key"] = req.external_api_key
    if req.webhook_url is not None:
        new_values["webhook_url"] = str(req.webhook_url)
    if req.camera_index is not None:
        new_values["camera_index"] = req.camera_index
    if req.scan_interval_ms is not None:
        new_values["scan_interval_ms"] = req.scan_interval_ms
    if req.send_unique_only is not None:
        new_values["send_unique_only"] = req.send_unique_only

    # Revalidate via model
    CONFIG = ServiceConfig(**new_values)  # type: ignore[arg-type]
    return ConfigResponse(config=CONFIG)
