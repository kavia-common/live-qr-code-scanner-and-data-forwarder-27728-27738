#!/usr/bin/env python3
import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import httpx
import cv2  # type: ignore
import numpy as np  # type: ignore


def rewrite_dropbox_url(url: str) -> str:
    try:
        p = urlparse(url)
        if p.netloc.lower().endswith("dropbox.com"):
            q = dict(parse_qsl(p.query, keep_blank_values=True))
            q["dl"] = "1"
            return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), p.fragment))
    except Exception:
        pass
    return url


def download_video(url: str, timeout: float = 60.0) -> str:
    url = rewrite_dropbox_url(url)
    td = tempfile.TemporaryDirectory()
    target = os.path.join(td.name, "video.mp4")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_bytes():
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        td.cleanup()
        raise RuntimeError(f"Failed to download video: {exc}") from exc
    # return path and keep tempdir alive by attaching to path object (store sentinel)
    # We return just the path; caller should process promptly. Alternatively, copy elsewhere.
    return target


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract first frame, preprocess, and attempt QR detection.")
    parser.add_argument("--url", required=True, help="Video URL (Dropbox links supported; dl rewritten to 1).")
    parser.add_argument("--outdir", default="utils/output/first_frame_analysis", help="Directory to save artifacts.")
    parser.add_argument("--grayscale", action="store_true", default=True, help="Apply grayscale (default on).")
    parser.add_argument("--adaptive_threshold", action="store_true", default=True, help="Apply adaptive threshold (default on).")
    parser.add_argument("--denoise", action="store_true", default=True, help="Apply Gaussian blur (default on).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Download and open
    rewritten_url = rewrite_dropbox_url(args.url)
    video_path = download_video(rewritten_url)

    cap = cv2.VideoCapture(video_path)
    diagnostics = {
        "input_url": args.url,
        "rewritten_url": rewritten_url,
        "cap_opened": bool(cap.isOpened()),
        "first_frame_read": False,
        "frame_shape": None,
        "preprocessing": {
            "grayscale": bool(args.grayscale),
            "adaptive_threshold": bool(args.adaptive_threshold),
            "denoise": bool(args.denoise),
        },
        "qr": {
            "detected": False,
            "data": "",
            "polygon": None,
        },
        "artifact_paths": {},
    }

    if not cap.isOpened():
        report_path = outdir / "report.json"
        report_path.write_text(json.dumps(diagnostics, indent=2))
        print(json.dumps(diagnostics, indent=2))
        return 2

    ret, frame = cap.read()
    cap.release()
    diagnostics["first_frame_read"] = bool(ret)
    if not ret or frame is None:
        report_path = outdir / "report.json"
        report_path.write_text(json.dumps(diagnostics, indent=2))
        print(json.dumps(diagnostics, indent=2))
        return 3

    diagnostics["frame_shape"] = list(frame.shape)

    # Save original
    orig_path = outdir / "first_frame.jpg"
    cv2.imwrite(str(orig_path), frame)
    diagnostics["artifact_paths"]["first_frame"] = str(orig_path)

    processed = frame.copy()
    if args.grayscale:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        gray_path = outdir / "first_frame_gray.jpg"
        cv2.imwrite(str(gray_path), processed)
        diagnostics["artifact_paths"]["first_frame_gray"] = str(gray_path)

    if args.denoise:
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        denoise_path = outdir / "first_frame_denoise.jpg"
        cv2.imwrite(str(denoise_path), processed)
        diagnostics["artifact_paths"]["first_frame_denoise"] = str(denoise_path)

    if args.adaptive_threshold:
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
        thr_path = outdir / "first_frame_threshold.jpg"
        cv2.imwrite(str(thr_path), processed)
        diagnostics["artifact_paths"]["first_frame_threshold"] = str(thr_path)

    # Run detection
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(processed)
    if data:
        diagnostics["qr"]["detected"] = True
        diagnostics["qr"]["data"] = data
    if points is not None:
        diagnostics["qr"]["polygon"] = points.tolist()

    # If polygon present and original frame is available, outline it
    if points is not None and frame is not None:
        overlay = frame.copy()
        pts = points.astype(int).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 165, 255), thickness=3)
        overlay_path = outdir / "first_frame_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        diagnostics["artifact_paths"]["first_frame_overlay"] = str(overlay_path)

    # Save report
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(diagnostics, indent=2))

    print(json.dumps(diagnostics, indent=2))
    # exit code indicates detection status
    return 0 if diagnostics["qr"]["detected"] else 1


if __name__ == "__main__":
    sys.exit(main())
