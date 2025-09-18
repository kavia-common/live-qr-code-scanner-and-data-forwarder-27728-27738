# Utils

This folder contains helper scripts for analysis and diagnostics.

## extract_first_frame_and_qr_scan.py

Downloads a video (supports Dropbox share links; automatically rewrites `dl=0` to `dl=1`), extracts the first frame (timestamp 0.0), applies preprocessing (grayscale, denoise, adaptive threshold), and attempts full-frame QR detection. Saves artifacts and a JSON report.

Usage:

```bash
python utils/extract_first_frame_and_qr_scan.py \
  --url "https://www.dropbox.com/scl/fi/kc29p81u69pouhsdkr6d4/2023-Conference-Trade-Show-Interview-Series-Part-2.mp4?rlkey=qzevjngye7q4g15mhw9uwh0lm&dl=0" \
  --outdir utils/output/first_frame_analysis
```

Artifacts saved:
- first_frame.jpg (original frame)
- first_frame_gray.jpg (grayscale)
- first_frame_denoise.jpg (after Gaussian blur)
- first_frame_threshold.jpg (after adaptive threshold)
- first_frame_overlay.jpg (only if a QR polygon is detected)
- report.json (diagnostics and detection results)

Exit code:
- 0: QR detected
- 1: No QR detected
- 2/3: Errors loading video/first frame
