#!/bin/bash
cd /home/kavia/workspace/code-generation/live-qr-code-scanner-and-data-forwarder-27728-27738/qr_code_processing_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

