#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_launchd_schedule.sh            # default 09:35 local time daily
#   ./setup_launchd_schedule.sh 10 15      # run daily at 10:15 local time

HOUR="${1:-9}"
MINUTE="${2:-35}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
RUN_SCRIPT="$SCRIPT_DIR/run_daily_live_cycle.py"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python venv executable at: $PYTHON_BIN"
  exit 1
fi
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script at: $RUN_SCRIPT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

LABEL="com.alpha.generator.daily"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>${RUN_SCRIPT}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${SCRIPT_DIR}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${HOUR}</integer>
    <key>Minute</key>
    <integer>${MINUTE}</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>${OUTPUT_DIR}/launchd_daily_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${OUTPUT_DIR}/launchd_daily_stderr.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "Installed daily schedule ${HOUR}:${MINUTE} (local time)"
echo "LaunchAgent: $PLIST_PATH"
