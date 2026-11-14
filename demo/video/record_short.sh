#!/bin/bash
# Records the scripted BNN Robot Studio demo window (only the app window) to MP4.
cd "$(dirname "$0")/../../app"
OUT=../demo/video
rm -f $OUT/raw.mkv
node_modules/electron/dist/electron.exe . --demo-short --demo-fleet-run=arm3_teach_fleet_s31_20261114-1515 > $OUT/electron.log 2>&1 &
EPID=$!
for i in $(seq 1 60); do sleep 0.5; tasklist //v //fi "imagename eq electron.exe" 2>/dev/null | grep -q "Eclipse Demo" && break; done
sleep 1
ffmpeg -hide_banner -loglevel error -f gdigrab -framerate 30 -offset_x 0 -offset_y 0 -video_size 1920x1080 -i desktop -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -t 45 $OUT/raw.mkv &
FPID=$!
wait $EPID
sleep 1
kill -INT $FPID 2>/dev/null; wait $FPID 2>/dev/null
ffmpeg -hide_banner -loglevel error -y -i $OUT/raw.mkv -c copy -movflags +faststart $OUT/eclipse_demo_30s.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 $OUT/eclipse_demo_30s.mp4
