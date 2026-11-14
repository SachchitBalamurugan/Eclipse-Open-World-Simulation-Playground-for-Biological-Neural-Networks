#!/bin/bash
echo "R=$(ls /home/ubuntu/nr/neurorover/results 2>/dev/null | wc -l)"
echo "P=$(pgrep -fc 'neurorover.bench')"
echo "N=$(nproc)"
echo "L=$(cat /proc/loadavg)"
echo "M=$(free -m | awk '/Mem:/{print $3" "$2}')"
echo "---LOG---"
tail -q -n 5 /home/ubuntu/nr/bench*.log 2>/dev/null
