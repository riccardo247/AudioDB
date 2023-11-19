#!/bin/bash

n=5  # Number of directories and scripts to run. Adjust as needed.
audio_in_path="../yt_audio"
script_path="./ProcessAudio.py"
export OPENAI_KEY_API=""

for ((i=1; i<=n; i++)); do
  out_dir="../out_audio_$i"
  log_file="../log_$i.txt"
  skip_n=$(( (i-1) *300 ))
  load_n=300

  # Create output directory
  mkdir -p "$out_dir"

  # Run script in background
  nohup python3 "$script_path" "$audio_in_path" "$out_dir" --skip_n "$skip_n" --load_n "$load_n" > "$log_file" 2>&1 &

  echo "Script $i started, output directory: $out_dir, log file: $log_file"
done

echo "All scripts started in background."
