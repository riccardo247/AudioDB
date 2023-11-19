#!/bin/bash

venv_dir="/ve/audiodb"
n=5  # Number of directories and scripts to run. Adjust as needed.
audio_in_path="../yt_audio"
script_path="./ProcessAudio.py"

# Create the virtual environment
python3 -m venv "$venv_dir"

# Activate the virtual environment
source "$venv_dir/bin/activate"

export OPENAI_KEY_API=""

# Install cpu/cuda pytorch (>=1.9) dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install DeepFilterNet
pip install deepfilternet
# Or install DeepFilterNet including data loading functionality for training (Linux only)
pip install deepfilternet[train]


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
