#!/bin/bash

# Define the virtual environment directory and the libraries to install
venv_dir="/ve/salmonn"
libraries=("soundfile" "librosa" "torch==2.0.1" "transformers==4.28.0" "peft==0.3.0", "torchaudio==2.0.2", "sentencepiece")

# Create the virtual environment
python3 -m venv "$venv_dir"

# Activate the virtual environment
source "$venv_dir/bin/activate"

# Install the required libraries
for library in "${libraries[@]}"; do
    pip3 install "$library"
done

# Create the salmonn directory
mkdir /salmonn

# Change to the salmonn directory
cd /salmonn

#make some directories
mkdir beats
mkdir ckpt

#install git large files
git lfs install

# Clone the Whisper model
git clone https://huggingface.co/openai/whisper-large-v2 whisper

#get beats model
curl -O "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
mv "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" ./beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt

#clone vicuna model
git clone https://huggingface.co/lmsys/vicuna-13b-v1.1/ ./vicuna

#clone SALMONN from hf
git clone https://huggingface.co/MSIIP/SALMONN/ ./SALMONN_hf
cd ./SALMONN_hf
git lfs pull
cp ./salmonn_v1.pth ../ckpt/salmonn_v1.pth

cd /salmonn
# Clone the SALMONN repository
git clone https://github.com/riccardo247/SALMONN

#install few other libraries
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
# Change to the SALMONN directory
cd SALMONN