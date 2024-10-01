#!/bin/bash

# Remove existing pip libraries and cached files
rm -r /tmp/trood/pip
mkdir /tmp/trood/pip
pip cache purge
pip freeze | xargs pip uninstall -y

# Activate environment
source ~/.bashrc
conda activate thesis
export HF_HOME="/tmp/trood"

# Keys and tokens
export HF_TOKEN="KEY"
export OPENAI_API_KEY="KEY"
export GOOGLE_API_KEY="KEY"
export PIP_TARGET="/tmp/trood"

# Add pip-installed directory to PATH and PYTHONPATH
export PATH="/tmp/trood/pip/bin:$PATH"
export PYTHONPATH="/tmp/trood/pip:$PYTHONPATH"

# Install requirements
pip install -r requirements.txt --target /tmp/trood/pip --upgrade
pip install huggingface_hub --target /tmp/trood/pip  # Install Hugging Face CLI explicitly
find /tmp/trood/pip -name huggingface-cli

# Check if Hugging Face token is passed as an environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Hugging Face token is not set. Please set the HF_TOKEN environment variable."
    exit 1
fi

# Automatically log in to Hugging Face CLI using the token and pipe "Y" for the Git credential prompt
echo "Logging in to Hugging Face using the provided token..."
yes Y | huggingface-cli login --token "$HF_TOKEN"

# Check if login was successful
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "Login successful."
else
    echo "Login failed. Please verify your token."
    exit 1
fi

# Change Hugging Face token permissions for safety
chmod 700 /tmp/trood/token

# Show Python being used (optional for debugging)
echo "Python being used in bash script:"
which python
