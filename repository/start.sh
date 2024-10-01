#!/bin/bash

# Remove existing pip libraries and cached files
rm -r /tmp/trood/pip
mkdir /tmp/trood/pip
pip cache purge
pip freeze | xargs pip uninstall -y

# Activate environment
source ~/.bashrc
conda activate thesis

# Load environment variables from .env file if present
if [ -f .env ]; then
    echo ".env file found. Loading environment variables..."
    source .env
else
    echo "Warning: .env file not found. Ensure all required environment variables are set externally."
fi

# Ensure all sensitive environment variables are set
if [ -z "$HF_TOKEN" ] || [ -z "$OPENAI_API_KEY" ] || [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: One or more API keys are not set. Please set HF_TOKEN, OPENAI_API_KEY, and GOOGLE_API_KEY."
    exit 1
fi

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

# Change .env permissions for safety
chmod 700 .env

# Show Python being used (optional for debugging)
echo "Python being used in bash script:"
which python
