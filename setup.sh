#!/bin/bash

# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# Install Python3 and pip
sudo apt install python3 python3-pip -y

# Install required Python libraries
pip3 install --upgrade pip
pip3 install opencv-python-headless numpy

# Notify user of successful setup
echo "Setup complete. You can now run the script directly using Python3."
