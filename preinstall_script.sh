#!/bin/bash

# Convenience script for setting Udacity workspace
apt-get update
apt-get install -y cuda-libraries-9-0  # Source Eric Lavigne on Udacity slack
pip install scikit-video
pip install opencv-python
pip install --upgrade tensorflow-gpu