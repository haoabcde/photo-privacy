#!/bin/bash
set -e

echo "Installing PyInstaller..."
pip3 install pyinstaller

echo "Building PhotoPrivacy for macOS..."
python3 -m PyInstaller --noconfirm --name PhotoPrivacy \
            --windowed \
            --add-data "templates:templates" \
            --add-data "static:static" \
            --add-data "models:models" \
            launcher.py

echo "Build complete! Check the 'dist' folder for PhotoPrivacy.app."
