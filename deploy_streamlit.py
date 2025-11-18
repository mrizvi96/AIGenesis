#!/usr/bin/env python3
"""
Quick deployment script for Streamlit Cloud
Run this to test the application locally before deploying
"""

import subprocess
import sys
import os

def main():
    print("🚀 AI Insurance Claims Processing - Streamlit Deployment")
    print("=" * 60)

    # Install streamlit if not installed
    try:
        import streamlit
        print("✅ Streamlit already installed")
    except ImportError:
        print("📦 Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

    # Run the Streamlit app
    print("🌐 Starting Streamlit application...")
    print("📍 Local URL: http://localhost:8501")
    print("📍 Network URL: http://localhost:8501")
    print("\n💡 For Streamlit Cloud deployment:")
    print("   1. Push this repository to GitHub")
    print("   2. Go to https://share.streamlit.io/")
    print("   3. Connect your GitHub account")
    print("   4. Select this repository and main.py file")
    print("   5. The app will be deployed at a share.streamlit.io URL")

    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")

if __name__ == "__main__":
    main()