#!/usr/bin/env python3
"""
Launcher script for the Streamlit web application
"""

import subprocess
import sys
import os

def main():
    # Check if model files exist
    if not os.path.exists("models/placement_model.pkl"):
        print("❌ Model files not found!")
        print("Please run training first: python train_model.py")
        return False
    
    try:
        print("🚀 Starting Streamlit application...")
        print("📱 The app will open in your browser at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped")
        return True
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)