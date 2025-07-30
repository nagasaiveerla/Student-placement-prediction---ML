#!/usr/bin/env python3
"""
Launcher script for the FastAPI server
"""

import uvicorn
import sys
import os

def main():
    # Check if model files exist
    if not os.path.exists("models/placement_model.pkl"):
        print("❌ Model files not found!")
        print("Please run training first: python train_model.py")
        return False
    
    try:
        print("🚀 Starting FastAPI server...")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 Interactive docs at: http://localhost:8000/docs")
        print("📖 ReDoc docs at: http://localhost:8000/redoc")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run FastAPI with uvicorn
        uvicorn.run(
            "src.fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["src"],
            log_level="info"
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
        return True
    except Exception as e:
        print(f"❌ Failed to start FastAPI server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)