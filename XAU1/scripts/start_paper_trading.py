#!/usr/bin/env python3
"""
XAU1 Paper Trading Launcher
Quick script to start paper trading dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the paper trading dashboard"""
    print("🚀 Starting XAU1 Paper Trading Dashboard...")
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        # Run the paper trading dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/xau1/dashboard/paper_trading_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()