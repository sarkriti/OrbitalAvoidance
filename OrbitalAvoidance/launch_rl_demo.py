#!/usr/bin/env python3
"""
Launch Enhanced RL Web Demo

This script launches the enhanced web simulator with automatic detection
of your trained 100k-step PPO model for visual demonstration.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

def main():
    """Launch the enhanced web simulator with RL demo."""
    
    print("🚀 Launching Enhanced RL Orbital Avoidance Demo")
    print("=" * 55)
    
    # Verify trained models exist
    model_dir = Path("trained_models/debris_avoidance_ppo")
    if not model_dir.exists():
        print("❌ No trained models found!")
        print("Expected directory: trained_models/debris_avoidance_ppo/")
        return
    
    # List available models
    print("📦 Available trained models:")
    model_files = list(model_dir.glob("*.zip"))
    for model_file in model_files:
        print(f"  ✅ {model_file.name}")
    
    if not model_files:
        print("❌ No .zip model files found in trained_models/debris_avoidance_ppo/")
        return
    
    print(f"\n🎯 Best model: {model_dir}/best_model.zip")
    print(f"🎯 Final model: {model_dir}/final_model.zip")
    
    # Import and start the enhanced web simulator
    print("\n🌐 Starting Enhanced Web Simulator...")
    print("🔧 The simulator includes:")
    print("  • Automatic model detection")
    print("  • Real-time control mode switching") 
    print("  • RL vs Manual performance comparison")
    print("  • Live action visualization")
    
    try:
        # Import the enhanced web simulator
        from enhanced_web_simulator import main as start_simulator
        
        print("\n📱 Opening browser in 3 seconds...")
        print("🎮 Demo Instructions:")
        print("  1. Wait for 3D visualization to load")
        print("  2. Select your trained model from the dropdown")
        print("  3. Switch control mode to 'RL Agent'")
        print("  4. Watch your AI navigate and avoid debris!")
        print("  5. Compare with 'Manual' or 'Zero' modes")
        
        print(f"\n👀 What to expect:")
        print("  • Smooth AI-controlled orbital maneuvers")
        print("  • Intelligent debris avoidance") 
        print("  • Fuel-efficient trajectory planning")
        print("  • Real-time thrust vector visualization")
        
        print("\n🔗 URL: http://localhost:8080")
        print("⚡ Powered by your 100k-step PPO model")
        print("\nPress Ctrl+C to stop the demo")
        print("=" * 55)
        
        # Auto-open browser after a short delay
        import threading
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8080")
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Start the enhanced web simulator
        start_simulator()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the OrbitalAvoidance directory")
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error starting demo: {e}")

if __name__ == "__main__":
    main()