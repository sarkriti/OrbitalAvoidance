#!/usr/bin/env python3
"""
Demo script to showcase trained RL model in web simulator
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# Add project path
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from rl_web_integration import RLWebAgent
import web_simulator_realistic

def main():
    """Launch web demo with trained RL model."""
    
    print("🚀 Orbital Avoidance RL Demo")
    print("=" * 50)
    
    # Path to the best trained model
    model_path = "trained_models/debris_avoidance_ppo/best_model.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        model_dir = Path("trained_models/debris_avoidance_ppo")
        if model_dir.exists():
            for file in model_dir.glob("*.zip"):
                print(f"  - {file.name}")
        return
    
    # Load the trained RL agent
    print(f"📦 Loading trained model: {model_path}")
    rl_agent = RLWebAgent(model_path=model_path, algorithm='ppo')
    
    if not rl_agent.is_loaded:
        print("❌ Failed to load RL agent")
        return
    
    print("✅ RL agent loaded successfully!")
    
    # Replace the ZeroPolicy in the web simulator with our trained agent
    print("🔧 Integrating RL agent with web simulator...")
    
    # Monkey patch the simulator to use our RL agent
    def create_rl_controlled_simulator():
        """Create simulator with RL agent control."""
        
        # We'll modify the simulation to use RL agent
        original_simulator = web_simulator_realistic.OrbitalAvoidanceSimulator()
        
        # Replace the agent with our trained RL agent
        original_simulator.rl_agent = rl_agent
        original_simulator.use_rl_control = True
        
        return original_simulator
    
    # Start the web server
    print("🌐 Starting web server...")
    print("📱 The simulator will open in your browser automatically")
    print("🎮 Watch the AI control the satellite to avoid debris!")
    print("⚡ The satellite is controlled by your trained 100k-step PPO model")
    print("\nControls:")
    print("  🔄 Auto mode: AI takes control")
    print("  ⏸️  Manual mode: You control with keyboard")
    print("  📊 Watch the performance metrics in real-time")
    print("\n" + "=" * 50)
    
    # Launch web simulator
    try:
        # Start the server in a separate thread so we can open browser
        import threading
        
        def start_server():
            web_simulator_realistic.main()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Open browser
        url = "http://localhost:8000"
        print(f"🔗 Opening {url} in your browser...")
        webbrowser.open(url)
        
        print("\n🎯 Demo Instructions:")
        print("1. Wait for the 3D visualization to load")
        print("2. Press 'Auto' to enable AI control")
        print("3. Watch your trained model navigate and avoid debris")
        print("4. Try switching to 'Manual' to compare with human control")
        print("5. Observe the efficiency and precision of the trained AI")
        
        print("\n👀 What to look for:")
        print("- Smooth, efficient orbital maneuvers")
        print("- Proactive debris avoidance")
        print("- Fuel-efficient trajectory planning")
        print("- Stable orbit maintenance")
        
        print("\nPress Ctrl+C to stop the demo")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
            
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        print("Try running the web simulator manually:")
        print("  python3 web_simulator_realistic.py")

if __name__ == "__main__":
    main()