#!/usr/bin/env python3
"""
Demo: Best Model Only Training

This script demonstrates how to train a model that only saves the best-performing
model based on evaluation performance, preventing disk clutter from multiple
checkpoint files.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_best_model_training():
    """Demonstrate training with best model only saving."""
    from rl_train import train_agent
    
    # Configuration for best model only training
    config = {
        'algorithm': 'ppo',
        'scenario_file': 'scenarios/leo_debris_avoidance.json',
        'reward_function': 'balanced',
        'training': {
            'total_timesteps': 50000,  # Short training for demo
            'checkpoint_freq': 10000,  # Ignored when save_best_only=True
            'eval_freq': 2000,        # Evaluate every 2000 steps
            'reward_threshold': None,
            'save_best_only': True    # Only save best model
        },
        'environment': {
            'max_debris_tracked': 5,  # Smaller environment for faster training
            'render_mode': None
        },
        'output': {
            'log_dir': './demo_logs',
            'save_dir': './demo_models',
            'experiment_name': 'best_model_demo'
        }
    }
    
    logger.info("🚀 Starting Best Model Only Training Demo")
    logger.info("=" * 50)
    logger.info("Configuration:")
    logger.info(f"  - Algorithm: {config['algorithm'].upper()}")
    logger.info(f"  - Total timesteps: {config['training']['total_timesteps']:,}")
    logger.info(f"  - Evaluation frequency: {config['training']['eval_freq']} steps")
    logger.info(f"  - Save mode: Best model only")
    logger.info(f"  - Save location: {config['output']['save_dir']}/{config['output']['experiment_name']}")
    
    try:
        # Run training
        training_info = train_agent(config)
        
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 50)
        
        # Check saved models
        import os
        from pathlib import Path
        
        save_dir = Path(config['output']['save_dir']) / config['output']['experiment_name']
        saved_files = list(save_dir.glob("*.zip"))
        
        logger.info(f"📁 Saved models in {save_dir}:")
        if saved_files:
            for model_file in saved_files:
                file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  - {model_file.name} ({file_size:.1f} MB)")
        else:
            logger.warning("  No model files found!")
        
        # Expected: only best_model.zip should be present
        best_model_path = save_dir / "best_model.zip"
        if best_model_path.exists():
            logger.info("✅ Best model saved successfully!")
            logger.info(f"📊 Best model location: {best_model_path}")
        else:
            logger.warning("⚠️  best_model.zip not found")
        
        # Check for unwanted checkpoint files
        checkpoint_files = list(save_dir.glob("model_checkpoint_*.zip"))
        if checkpoint_files:
            logger.warning(f"⚠️  Found {len(checkpoint_files)} unexpected checkpoint files:")
            for file in checkpoint_files:
                logger.warning(f"    - {file.name}")
        else:
            logger.info("✅ No unwanted checkpoint files (as expected)")
        
        logger.info("=" * 50)
        logger.info("🎯 Demo Summary:")
        logger.info("   - Training completed with best-model-only saving")
        logger.info("   - Only the highest-performing model is retained")
        logger.info("   - Disk space is optimized by avoiding checkpoint clutter")
        logger.info("   - Model selection is automated based on evaluation performance")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def demo_comparison():
    """Show the difference between best-only and checkpoint saving."""
    logger.info("📊 Saving Strategy Comparison:")
    logger.info("=" * 50)
    
    logger.info("🎯 Best Model Only (save_best_only=True):")
    logger.info("   ✅ Saves only the highest-performing model")
    logger.info("   ✅ Minimal disk usage")
    logger.info("   ✅ Automatic model selection")
    logger.info("   ✅ No manual model comparison needed")
    logger.info("   📁 Output: best_model.zip")
    
    logger.info("\n📚 All Checkpoints (save_best_only=False):")
    logger.info("   ✅ Saves models at regular intervals")
    logger.info("   ✅ Useful for training analysis")
    logger.info("   ❌ High disk usage")
    logger.info("   ❌ Manual model selection required")
    logger.info("   📁 Output: model_checkpoint_10000_steps.zip, model_checkpoint_20000_steps.zip, ...")
    
    logger.info("\n💡 Recommendation:")
    logger.info("   Use save_best_only=True for production training")
    logger.info("   Use save_best_only=False only for research/debugging")

if __name__ == "__main__":
    try:
        demo_comparison()
        print("\n" + "="*50)
        print("Starting training demo...")
        print("="*50)
        demo_best_model_training()
        
    except KeyboardInterrupt:
        logger.info("\n❌ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()