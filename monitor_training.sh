#!/bin/bash
# Advanced Research Training Monitor
# Use these commands to monitor your training progress

echo "🚀 Advanced Research Dense LEO Training Monitor"
echo "================================================"

# Check if training process is running
echo "📊 Training Process Status:"
if ps aux | grep "rl_train.py" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    ps aux | grep "rl_train.py" | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
else
    echo "❌ Training is NOT running"
fi

echo ""

# Check GPU utilization
echo "🎮 GPU Utilization (GPUs 3&5):"
export CUDA_VISIBLE_DEVICES=3,5
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   GPU $line"
    done
else
    echo "   nvidia-smi not available"
fi

echo ""

# Show recent training logs
echo "📜 Recent Training Logs (last 10 lines):"
if [ -f "advanced_training.log" ]; then
    tail -10 advanced_training.log | sed 's/^/   /'
else
    echo "   No log file found"
fi

echo ""

# Check training results directory
echo "💾 Training Results:"
if [ -d "advanced_research_results" ]; then
    echo "   Results directory exists:"
    ls -la advanced_research_results/ | tail -5 | sed 's/^/     /'
else
    echo "   No results directory yet"
fi

echo ""

# Check for TensorBoard logs
echo "📈 TensorBoard Logs:"
if [ -d "training_logs" ]; then
    echo "   TensorBoard logs available:"
    find training_logs -name "*.tfevents.*" 2>/dev/null | tail -3 | sed 's/^/     /'
    echo "   Run: tensorboard --logdir ./training_logs/"
else
    echo "   No TensorBoard logs yet"
fi

echo ""
echo "🔄 Refresh: Run this script again to update status"
echo "⏹️  Stop training: pkill -f rl_train.py"
echo "📊 Full log: tail -f advanced_training.log"