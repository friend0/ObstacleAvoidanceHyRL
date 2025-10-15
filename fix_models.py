#!/usr/bin/env python3
"""
Script to fix model compatibility warnings by re-saving models with current SB3 version.
"""
from pathlib import Path
from stable_baselines3 import DQN

models_dir = Path("src/HyRL/models")
model_files = [
    "dqn_obstacleavoidance",
    "dqn_obstacleavoidance_0", 
    "dqn_obstacleavoidance_1"
]

for model_name in model_files:
    model_path = models_dir / model_name
    if model_path.with_suffix('.zip').exists():
        print(f"Loading and re-saving {model_name}...")
        
        # Load the model (without specifying env, as it's stored in the model)
        model = DQN.load(str(model_path))
        
        # Save it again to fix compatibility issues
        model.save(str(model_path))
        
        print(f"✅ Fixed {model_name}")
    else:
        print(f"⚠️  Model {model_name} not found")

print("\n🎉 All models have been updated!")