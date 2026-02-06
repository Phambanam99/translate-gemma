"""
Load environment variables from .env file
"""
import os
from pathlib import Path

def load_env():
    """Load .env file if it exists"""
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        print(f"[Config] Loading environment from {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set environment variable if not already set
                    if key and key not in os.environ:
                        os.environ[key] = value
                        print(f"[Config] Set {key}={value}")
    else:
        print(f"[Config] No .env file found at {env_file}")
        print("[Config] Using system environment variables")

if __name__ == "__main__":
    load_env()
    print("\nCurrent environment variables:")
    print(f"  HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'Not set')}")
    print(f"  TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', 'Not set')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
