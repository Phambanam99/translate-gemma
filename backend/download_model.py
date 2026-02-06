"""
Script to download TranslateGemma model for offline use
Run this script ONCE when you have internet connection
"""
import os
from transformers import pipeline, BitsAndBytesConfig
import torch

MODEL_NAME = "google/translategemma-4b-it"

def download_model():
    """Download and cache the model locally"""
    # Load .env if present (so HF_TOKEN can be picked up)
    try:
        from load_env import load_env
        load_env()
    except Exception:
        pass

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        print("[Auth] HF token detected in environment.")
    else:
        print("[Auth] WARNING: HF_TOKEN not found. If the repo is gated, download will fail (401).")

    print(f"Downloading {MODEL_NAME}...")
    print("This will take a while (model is ~8GB)...")
    print("Make sure you have internet connection!")
    
    # Get cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Download model files
        print("\n[1/2] Downloading model files...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False,
            token=token,
        )
        print("✓ Model files downloaded!")
        
        # Test loading the pipeline (this will cache tokenizer and other files)
        print("\n[2/2] Testing model load (this caches additional files)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            if vram_gb < 12:
                # Use 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                pipe = pipeline(
                    "image-text-to-text",
                    model=MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map="auto",
                    token=token,
                )
            else:
                pipe = pipeline(
                    "image-text-to-text",
                    model=MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    token=token,
                )
        else:
            pipe = pipeline(
                "image-text-to-text",
                model=MODEL_NAME,
                device="cpu",
                torch_dtype=torch.float32,
                token=token,
            )
        
        print("✓ Model loaded successfully!")
        print("\n" + "="*60)
        print("SUCCESS! Model is now cached locally.")
        print("You can now run the server offline.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have:")
        print("1. Internet connection")
        print("2. Enough disk space (~10GB)")
        print("3. Hugging Face account (if model requires authentication)")
        raise

if __name__ == "__main__":
    download_model()
