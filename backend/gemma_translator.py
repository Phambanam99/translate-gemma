"""
Translation service using Google TranslateGemma from Hugging Face
Supports text translation and image OCR translation
Uses 4-bit quantization to fit in 8GB GPU (RTX 3060 Ti etc.)
"""
import base64
import threading
from io import BytesIO
from typing import List

import requests
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline

# Model selection
MODEL_NAME = "google/translategemma-4b-it"

# Language codes supported by TranslateGemma
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "vi": "Vietnamese", 
    "en": "English",
    "de-DE": "German",
    "cs": "Czech",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "th": "Thai",
}

class GemmaTranslationService:
    def __init__(self):
        torch_version = torch.__version__
        print(f"[Gemma] PyTorch version: {torch_version}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Gemma] Using device: {self.device}")
        
        if self.device == "cuda":
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Gemma] GPU: {gpu_name} ({vram_gb} GB VRAM)")
            
            # Determine quantization based on VRAM
            # NVIDIA A4000 has 16GB VRAM - perfect for float16 without quantization
            if vram_gb < 12:
                print(f"[Gemma] VRAM < 12GB -> Using 4-bit quantization (NF4)")
                self._use_quantization = "4bit"
            elif vram_gb < 16:
                print(f"[Gemma] VRAM < 16GB -> Using 8-bit quantization")
                self._use_quantization = "8bit"
            else:
                print(f"[Gemma] VRAM >= 16GB -> Using float16 (no quantization, optimal performance)")
                self._use_quantization = None
        else:
            self._use_quantization = None
            
        print(f"[Gemma] Model: {MODEL_NAME}")
        self._pipe = None
        self._pipeline_init_lock = threading.Lock()
        # Single GPU lock for all requests (CSV/text/image)
        self._gpu_lock = threading.Lock()

    def _get_pipeline(self):
        """Lazy load pipeline with appropriate quantization for GPU VRAM"""
        if self._pipe is None:
            with self._pipeline_init_lock:
                if self._pipe is not None:
                    return self._pipe

                print(f"[Gemma] Loading {MODEL_NAME}...")
                print("[Gemma] This may take a few minutes on first load...")

                # Check for offline mode
                import os

                offline_mode = os.environ.get("HF_HUB_OFFLINE", "false").lower() == "true" or \
                              os.environ.get("TRANSFORMERS_OFFLINE", "false").lower() == "true"

                if offline_mode:
                    print("[Gemma] Offline mode detected - using local cache only")

                # Get token from environment
                token = os.environ.get("HF_TOKEN")
                if not token:
                    print("[Gemma] Warning: HF_TOKEN not found in environment. Model download may fail if gated.")

                model_kwargs = {}
                if token:
                    model_kwargs["token"] = token

                if offline_mode:
                    model_kwargs["local_files_only"] = True
                    # Also prevent any hub access during pipeline component resolution
                    model_kwargs["proxies"] = None

                if self._use_quantization == "4bit":
                    # 4-bit quantization: ~2.5GB VRAM for 4B model
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    print("[Gemma] Loading with 4-bit NF4 quantization...")

                elif self._use_quantization == "8bit":
                    # 8-bit quantization: ~4GB VRAM for 4B model
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    print("[Gemma] Loading with 8-bit quantization...")

                else:
                    # No quantization: ~8GB VRAM for 4B model in float16
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["dtype"] = torch.float16
                    print("[Gemma] Loading with float16 (no quantization)...")

                try:
                    self._pipe = pipeline(
                        "image-text-to-text",
                        model=MODEL_NAME,
                        **model_kwargs,
                    )
                    print("[Gemma] Model loaded successfully!")
                except Exception as e:
                    print(f"[Gemma] Failed to load model: {e}")
                    # If offline or gated, CPU fallback will still fail if cache is incomplete.
                    # In that case, raise a clear error so the API can report it.
                    if offline_mode:
                        raise RuntimeError(
                            "Offline mode is enabled but the model cache is incomplete (or the repo is gated). "
                            "You must download ALL model files while authenticated, then run offline with local cache."
                        ) from e

                    print("[Gemma] Trying CPU fallback...")
                    # Fallback to CPU
                    self._pipe = pipeline(
                        "image-text-to-text",
                        model=MODEL_NAME,
                        device="cpu",
                        dtype=torch.float32,
                        token=token,
                        local_files_only=offline_mode,
                    )
                    print("[Gemma] Model loaded on CPU (slower but works)")

        return self._pipe

    def _run_pipe(self, messages, max_new_tokens: int):
        """Run model inference through a single global lock for 1-GPU stability."""
        pipe = self._get_pipeline()
        with self._gpu_lock:
            return pipe(text=messages, max_new_tokens=max_new_tokens)

    @staticmethod
    def _extract_generated_content(output_item: dict) -> str:
        generated_text = output_item.get("generated_text")
        if isinstance(generated_text, list) and generated_text:
            last_message = generated_text[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", ""))
            return str(last_message)
        if isinstance(generated_text, str):
            return generated_text
        return ""

    @staticmethod
    def _build_text_message(text: str, source_lang: str, target_lang: str):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]

    def _reset_cuda(self):
        """Reset CUDA state after error"""
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
    
    def translate_text(
        self, 
        text: str, 
        source_lang: str = "ar", 
        target_lang: str = "vi",
        max_new_tokens: int = 256
    ) -> str:
        """Translate text from source language to target language"""
        if not text or not text.strip():
            return ""

        messages = self._build_text_message(text, source_lang, target_lang)

        try:
            output = self._run_pipe(messages, max_new_tokens=max_new_tokens)
            if isinstance(output, dict):
                output = [output]
            if not output:
                return ""
            return self._extract_generated_content(output[0])
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[Gemma] CUDA/Runtime error: {error_msg}")
            self._reset_cuda()
            
            # Try again with smaller max_new_tokens
            if "CUDA" in error_msg or "out of memory" in error_msg.lower():
                try:
                    print("[Gemma] Retrying with reduced tokens...")
                    output = self._run_pipe(messages, max_new_tokens=min(max_new_tokens, 64))
                    return self._extract_generated_content(output[0])
                except Exception:
                    self._reset_cuda()
                    return f"[Error: GPU memory insufficient. Text: {text[:50]}...]"
            
            return f"[Error: {error_msg}]"
        except Exception as e:
            error_msg = str(e)
            print(f"[Gemma] Translation error: {error_msg}")
            self._reset_cuda()
            return f"[Error: {error_msg}]"
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "vi",
        progress_callback=None,
        batch_size: int = 4,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """Translate texts using micro-batching with safe per-item fallback."""
        total = len(texts)
        if total == 0:
            return []

        batch_size = max(1, int(batch_size))
        results = [""] * total

        text_to_positions = {}
        for idx, text in enumerate(texts):
            if not text or not str(text).strip():
                continue
            text_to_positions.setdefault(text, []).append(idx)

        if not text_to_positions:
            if progress_callback:
                progress_callback(100, 100)
            return results

        unique_texts = list(text_to_positions.keys())
        processed_items = 0

        for start in range(0, len(unique_texts), batch_size):
            batch_texts = unique_texts[start : start + batch_size]
            batch_messages = [
                self._build_text_message(text, source_lang, target_lang)
                for text in batch_texts
            ]

            translated_batch: List[str] = []
            try:
                batch_output = self._run_pipe(batch_messages, max_new_tokens=max_new_tokens)
                if isinstance(batch_output, dict):
                    batch_output = [batch_output]
                translated_batch = [
                    self._extract_generated_content(output_item)
                    for output_item in batch_output
                ]
                if len(translated_batch) != len(batch_texts):
                    raise RuntimeError("Unexpected batch output size")
            except Exception as batch_error:
                print(f"[Gemma] Batch inference failed, fallback to per-item mode: {batch_error}")
                translated_batch = [
                    self.translate_text(
                        text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_new_tokens=max_new_tokens,
                    )
                    for text in batch_texts
                ]

            for text, translated_text in zip(batch_texts, translated_batch):
                positions = text_to_positions[text]
                for position in positions:
                    results[position] = translated_text
                processed_items += len(positions)

            if progress_callback:
                progress = int((processed_items / total) * 100)
                progress_callback(progress, 100)

            if processed_items % 10 == 0 or processed_items == total:
                print(f"[Gemma] Translated {processed_items}/{total}")

        return results
    
    def translate_image(
        self,
        image_source: str,
        source_lang: str = "ar",
        target_lang: str = "vi",
        max_new_tokens: int = 256,
        max_image_side: int = 1024,
        jpeg_quality: int = 90,
    ) -> str:
        """Extract and translate text from an image"""
        def _resize_image_bytes(image_bytes: bytes) -> bytes:
            """
            Resize image keeping aspect ratio so that max(width, height) <= max_image_side.
            Returns JPEG bytes.
            """
            with Image.open(BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                w, h = im.size
                if max(w, h) <= max_image_side:
                    out = BytesIO()
                    im.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
                    return out.getvalue()

                if w >= h:
                    new_w = max_image_side
                    new_h = int(h * (max_image_side / w))
                else:
                    new_h = max_image_side
                    new_w = int(w * (max_image_side / h))

                im = im.resize((new_w, new_h), Image.BICUBIC)
                out = BytesIO()
                im.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
                return out.getvalue()

        # Normalize input into a resized data URL when possible (faster + more stable)
        try:
            if image_source.startswith(("http://", "https://")):
                # Download then resize (may require internet for remote URLs)
                resp = requests.get(image_source, timeout=30)
                resp.raise_for_status()
                resized_bytes = _resize_image_bytes(resp.content)
                b64 = base64.b64encode(resized_bytes).decode("utf-8")
                image_data = {"url": f"data:image/jpeg;base64,{b64}"}
            else:
                # Base64 -> bytes -> resize -> base64
                raw_bytes = base64.b64decode(image_source)
                resized_bytes = _resize_image_bytes(raw_bytes)
                b64 = base64.b64encode(resized_bytes).decode("utf-8")
                image_data = {"url": f"data:image/jpeg;base64,{b64}"}
        except Exception as e:
            # If resizing fails, fallback to original input
            print(f"[Gemma] Warning: image resize failed, using original input. Error: {e}")
            if image_source.startswith(("http://", "https://")):
                image_data = {"url": image_source}
            else:
                image_data = {"url": f"data:image/jpeg;base64,{image_source}"}
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        **image_data
                    },
                ],
            }
        ]
        
        try:
            output = self._run_pipe(messages, max_new_tokens=max_new_tokens)
            if isinstance(output, dict):
                output = [output]
            if not output:
                return ""
            return self._extract_generated_content(output[0])
        except Exception as e:
            error_msg = str(e)
            print(f"[Gemma] Image translation error: {error_msg}")
            self._reset_cuda()
            return f"[Error: {error_msg}]"

# Global instance (lazy loaded)
_gemma_translator = None

def get_gemma_translator() -> GemmaTranslationService:
    global _gemma_translator
    if _gemma_translator is None:
        _gemma_translator = GemmaTranslationService()
    return _gemma_translator

def get_supported_languages():
    """Return dict of supported language codes and names"""
    return SUPPORTED_LANGUAGES
