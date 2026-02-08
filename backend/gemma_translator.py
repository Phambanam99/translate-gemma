"""
Translation service using Google TranslateGemma 27B from Hugging Face
Supports text translation and image OCR translation
Runs full-precision bfloat16 on A100 80GB GPU (~55GB VRAM)

Bypasses HF pipeline() to call model.generate() directly for:
- Full control over GenerationConfig (no max_length=20 conflicts)
- Adaptive max_new_tokens (short text = fewer tokens = faster)
- torch.inference_mode() for lower overhead
"""
import base64
import math
import os
import threading
import time
from io import BytesIO
from typing import List, Optional

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

# Model selection — 27B full precision (bfloat16) on A100 80GB
MODEL_NAME = os.environ.get("MODEL_NAME", "google/translategemma-27b-it")

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

        # Get GPU device ID from environment variable (default: 0)
        try:
            self.gpu_id = int(os.environ.get("GPU_DEVICE_ID", "0"))
        except (ValueError, TypeError):
            self.gpu_id = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Gemma] Using device: {self.device}")

        if self.device == "cuda":
            torch.cuda.set_device(self.gpu_id)
            vram_gb = round(
                torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3, 1
            )
            gpu_name = torch.cuda.get_device_name(self.gpu_id)
            print(f"[Gemma] GPU {self.gpu_id}: {gpu_name} ({vram_gb} GB VRAM)")
            print(f"[Gemma] Will load {MODEL_NAME} in bfloat16 (no quantization)")

        print(f"[Gemma] Model: {MODEL_NAME}")
        self._model: Optional[AutoModelForImageTextToText] = None
        self._processor: Optional[AutoProcessor] = None
        self._gen_config: Optional[GenerationConfig] = None
        self._init_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load model + processor directly (no pipeline wrapper)."""
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return

            print(f"[Gemma] Loading {MODEL_NAME}...")
            print("[Gemma] This may take a few minutes on first load...")

            offline_mode = (
                os.environ.get("HF_HUB_OFFLINE", "false").lower() == "true"
                or os.environ.get("TRANSFORMERS_OFFLINE", "false").lower() == "true"
            )
            if offline_mode:
                print("[Gemma] Offline mode detected - using local cache only")

            token = os.environ.get("HF_TOKEN")
            if not token:
                print(
                    "[Gemma] Warning: HF_TOKEN not found. "
                    "Model download may fail if gated."
                )

            common = {}
            if token:
                common["token"] = token
            if offline_mode:
                common["local_files_only"] = True

            # --- Processor ---
            self._processor = AutoProcessor.from_pretrained(
                MODEL_NAME, **common
            )

            # --- Model (full bfloat16, no quantization) ---
            model_kwargs = {**common}
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16
            print(f"[Gemma] Loading {MODEL_NAME} in bfloat16 (full precision)...")

            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME, **model_kwargs
                )
            except Exception as e:
                print(f"[Gemma] GPU load failed: {e}")
                if offline_mode:
                    raise RuntimeError(
                        "Offline mode enabled but model cache incomplete."
                    ) from e
                print("[Gemma] Trying CPU fallback...")
                self._model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    **({"token": token} if token else {}),
                    local_files_only=offline_mode,
                )

            # Build a clean GenerationConfig template (no max_length!)
            eos = getattr(self._model.config, "eos_token_id", 1)
            if not isinstance(eos, int):
                eos = eos[0] if isinstance(eos, (list, tuple)) and eos else 1
            self._gen_config = GenerationConfig(
                do_sample=False,
                pad_token_id=eos,
            )

            print("[Gemma] Model loaded successfully!")

    def _ensure_loaded(self):
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    # Core inference  (NO pipeline, direct model.generate)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_text(self, messages: list, max_new_tokens: int) -> str:
        """Translate a single conversation (single text)."""
        results = self._generate_text_batch([messages], max_new_tokens)
        return results[0]

    @torch.inference_mode()
    def _generate_text_batch(
        self, messages_list: List[list], max_new_tokens: int
    ) -> List[str]:
        """Translate multiple texts in ONE GPU forward pass (true batching).

        1. apply_chat_template on each conversation -> list of prompt strings
        2. Batch-tokenize with left-padding (required for generate())
        3. Single model.generate() call -> all outputs at once
        4. Decode each output, stripping input tokens
        """
        self._ensure_loaded()
        with self._gpu_lock:
            # Step 1: Get prompt strings via chat template
            prompts: List[str] = []
            for messages in messages_list:
                prompt_str = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,  # return string, not tokens
                )
                prompts.append(prompt_str)

            tokenizer = self._processor.tokenizer

            # Step 2: Batch-tokenize with LEFT padding (needed for generate)
            original_side = tokenizer.padding_side
            original_pad = tokenizer.pad_token_id
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = self._gen_config.pad_token_id

            try:
                batch_enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
            finally:
                tokenizer.padding_side = original_side
                tokenizer.pad_token_id = original_pad

            # Move to model device
            batch_enc = {
                k: v.to(self._model.device) if hasattr(v, "to") else v
                for k, v in batch_enc.items()
            }

            # Track each sequence's real (non-padded) length
            # With left-padding, real tokens are at the RIGHT end.
            attention_mask = batch_enc["attention_mask"]
            input_lengths = attention_mask.sum(dim=1).tolist()

            # Step 3: Single generate() call for the whole batch
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._gen_config.pad_token_id,
            )

            output_ids = self._model.generate(
                **batch_enc, generation_config=gen_config
            )

            # Step 4: Decode each output, slicing off input tokens
            total_len = batch_enc["input_ids"].shape[-1]  # padded length
            results: List[str] = []
            for i in range(len(prompts)):
                # All sequences were padded to total_len.  Generated tokens
                # start right after total_len in the output.
                new_ids = output_ids[i, total_len:]
                text = self._processor.decode(
                    new_ids, skip_special_tokens=True
                )
                results.append(text)

            return results

    @torch.inference_mode()
    def _generate_from_image(
        self, messages: list, images: list, max_new_tokens: int
    ) -> str:
        """Same as _generate_text but with image inputs."""
        self._ensure_loaded()
        with self._gpu_lock:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Process images
            image_inputs = self._processor.image_processor(
                images=images, return_tensors="pt"
            )
            inputs.update(image_inputs)

            inputs = {
                k: v.to(self._model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            input_len = inputs["input_ids"].shape[-1]

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._gen_config.pad_token_id,
            )

            output_ids = self._model.generate(**inputs, generation_config=gen_config)
            new_ids = output_ids[0, input_len:]
            return self._processor.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    @staticmethod
    def _estimate_max_tokens(text: str, default: int = 256) -> int:
        """Estimate max_new_tokens based on input length.

        Translation typically produces output of similar length.
        For short CSV cells we can use much fewer tokens than 256.
        """
        if not text:
            return 32
        # ~1 token per 3-4 chars; output ~1.5x input length; minimum 32
        estimated = max(32, int(len(text) / 3 * 1.5))
        return min(estimated, default)

    def _reset_cuda(self):
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "vi",
        max_new_tokens: int = 256,
    ) -> str:
        """Translate a single text string."""
        if not text or not text.strip():
            return ""

        adaptive_tokens = self._estimate_max_tokens(text, max_new_tokens)
        messages = self._build_text_message(text, source_lang, target_lang)

        try:
            t0 = time.perf_counter()
            result = self._generate_text(messages, max_new_tokens=adaptive_tokens)
            elapsed = time.perf_counter() - t0
            print(
                f"[Gemma] translate: {elapsed:.2f}s  "
                f"max_tok={adaptive_tokens}  in={len(text)} chars"
            )
            return result
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[Gemma] CUDA/Runtime error: {error_msg}")
            self._reset_cuda()
            if "CUDA" in error_msg or "out of memory" in error_msg.lower():
                try:
                    print("[Gemma] Retrying with reduced tokens...")
                    return self._generate_text(
                        messages, max_new_tokens=min(adaptive_tokens, 64)
                    )
                except Exception:
                    self._reset_cuda()
                    return f"[Error: GPU memory insufficient. Text: {text[:50]}...]"
            return f"[Error: {error_msg}]"
        except Exception as e:
            print(f"[Gemma] Translation error: {e}")
            self._reset_cuda()
            return f"[Error: {e}]"

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "vi",
        progress_callback=None,
        batch_size: int = 20,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """Translate list of texts with TRUE GPU batching.

        - Deduplicates identical texts
        - Groups unique texts into batches of `batch_size`
        - Each batch = 1 GPU forward pass (e.g. 4 texts at once)
        - Falls back to per-item mode on OOM
        """
        total = len(texts)
        if total == 0:
            return []

        batch_size = max(1, int(batch_size))

        def _percent(done: int, overall: int) -> int:
            if overall <= 0:
                return 100
            if done >= overall:
                return 100
            return max(1, min(99, math.ceil((done * 100) / overall)))

        results = [""] * total

        # Deduplicate
        text_to_positions = {}
        for idx, text in enumerate(texts):
            if not text or not str(text).strip():
                continue
            text_to_positions.setdefault(text, []).append(idx)

        if not text_to_positions:
            if progress_callback:
                progress_callback(100, total)
            return results

        unique_texts = list(text_to_positions.keys())
        unique_total = len(unique_texts)
        processed_items = 0
        batch_start_time = time.perf_counter()

        if progress_callback:
            progress_callback(0, 0)

        print(
            f"[Gemma] Batch: {total} rows, {unique_total} unique, "
            f"batch_size={batch_size}"
        )

        # Process in chunks of batch_size
        for start in range(0, unique_total, batch_size):
            chunk_texts = unique_texts[start : start + batch_size]
            chunk_messages = [
                self._build_text_message(t, source_lang, target_lang)
                for t in chunk_texts
            ]
            # Use max adaptive tokens across the chunk (generate uses one limit)
            chunk_max_tokens = max(
                self._estimate_max_tokens(t, max_new_tokens) for t in chunk_texts
            )

            translated_chunk: List[str] = []
            try:
                t0 = time.perf_counter()
                translated_chunk = self._generate_text_batch(
                    chunk_messages, max_new_tokens=chunk_max_tokens
                )
                elapsed = time.perf_counter() - t0
                print(
                    f"[Gemma] batch[{start+1}-{start+len(chunk_texts)}/{unique_total}] "
                    f"{elapsed:.2f}s  x{len(chunk_texts)} texts  "
                    f"max_tok={chunk_max_tokens}"
                )
            except RuntimeError as e:
                error_msg = str(e)
                print(
                    f"[Gemma] Batch GPU error: {error_msg} "
                    f"-> falling back to per-item"
                )
                self._reset_cuda()
                # Fallback: translate one by one
                for text in chunk_texts:
                    try:
                        adaptive = self._estimate_max_tokens(text, max_new_tokens)
                        msgs = self._build_text_message(
                            text, source_lang, target_lang
                        )
                        result = self._generate_text(msgs, max_new_tokens=adaptive)
                        translated_chunk.append(result)
                    except Exception:
                        self._reset_cuda()
                        translated_chunk.append("[Error: GPU memory insufficient]")
            except Exception as e:
                print(f"[Gemma] Batch error: {e} -> falling back to per-item")
                self._reset_cuda()
                for text in chunk_texts:
                    try:
                        adaptive = self._estimate_max_tokens(text, max_new_tokens)
                        msgs = self._build_text_message(
                            text, source_lang, target_lang
                        )
                        result = self._generate_text(msgs, max_new_tokens=adaptive)
                        translated_chunk.append(result)
                    except Exception:
                        self._reset_cuda()
                        translated_chunk.append(f"[Error: {e}]")

            # Write results back
            for text, translated_text in zip(chunk_texts, translated_chunk):
                for pos in text_to_positions[text]:
                    results[pos] = translated_text
                    processed_items += 1
                    if progress_callback:
                        progress_callback(
                            _percent(processed_items, total), processed_items
                        )

        total_elapsed = time.perf_counter() - batch_start_time
        avg = total_elapsed / unique_total if unique_total else 0
        print(
            f"[Gemma] Done: {unique_total} unique in {total_elapsed:.1f}s "
            f"(avg {avg:.2f}s/text, batch_size={batch_size})"
        )
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
        """Extract and translate text from an image."""

        def _resize(image_bytes: bytes) -> bytes:
            with Image.open(BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                w, h = im.size
                if max(w, h) > max_image_side:
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

        # Load + resize image
        try:
            if image_source.startswith(("http://", "https://")):
                resp = requests.get(image_source, timeout=30)
                resp.raise_for_status()
                raw_bytes = resp.content
            else:
                raw_bytes = base64.b64decode(image_source)
            resized = _resize(raw_bytes)
            pil_image = Image.open(BytesIO(resized)).convert("RGB")
        except Exception as e:
            print(f"[Gemma] Image load/resize failed: {e}")
            return f"[Error: could not load image: {e}]"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                    },
                ],
            }
        ]

        try:
            return self._generate_from_image(
                messages, [pil_image], max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"[Gemma] Image translation error: {e}")
            self._reset_cuda()
            return f"[Error: {e}]"


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_gemma_translator = None


def get_gemma_translator() -> GemmaTranslationService:
    global _gemma_translator
    if _gemma_translator is None:
        _gemma_translator = GemmaTranslationService()
    return _gemma_translator


def get_supported_languages():
    """Return dict of supported language codes and names"""
    return SUPPORTED_LANGUAGES
