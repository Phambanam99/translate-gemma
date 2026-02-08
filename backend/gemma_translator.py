"""
Translation service using Google TranslateGemma 27B from Hugging Face
Supports text translation and image OCR translation
Runs full-precision bfloat16 on A100 80GB GPU (~55GB VRAM)

Bypasses HF pipeline() to call model.generate() directly for:
- Full control over GenerationConfig (no max_length=20 conflicts)
- Adaptive max_new_tokens (short text = fewer tokens = faster)
- torch.inference_mode() for lower overhead

Performance optimizations:
- Flash Attention 2 / SDPA for faster attention computation
- CUDA TF32 + cuDNN benchmark for faster matmul
- Length-sorted batching to minimize padding waste
- Optional torch.compile() for JIT compilation (env TORCH_COMPILE=1)
"""
import base64
import math
import os
import tempfile
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

            # ── CUDA performance flags ──────────────────────────────
            # TF32: ~2x faster matmul on Ampere+ (A100, A4000, RTX 30xx)
            # with negligible precision loss for inference
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # cuDNN auto-tuner: picks fastest convolution algorithms
            torch.backends.cudnn.benchmark = True
            # Allow reduced precision reductions in bf16 matmul
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            print("[Gemma] CUDA flags: TF32=on, cuDNN benchmark=on, bf16 reduction=on")

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
    @staticmethod
    def _get_best_attn_implementation() -> str:
        """Detect the fastest available attention implementation.

        Priority: flash_attention_2 > sdpa > eager
        - flash_attention_2: ~2-3x faster, requires flash-attn package
        - sdpa: ~1.5-2x faster, built into PyTorch 2.x (no extra install)
        - eager: baseline
        """
        # 1. Try Flash Attention 2 (fastest)
        try:
            import flash_attn  # noqa: F401
            print(f"[Gemma] Flash Attention 2 available (v{flash_attn.__version__})")
            return "flash_attention_2"
        except ImportError:
            pass

        # 2. Try SDPA (built into PyTorch >= 2.0)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("[Gemma] Using SDPA (Scaled Dot-Product Attention)")
            return "sdpa"

        # 3. Fallback
        print("[Gemma] Using eager attention (consider installing flash-attn)")
        return "eager"

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

            # --- Attention implementation (flash_attention_2 > sdpa > eager) ---
            attn_impl = self._get_best_attn_implementation()

            # --- Model (full bfloat16, no quantization) ---
            model_kwargs = {**common}
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = attn_impl
            print(f"[Gemma] Loading {MODEL_NAME} in bfloat16 (attn={attn_impl})...")

            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME, **model_kwargs
                )
            except Exception as e:
                error_msg = str(e)
                # If attention implementation fails, retry without it
                if "attention" in error_msg.lower() or "flash" in error_msg.lower():
                    print(f"[Gemma] {attn_impl} failed, retrying with eager attention...")
                    model_kwargs.pop("attn_implementation", None)
                    try:
                        self._model = AutoModelForImageTextToText.from_pretrained(
                            MODEL_NAME, **model_kwargs
                        )
                    except Exception as e2:
                        print(f"[Gemma] GPU load failed: {e2}")
                        if offline_mode:
                            raise RuntimeError(
                                "Offline mode enabled but model cache incomplete."
                            ) from e2
                        print("[Gemma] Trying CPU fallback...")
                        self._model = AutoModelForImageTextToText.from_pretrained(
                            MODEL_NAME,
                            device_map="cpu",
                            torch_dtype=torch.float32,
                            **({"token": token} if token else {}),
                            local_files_only=offline_mode,
                        )
                else:
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

            # ── Optional: torch.compile for JIT speedup ──────────────
            if os.environ.get("TORCH_COMPILE", "").lower() in ("1", "true"):
                compile_mode = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")
                print(f"[Gemma] Compiling model with torch.compile(mode='{compile_mode}')...")
                print("[Gemma] First inference will be slower (JIT compilation warmup).")
                try:
                    self._model = torch.compile(
                        self._model, mode=compile_mode
                    )
                    print("[Gemma] torch.compile() applied successfully")
                except Exception as e:
                    print(f"[Gemma] torch.compile() failed (non-fatal): {e}")

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
            # Reuse cached gen config; use_cache enables KV-cache for faster
            # autoregressive decoding (avoids recomputing past key/values)
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._gen_config.pad_token_id,
                use_cache=True,
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
        self, messages: list, max_new_tokens: int
    ) -> str:
        """Generate translation from image using apply_chat_template.

        Following the official HuggingFace example:
        The image URL/path is embedded in the messages and
        apply_chat_template handles image loading + processing.
        """
        self._ensure_loaded()
        with self._gpu_lock:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._gen_config.pad_token_id,
                use_cache=True,
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
    def _estimate_max_tokens(text: str, default: int = 1024) -> int:
        """Estimate max_new_tokens based on input length.

        Translation typically produces output of similar length.
        For short CSV cells we can use much fewer tokens than 1024.
        """
        if not text:
            return 64
        # ~1 token per 3-4 chars; output ~1.5x input length; minimum 64
        estimated = max(64, int(len(text) / 3 * 1.5))
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
    # Maximum characters per chunk for splitting long text.
    # ~2000 chars ≈ ~600-700 tokens input, safe for 8K context window.
    MAX_CHUNK_CHARS = 2000

    def _split_text(self, text: str) -> List[str]:
        """Split long text into chunks, preserving paragraph/sentence boundaries.

        Strategy:
        1. Split by double newline (paragraphs) first
        2. If a paragraph is still too long, split by sentence (. ! ?)
        3. If a sentence is still too long, split by MAX_CHUNK_CHARS hard cut
        """
        import re

        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        # Step 1: Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks: List[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.MAX_CHUNK_CHARS:
                chunks.append(para)
                continue

            # Step 2: Split long paragraph by sentences
            # Handles Arabic (.)، English (.!?), and common punctuation
            sentences = re.split(r'(?<=[.!?。؟])\s+', para)
            current = ""
            for sent in sentences:
                if not sent.strip():
                    continue
                if current and len(current) + len(sent) + 1 > self.MAX_CHUNK_CHARS:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent

                # Step 3: Hard split if single sentence is too long
                while len(current) > self.MAX_CHUNK_CHARS:
                    chunks.append(current[:self.MAX_CHUNK_CHARS])
                    current = current[self.MAX_CHUNK_CHARS:]

            if current.strip():
                chunks.append(current.strip())

        return chunks if chunks else [text]

    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "vi",
        max_new_tokens: int = 1024,
    ) -> str:
        """Translate a single text string. Auto-splits long text into chunks."""
        if not text or not text.strip():
            return ""

        chunks = self._split_text(text.strip())

        if len(chunks) == 1:
            return self._translate_single_chunk(
                chunks[0], source_lang, target_lang, max_new_tokens
            )

        # Translate each chunk and join with double newline
        print(f"[Gemma] Long text split into {len(chunks)} chunks")
        translated_parts = []
        for i, chunk in enumerate(chunks):
            print(f"[Gemma] Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            part = self._translate_single_chunk(
                chunk, source_lang, target_lang, max_new_tokens
            )
            translated_parts.append(part)

        return "\n\n".join(translated_parts)

    def _translate_single_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """Translate a single chunk that fits within model context."""
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
        max_new_tokens: int = 1024,
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

        # ── Length-sorted batching ──────────────────────────────────
        # Sort texts by character length so similar-length texts are
        # batched together.  This minimises padding waste in the
        # tokenised batch (all seqs padded to the longest in batch).
        sorted_indices = sorted(
            range(unique_total), key=lambda i: len(unique_texts[i])
        )
        sorted_unique = [unique_texts[i] for i in sorted_indices]

        print(
            f"[Gemma] Batch: {total} rows, {unique_total} unique, "
            f"batch_size={batch_size}, length-sorted=yes"
        )

        # Process in chunks of batch_size (length-sorted)
        for start in range(0, unique_total, batch_size):
            chunk_sorted_idx = sorted_indices[start : start + batch_size]
            chunk_texts = sorted_unique[start : start + batch_size]
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

            # Write results back (map through sorted indices)
            for j, translated_text in enumerate(translated_chunk):
                orig_idx = chunk_sorted_idx[j]
                orig_text = unique_texts[orig_idx]
                for pos in text_to_positions[orig_text]:
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
        max_new_tokens: int = 1024,
        max_image_side: int = 2048,
        jpeg_quality: int = 95,
    ) -> str:
        """Extract and translate text from an image.

        Following the official TranslateGemma usage: the image URL/path
        is passed inside the message content and apply_chat_template
        handles all image loading and processing.
        """

        def _resize_and_save(image_bytes: bytes, path: str) -> None:
            """Resize image and save to a temporary file."""
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
                im.save(path, format="JPEG", quality=jpeg_quality, optimize=True)

        tmp_path = None
        try:
            # Download / decode image, resize, save to temp file
            if image_source.startswith(("http://", "https://")):
                resp = requests.get(image_source, timeout=30)
                resp.raise_for_status()
                raw_bytes = resp.content
            else:
                raw_bytes = base64.b64decode(image_source)

            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            _resize_and_save(raw_bytes, tmp_path)

            # Official TranslateGemma message format with image URL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "url": tmp_path,
                        },
                    ],
                }
            ]

            return self._generate_from_image(
                messages, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"[Gemma] Image translation error: {e}")
            self._reset_cuda()
            return f"[Error: {e}]"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


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
