"""
Translation service using Hugging Face Helsinki-NLP models
Arabic -> English -> Vietnamese
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List
import torch

class TranslationService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Helsinki-NLP models are disabled - using TranslateGemma instead
        # Load Arabic to English model
        # print("Loading Arabic -> English model...")
        # self.ar_en_model_name = "Helsinki-NLP/opus-mt-ar-en"
        # self.ar_en_tokenizer = AutoTokenizer.from_pretrained(self.ar_en_model_name)
        # self.ar_en_model = AutoModelForSeq2SeqLM.from_pretrained(self.ar_en_model_name).to(self.device)
        
        # Load English to Vietnamese model
        # print("Loading English -> Vietnamese model...")
        # self.en_vi_model_name = "Helsinki-NLP/opus-mt-en-vi"
        # self.en_vi_tokenizer = AutoTokenizer.from_pretrained(self.en_vi_model_name)
        # self.en_vi_model = AutoModelForSeq2SeqLM.from_pretrained(self.en_vi_model_name).to(self.device)
        
        print("Helsinki-NLP models are disabled. Using TranslateGemma instead.")
        # print("Models loaded successfully!")
    
    def translate_batch(self, texts: List[str], tokenizer, model) -> List[str]:
        """Translate a batch of texts"""
        if not texts:
            return []
        
        # Debug: Print first few texts to check for empty strings or weird chars
        print(f"Batch sample: {texts[:3]}")
        
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return [""] * len(texts)
        
        # Tokenize
        inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translations with anti-repetition parameters
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True,
                max_length=256
            )
        
        # Decode
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        # Reconstruct with empty strings for invalid indices
        result = [""] * len(texts)
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = translated_texts[idx] if idx < len(translated_texts) else ""
        
        return result
    
    def translate_arabic_to_vietnamese(self, texts: List[str], batch_size: int = 8, progress_callback=None) -> List[str]:
        """
        Translate Arabic texts to Vietnamese via English
        
        Args:
            texts: List of Arabic texts
            batch_size: Number of texts to process at once
            progress_callback: Optional callback(current, total) for progress updates
        
        Returns:
            List of Vietnamese translations
        """
        import re
        
        total = len(texts)
        english_results = [""] * total
        vietnamese_results = [""] * total
        
        # Regex to detect Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        # Regex to detect if text has any letters (for Step 2 filtering)
        letter_pattern = re.compile(r'[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]') # Basic Latin + Vietnamese exts
        
        print(f"Starting translation of {total} texts...")
        
        # Step 1: Arabic to English (0-50%)
        # Only translate texts that actually contain Arabic characters
        print("Step 1: AR -> EN")
        for i in range(0, total, batch_size):
            batch_indices = range(i, min(i + batch_size, total))
            batch_texts = [texts[idx] for idx in batch_indices]
            
            # Identify which texts in this batch are actually Arabic
            ar_indices_in_batch = [j for j, t in enumerate(batch_texts) if arabic_pattern.search(t)]
            ar_texts_to_translate = [batch_texts[j] for j in ar_indices_in_batch]
            
            translated_batch_fragment = []
            if ar_texts_to_translate:
                try:
                    translated_batch_fragment = self.translate_batch(ar_texts_to_translate, self.ar_en_tokenizer, self.ar_en_model)
                except Exception as e:
                    print(f"Error in AR->EN batch {i}: {e}")
                    translated_batch_fragment = [""] * len(ar_texts_to_translate)
            
            # Reconstruct batch results
            # Default to original text if not translated (e.g. numbers, English already)
            processed_fragment_idx = 0
            for j, original_text in enumerate(batch_texts):
                global_idx = batch_indices[j]
                if j in ar_indices_in_batch:
                    english_results[global_idx] = translated_batch_fragment[processed_fragment_idx]
                    processed_fragment_idx += 1
                else:
                    # Pass specific original text through if it wasn't Arabic
                    english_results[global_idx] = original_text
            
            # Progress update
            processed = min(i + batch_size, total)
            progress = int((processed / total) * 50)
            if i % 100 == 0: # Reduce log spam
                print(f"AR->EN: {processed}/{total} ({progress}%)")
            if progress_callback:
                progress_callback(progress, 100)
                
        # Step 2: English to Vietnamese (50-100%)
        # Only translate texts that have letters (skip pure numbers/symbols)
        print("Step 2: EN -> VI")
        for i in range(0, total, batch_size):
            batch_indices = range(i, min(i + batch_size, total))
            batch_texts = [english_results[idx] for idx in batch_indices]
            
            # Filter: Translate if it has letters. 
            # Note: At this point, Arabic texts are now English. Original English/Numbers are still English/Numbers.
            # We want to translate English text, but skip "12345" or "12°N 45°E".
            en_indices_in_batch = [j for j, t in enumerate(batch_texts) if letter_pattern.search(t)]
            en_texts_to_translate = [batch_texts[j] for j in en_indices_in_batch]
            
            translated_batch_fragment = []
            if en_texts_to_translate:
                try:
                    translated_batch_fragment = self.translate_batch(en_texts_to_translate, self.en_vi_tokenizer, self.en_vi_model)
                except Exception as e:
                    print(f"Error in EN->VI batch {i}: {e}")
                    translated_batch_fragment = [""] * len(en_texts_to_translate)
            
            # Reconstruct
            processed_fragment_idx = 0
            for j, original_text in enumerate(batch_texts):
                global_idx = batch_indices[j]
                if j in en_indices_in_batch:
                    vietnamese_results[global_idx] = translated_batch_fragment[processed_fragment_idx]
                    processed_fragment_idx += 1
                else:
                    # Keep original (numbers/symbols)
                    vietnamese_results[global_idx] = original_text
            
            # Progress update
            processed = min(i + batch_size, total)
            progress = 50 + int((processed / total) * 50)
            if i % 100 == 0:
                print(f"EN->VI: {processed}/{total} ({progress}%)")
            if progress_callback:
                progress_callback(progress, 100)
    
        print("Translation complete!")
        return vietnamese_results

# Global instance (lazy loaded)
_translator = None

def get_translator() -> TranslationService:
    global _translator
    if _translator is None:
        _translator = TranslationService()
    return _translator
