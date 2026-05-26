import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizers import Tokenizer as HFTokenizer
from typing import List, Optional

# Multiprocessing worker helpers (must be top-level functions in the module)
_mp_tokenizer = None

def _mp_init(tokenizer_path: str) -> None:
    """Initializes the BPE tokenizer with cache capacity optimization in each worker process."""
    global _mp_tokenizer
    _mp_tokenizer = HFTokenizer.from_file(tokenizer_path)
    _mp_tokenizer.no_padding()
    _mp_tokenizer.no_truncation()
    if hasattr(_mp_tokenizer.model, "cache_capacity"):
        try:
            _mp_tokenizer.model.cache_capacity = 1000000
        except Exception:
            pass

def _mp_encode(chunk: List[str]) -> List[List[int]]:
    """Encodes a chunk of strings in the worker process using encode_batch_fast."""
    global _mp_tokenizer
    assert _mp_tokenizer is not None
    encodings = _mp_tokenizer.encode_batch_fast(chunk, add_special_tokens=False)
    return [enc.ids for enc in encodings]

class Tokenizer:
    def __init__(self, tokenizer_path: str = "tokenizer.json") -> None:
        """Initializes the Hugging Face BPE Tokenizer from a local configuration file."""
        current_dir: str = os.path.dirname(os.path.abspath(__file__))
        resolved_path: str = os.path.join(current_dir, tokenizer_path)
        self.tokenizer_path: str = resolved_path if os.path.exists(resolved_path) else tokenizer_path
        
        self.tokenizer: HFTokenizer = HFTokenizer.from_file(self.tokenizer_path)
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        
        # Optimize BPE cache capacity for faster tokenization
        if hasattr(self.tokenizer, "model") and hasattr(self.tokenizer.model, "cache_capacity"):
            try:
                self.tokenizer.model.cache_capacity = 1000000
            except Exception:
                pass

    def encode(self, text: str) -> List[int]:
        """Encodes a string to a list of token IDs."""
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Single-process batch encode. Best for small batches or inside worker processes."""
        encodings = self.tokenizer.encode_batch_fast(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def encode_batch_parallel(
        self,
        texts: List[str],
        num_workers: Optional[int] = None,
        add_special_tokens: bool = False
    ) -> List[List[int]]:
        """
        Multi-process batch encode. Bypasses the Python GIL completely; ideal for large datasets.
        Preserves original document order.
        """
        workers: int = num_workers or min(os.cpu_count() or 4, 8)
        chunk_size: int = max(500, math.ceil(len(texts) / workers))
        
        # Don't spawn processes for small batches
        if len(texts) <= chunk_size * 2:
            return self.encode_batch(texts, add_special_tokens=add_special_tokens)
            
        chunks: List[List[str]] = [
            texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
        ]
        
        result: List[List[List[int]]] = [[]] * len(chunks)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_mp_init,
            initargs=(self.tokenizer_path,)
        ) as pool:
            future_to_idx = {
                pool.submit(_mp_encode, c): i for i, c in enumerate(chunks)
            }
            for fut in as_completed(future_to_idx):
                result[future_to_idx[fut]] = fut.result()
                
        # Flatten and return
        return [ids for chunk_ids in result for ids in chunk_ids]

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back to a string."""
        return self.tokenizer.decode(ids)

    def token_to_id(self, token: str) -> Optional[int]:
        """Returns the ID of a specific token, or None if not found."""
        return self.tokenizer.token_to_id(token)

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.tokenizer.get_vocab_size()
