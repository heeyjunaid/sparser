from typing import Any, List, Dict, Tuple, Optional

import os

from sparser.tokenizer import BaseTokenizer


class BaseVectorizer:
    def __init__(self, tokenizer : BaseTokenizer, vectorizer_file_name : Optional[str], tokenizer_file_name : Optional[str]) -> None:
        if tokenizer is None:
            raise ValueError("`tokenizer` is not set")

        self.tokenizer = tokenizer
        self.tokenizer_file_name = tokenizer_file_name or "tokenizer.sps"
        self.vectorizer_file_name = vectorizer_file_name or "vectorizer.sps"

    def _save_model(self, path) -> None:
        raise NotImplementedError
    
    def _load_model(self, path) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, path:str) -> None:
        self.tokenizer.save(os.path.join(path, self.tokenizer_file_name))
        self._save_model(path)
        
    def dumps(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def train(self, texts:List[str]) -> List[Tuple[int, float]]:
        raise NotImplementedError
    
    def load(self, path:str) -> Dict[str, Any]:
        self.tokenizer.load(os.path.join(path, self.tokenizer_file_name))
        self._load_model(path)

    def _vectorize(self, tokens:List[str], **kwargs: Any) -> List[Tuple[int, float]]:
        raise NotImplementedError
    
    def __call__(self, text:str, **kwargs: Any) -> List[Tuple[int, float]]:
        tokens = self.tokenizer(text)
        return self._vectorize(tokens, **kwargs)