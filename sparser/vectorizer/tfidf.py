import numpy as np
from math import log1p
from typing import List, Tuple, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from sparser.tokenizer.base import BaseTokenizer
from sparser.vectorizer.base import BaseVectorizer


class TFIDFVectorizer(BaseVectorizer):
    def __init__(self, tokenizer: BaseTokenizer, 
                 max_features : int,
                 vectorizer_file_name: str | None, 
                 tokenizer_file_name: str | None,
                 use_lowercase : bool =True
                 ) -> None:
        super().__init__(tokenizer, vectorizer_file_name, tokenizer_file_name)

        self.max_features = max_features
        self.use_lowercase = use_lowercase
        self.tfidf = TfidfVectorizer(lowercase=use_lowercase, tokenizer=lambda x: x, preprocessor=lambda x: x, max_features=max_features)
        self.key_idf_map = {}

    def _tokenize(self, texts:List[str]) -> List[List[str]]:
        tokenized_texts = []
        for t in texts:
            tokenized_texts.append(self.tokenizer(t))
        return tokenized_texts
    
    def train(self, texts: List[str]) -> List[Tuple[int | float]]:
        tokenized_texts = self._tokenize(texts)
        self.tfidf.fit_transform(tokenized_texts)

        keys = self.tfidf.vocabulary_
        idf_vals = self.tfidf.idf_

        for key, idx in keys.items():
            self.key_idf_map[key] = {"idf":idf_vals[idx].item(), "idx":int(idx)}

        return self.key_idf_map
    
    def _calculate_tfidf_score(self, tf:int, idf:float) -> float :
        # TODO: verify this logic with sklearn TfidfVectorizer 
        # tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        return log1p(tf+1)*idf

    def _vectorize(self, tokens: List[str], **kwargs: Any) -> List[Tuple[int, float]]:
        tokens = [t for t in tokens if t in self.key_idf_map]
        key_score = [0]*len(self.key_idf_map)

        counts = defaultdict(int)
        for t in tokens:
            counts[t] += 1

        for key, freq in counts.items():
            idx_ = self.key_idf_map[key]["idx"]
            idf_ = self.key_idf_map[key]["idf"]
            key_score[idx_] = self._calculate_tfidf_score(freq, idf_)

        return key_score
