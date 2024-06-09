from typing import Optional

from sparser.tokenizer.base import BaseTokenizer

class HFTokenizer(BaseTokenizer):
    def __init__(self, model_id:Optional[str] = None) -> None:
        """ Directly use HF tokenizer
        """
        super().__init__()
        self.model_id = model_id or "naver/splade-cocondenser-ensembledistil"
        raise NotImplementedError