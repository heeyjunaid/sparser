import regex
from typing import List, Set, Optional

from sparser.tokenizer.base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, ignore_words:Optional[Set[str]]) -> None:
        super().__init__()

        self.ignore_words = ignore_words
        # TODO: Fix tokenization for didn't -> (didn, ', t) should be (did, n't) 
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def _tokenize(self, text: str, use_lower: bool=True, ignore_special_char: bool=True) -> List[str]:
        if use_lower:
            text = text.lower()
        
        tokens = []
        matches = [m for m in self._regexp.finditer(text)]

        for i in range(len(matches)):
            token = matches[i].group()

            if token not in self.ignore_words and len(token)>1:
                if not ignore_special_char: 
                    tokens.append(token)
                elif token.isalnum():
                    tokens.append(token)
                
        return tokens

        

