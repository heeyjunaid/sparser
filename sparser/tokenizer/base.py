from typing import List, Optional, Any, Dict


class BaseTokenizer:
    def __init__(self) -> None:
        self.vocab : List[str] = []
        self.word_to_idx : Dict[str, int] = {}
    
    def save(self, path:str) -> None:
        raise NotImplementedError
    
    def dump(self) -> Dict[str, Any]:
        return self.word_to_idx

    def load(self, path:str) -> None:
        raise NotImplementedError

    def train(self, texts: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def _tokenize(self, text:str) -> List[str]:
        raise NotImplementedError

    def __call__(self, text:str, **kwargs:Optional[Any]) ->  List[str]:
        if len(self.vocab) == 0:
            raise Exception("Either vectorizer is not trained or no vocab found")
        return self._tokenize(text, **kwargs)