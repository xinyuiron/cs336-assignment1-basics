import importlib.metadata
from .tokenizer import tokenizer_trainer, tokenizer
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
