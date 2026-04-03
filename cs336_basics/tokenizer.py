import multiprocessing
import regex as re
import json
from collections.abc import Iterable, Iterator
from collections import Counter
from .pretokenization_example import find_chunk_boundaries

import pdb

# PAT_RULE = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
# PAT_RULE = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
# PAT_RULE = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class tokenizer_trainer:
    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] = ["<|endoftext|>"],
        num_processors: int = None
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processors = num_processors if num_processors is not None else multiprocessing.cpu_count()

        self.delimiter_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        self.pat_rule = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

    def pre_tokenization(self, start, end):
        subwords = {}
        with open(self.input_path, "rb") as file:
            file.seek(start)
            chunk = file.read(end-start).decode("utf-8", errors="ignore")
            split_chunks = re.split(self.delimiter_pattern, chunk)
        for sub_chunk in split_chunks:
            for match in re.finditer(self.pat_rule, sub_chunk):
                match_str = tuple(match.group().encode("utf-8"))
                subwords[match_str] = subwords.get(match_str, 0) + 1
        
        return subwords

    def pre_tokenization_wrapper(self, args):
        return self.pre_tokenization(*args)

    def multiprocessing_pre_tokenization(self):
        boundaries = find_chunk_boundaries(open(self.input_path, "rb"), self.num_processors, b"<|endoftext|>")
        tasks = [
            (start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        total_subwords = Counter()

        with multiprocessing.Pool(self.num_processors) as pool:
            for sub_dict in pool.imap_unordered(self.pre_tokenization_wrapper, tasks):
                total_subwords.update(sub_dict)
        
        return total_subwords

    def get_stats(self, subwords):
        stats = {}
        for key, value in subwords.items():
            for pair in zip(key, key[1:]):
                stats[pair] = stats.get(pair, 0) + value
        return stats

    def __call__(self):
        return self.train_bpe()

    def train_bpe(self):
        total_subwords = self.multiprocessing_pre_tokenization()
        num_merges = self.vocab_size - 256 - len(self.special_tokens)

        stats = self.get_stats(total_subwords)

        for i in range(num_merges):
            pair = max(stats, key=lambda x: (stats[x], self.vocab[x[0]], self.vocab[x[1]]))
            stats.pop(pair)
            idx = 256 + i
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))

            new_subwords = {}
            for key, value in total_subwords.items():
                new_key = []
                j = 0
                while j < len(key):
                    if (j < len(key) - 1) and (key[j] == pair[0]) and (key[j+1] == pair[1]):
                        new_key.append(idx)
                        if (j >= 1):
                            stats[(key[j-1], idx)] = stats.get((key[j-1], idx), 0) + value
                            stats[(key[j-1], pair[0])] = stats.get((key[j-1], pair[0]), 0) - value
                        if (j <= len(key) - 3):
                            stats[(idx, key[j+2])] = stats.get((idx, key[j+2]), 0) + value
                            stats[(pair[1], key[j+2])] = stats.get((pair[1], key[j+2]), 0) - value
                        j += 2
                    else:
                        new_key.append(key[j])
                        j += 1
                new_subwords[tuple(new_key)] = new_subwords.get(tuple(new_key), 0) + value
            total_subwords = new_subwords
        
        for i in range(self.vocab_size - len(self.special_tokens), self.vocab_size):
            special_token_idx = i - (self.vocab_size - len(self.special_tokens))
            special_token = self.special_tokens[special_token_idx].encode("utf-8")
            self.vocab[i] = special_token
        
        return self.vocab, self.merges

class tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        vocab_size = len(vocab)
        if special_tokens:
            for special_token in special_tokens:
                special_token = special_token.encode("utf-8")
                if special_token not in set(vocab.values()):
                    vocab[vocab_size] = special_token
                    vocab_size += 1
        
        # self.vocab_size = vocab_size
        self.decode_vocab = vocab
        self.encode_vocab = {v: k for k, v in vocab.items()}
        self.encode_merges = {v: k for k, v in enumerate(merges)}
        # self.encode_merges = set(merges)
        self.special_tokens = special_tokens
        self.pat_rule = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def pre_tokenization(self, text):
        subwords = []
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            split_chunks = re.split(f"({pattern})", text)
        else:
            split_chunks = []
            split_chunks.append(text)
        # pdb.set_trace()
        for sub_chunk in split_chunks:
            if self.special_tokens and sub_chunk in self.special_tokens:
                sub_chunk_encoded = []
                sub_chunk_encoded.append(sub_chunk.encode("utf-8"))
                subwords.append(sub_chunk_encoded)
            else:
                for match in re.finditer(self.pat_rule, sub_chunk):
                    match_encoded = match.group().encode("utf-8")
                    match_encoded_list = [match_encoded[i:i+1] for i in range(len(match_encoded))]
                    subwords.append(match_encoded_list)
        
        return subwords

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        # with open(vocab_filepath) as vocab_file:
        #     vocab = json.load(vocab_filepath)
        # return cls(...)
        ...

    def encode(self, text: str) -> list[int]:
        subwords = self.pre_tokenization(text)
        encoded_list = []
        for word in subwords:
            if self.special_tokens and (len(word) == 1) and (word[0].decode("utf-8") in self.special_tokens):
                encoded_list.append(self.encode_vocab.get(word[0]))
            else:
                while len(word)>=2:
                    word_pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                    to_merge_pair = min(word_pairs, key=lambda x: self.encode_merges.get(x, float('inf')))
                    if to_merge_pair not in self.encode_merges:
                        break
                    # do_merge
                    new_word = []
                    i = 0
                    while i < len(word):
                        if (i < len(word)-1) and (word[i] == to_merge_pair[0]) and (word[i+1] == to_merge_pair[1]):
                            new_word.append(to_merge_pair[0]+to_merge_pair[1])
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word
                    # new_word = []
                    # num_merged = 0
                    # i = 0
                    # while i < len(word):
                    #     # if i < len(word)-1 and (word[i], word[i+1]) in self.encode_merges:
                    #     if i < len(word)-1 and (word[i]+word[i+1]) in self.encode_vocab.keys():
                    #         new_word.append(word[i]+word[i+1])
                    #         i += 2
                    #         num_merged += 1
                    #     else:
                    #         new_word.append(word[i])
                    #         i += 1
                    # word = new_word
                    # if num_merged == 0:
                    #     break
                for w in word:
                    encoded_list.append(self.encode_vocab.get(w))
        
        return encoded_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            encoded_tokens = self.encode(text)
            for token in encoded_tokens:
                yield token

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b''
        for token in ids:
            decoded_bytes += self.decode_vocab.get(token)
        decoded_str = decoded_bytes.decode("utf-8", errors="replace")
        return decoded_str

if __name__ == "__main__":
    pass