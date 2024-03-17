from mlx.data.core import CharTrie
from typing import Set

# def construct_eng_char_trie_for_ctc(additional_chars):
#     trie = CharTrie()
#     trie.insert("@")  # blank
#     trie.insert(" ")
#     trie.insert("'")
#     for c in range(ord("a"), ord("z") + 1):
#         trie.insert(chr(c))
#     if additional_chars:
#         for c in additional_chars:
#             trie.insert(c)
#     return trie

