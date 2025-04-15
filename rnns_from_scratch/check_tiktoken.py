import tiktoken
import numpy as np

enc = tiktoken.get_encoding("o200k_base")
# print(enc.decode(enc.encode("hello world")))
print(enc.decode(np.arange(20)))
print(enc.decode([10]))