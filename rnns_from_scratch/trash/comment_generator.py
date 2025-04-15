import os
from collections import Counter
import numpy as np
import tiktoken
import re

def tokenize(file_path="paul_graham_essay.txt"):
    enc = tiktoken.get_encoding("o200k_base")
    vocab_size = 20000
    max_line_length = 20

    f = open(file_path, "r")
    lines = f.readlines()
    tokenized_lines = []
    max_length = 0
    check_index = 400
    for index, line in enumerate(lines):
        html_remove_line = re.sub(r"<[^>]*>", "", line)
        if index == check_index:
            print(html_remove_line)
        small_case_line = html_remove_line.lower()
        if index == check_index:
            print(small_case_line)
        splitted_lines = small_case_line.split(".")
        if index == check_index:
            print(splitted_lines)
        for splitted_line in splitted_lines:
            stripped_line = splitted_line.strip()
            if len(stripped_line.split(" ")) > 0:
                tokenized_line = enc.encode(stripped_line)
                tokenized_lines.append(tokenized_line)
                if len(tokenized_line) > max_length:
                    max_length = len(tokenized_line)

        # splitted_line = small_case_line.split(" ")
        # stripped_line = " ".join(splitted_line[:max_line_length]).strip()
        # if len(stripped_line) > 70: 
        #     tokenized_line = enc.encode(stripped_line)
        #     tokenized_lines.append(tokenized_line)
        #     if len(tokenized_line) > max_length:
        #         max_length = len(tokenized_line)
    print("max_length: ", max_length)
    # all_tokens = [token for line in tokenized_lines for token in line]
    # token_counts = Counter(all_tokens)
    # top_tokens = {token for token, count in token_counts.most_common(vocab_size - 1)}
    # unk_token = enc.encode("<|UNK|>")[0]
    # top_k_token_lines = []
    # for line in tokenized_lines:
    #     top_k_token_line = [token if token in top_tokens else unk_token for token in line]
    #     top_k_token_lines.append(top_k_token_line)
        
    # padded_tokenized_lines = []
    # for top_k_token_line in top_k_token_lines:
    #     padding = [0] * (max_length - len(top_k_token_line))
    #     padded_tokenized_line = top_k_token_line + padding
    #     padded_tokenized_lines.append(padded_tokenized_line)
    
    padded_tokenized_lines = []
    for tokenized_line in tokenized_lines:
        padding = [0] * (max_length - len(tokenized_line))
        padded_tokenized_line = tokenized_line + padding
        padded_tokenized_lines.append(padded_tokenized_line)

    padded_tokenized_lines_array = np.array(padded_tokenized_lines)
    print("padded_tokenized_lines_array.shape: ", padded_tokenized_lines_array.shape)
    print("len(np.unique(padded_tokenized_lines_array)): ", len(np.unique(padded_tokenized_lines_array)))

    X = [[token for token in padded_tokenized_line[:-1]] for padded_tokenized_line in padded_tokenized_lines]
    y = [[token for token in padded_tokenized_line[1:]] for padded_tokenized_line in padded_tokenized_lines]

    index_to_check = 400
    print("X: ", X[index_to_check][:], 
          "\ny: ", y[index_to_check][:],
          "\ndecoded: ", enc.decode(X[index_to_check][:]))
    
    # if there are more zeros in more than half of the array then remove it from the dataset
    removed_tokenized_lines = []
    count_of_removed_lines = 0
    for padded_tokenized_line in padded_tokenized_lines:
        # print("number of zeros: ", (padded_tokenized_line.count(0) / len(padded_tokenized_line)) * 100)
        if padded_tokenized_line.count(0) > max_length // 2:
            count_of_removed_lines += 1
        else:
            removed_tokenized_lines.append(padded_tokenized_line)
    print("count_of_removed_lines: ", count_of_removed_lines)
    print("len(removed_tokenized_lines): ", len(removed_tokenized_lines))
    
if __name__ == "__main__":
    tokenize()