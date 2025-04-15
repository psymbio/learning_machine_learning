import os
import numpy as np
import pandas as pd
import tiktoken
import pickle

enc = tiktoken.get_encoding("o200k_base")

def softmax(x):
    """
    Compute the softmax of a vector x.
    """
    return np.exp(x)/sum(np.exp(x))

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    
    def forward_propagation(self, x):
        """
        x_t is the input at time step t.
        For example, x_1 could be a one-hot vector corresponding to the second word of a sentence.

        s_t is the hidden state at time step t.
        It's the "memory" of the network.
        Its dimension is T x hidden_dim

        s_t is calculated based on the previous hidden state and the input at the current step:
        $$ s_t = f(U x_t + W s_{t-1}) $$
        
        The function f is usually a nonlinearity such as tanh or ReLU.

        s_{-1}, which is required to calculate the first hidden state, is typically initialized to all zeroes.

        o_t is the output at step t.
        For example, if we wanted to predict the next word in a sentence, o_t would be a vector of probabilities across our vocabulary.

        o_t is calculated as:
        $$ o_t = \text{softmax}(V s_t) $$
        """
        
        T = len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))

        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            # print(x[t])
            s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

def tokenize(file_path="poems-100.csv"):
    df = pd.read_csv(file_path)
    check_index = 50
    max_tokenized_length = 0
    tokenized_lines = []
    for index, row in df.iterrows():
        line = row['text']
        small_case_line = line.lower()
        stripped_line = small_case_line.strip()
        if index == check_index:
            print(stripped_line)
        
        for line in stripped_line.split("\n"):
            tokenized_line = enc.encode(line)
            tokenized_lines.append(tokenized_line)
            if max_tokenized_length < len(tokenized_line):
                max_tokenized_length = len(tokenized_line)

    
    padded_tokenized_lines = []
    for tokenized_line in tokenized_lines:
        padding = [10] * (max_tokenized_length - len(tokenized_line))
        padded_tokenized_line = tokenized_line + padding
        padded_tokenized_lines.append(padded_tokenized_line)

    padded_tokenized_lines_array = np.array(padded_tokenized_lines)
    print("padded_tokenized_lines_array.shape: ", padded_tokenized_lines_array.shape)
    print("len(np.unique(padded_tokenized_lines_array)): ", len(np.unique(padded_tokenized_lines_array)))

    X = np.array([[token for token in padded_tokenized_line[:-1]] for padded_tokenized_line in padded_tokenized_lines])
    y = np.array([[token for token in padded_tokenized_line[1:]] for padded_tokenized_line in padded_tokenized_lines])

    index_to_check = 30
    print("X: ", X[index_to_check][:30], 
          "\ny: ", y[index_to_check][:20],
          "\ndecoded: ", enc.decode(X[index_to_check][:30]))
    
    def manual_one_hot(matrix, vocab_size, save_mapping):
        one_hot = np.zeros((matrix.shape[0], matrix.shape[1], vocab_size), dtype=np.float32)
        mapping = {}
        key_count = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] not in mapping.keys():
                    mapping[matrix[i][j]] = key_count
                    key_count += 1
                    one_hot[i, j, mapping[matrix[i][j]]] = 1.0
                else:
                    one_hot[i, j, mapping[matrix[i][j]]] = 1.0
        print("mapping keys len: ", len(mapping.keys()), "saved: ", save_mapping)
        if save_mapping:
            with open('mapping.pkl', 'wb') as f:
                pickle.dump(mapping, f)
        return one_hot
    
    vocab_size = len(np.unique(padded_tokenized_lines_array))
    X_one_hot = manual_one_hot(X, vocab_size, save_mapping=True)
    y_one_hot = manual_one_hot(y, vocab_size, save_mapping=False)

    return X_one_hot, y_one_hot

def see_sentence(sample):
    with open('mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    reversed_mapping = {v: k for k, v in mapping.items()}
    indexes_of_word = np.argmax(sample, axis=1)
    sentence = []
    for index_of_word in indexes_of_word:
        sentence.append(reversed_mapping[int(index_of_word)])
    print(enc.decode(sentence))
    

if __name__ == "__main__":
    X_path = "X.npy"
    y_path = "y.npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        X, y = tokenize()
        np.save(X_path, X)
        np.save(y_path, y)
    
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)


    np.random.seed(10)
    vocabulary_size = 5721
    model = RNNNumpy(vocabulary_size)
    o, s = model.forward_propagation(X[10])
    # print(o.shape)
    # print(o)

    see_sentence(o)