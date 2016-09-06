import numpy as np
import pdb

def trimer(word):
    return word.replace(".", "").replace("?", "").replace(",", "").replace("'", "").replace('"', "").replace("!", "")

def sentence_to_indices(sentence, word2idx, size, is_target = True):
    splits = sentence.split()
    idx = 0
    if is_target:
        splits = splits + ["<EOS>"]
        idx = 0

    indices = np.zeros((size), np.int32)
    for w in splits:
        if trimer(w) == '':
            continue
        try:
            if w not in word2idx:
                indices[idx] = word2idx['<UNK>']
            else:
                indices[idx] = word2idx[trimer(w)]
        except(IndexError):
            break
        idx += 1

    return indices

def indices_to_sentence(indices, idx2word, size):
    tokens = []
    for i in range(size):
        token = idx2word[indices[i]]
        if token == "<EOS>":
            break
        tokens.append(token)
    return ' '.join(tokens)

def sentence_to_wordvecs(sentence, word2vec, size, edim):
    splits = sentence.split()

    wordvecs = np.zeros((size, edim), np.float32)
    wordvecs[0, :] = word2vec["<GO>"]
    idx = 1
    for w in splits:
        if trimer(w) == '':
            continue
        try:
            wordvecs[idx, :] = word2vec[trimer(w)]
        except(IndexError):
            break
        idx += 1

    return wordvecs

def bow(size, indices): # in this, same as wordvec
    bow = np.zeros((size,), dtype = np.int16)
    for idx in indices:
        bow[idx] += 1
    return bow
