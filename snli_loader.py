import numpy as np
import io
import os
import cPickle
from tqdm import tqdm as it
from collections import defaultdict, Counter, OrderedDict
from txt_maker import *

np.random.seed(1003)

def build_vocab(train, test):
    word2idx = { '<PAD>':0, '<GO>':1, '<EOS>':2 } # seed
    idx2word = { 0: '<PAD>', 1: '<GO>', 2: '<EOS>'}

    train_txt = open(train, 'r').read()
    test_txt = open(test, 'r').read()

    tokens = train_txt.split() + test_txt.split()
    tokens = set(tokens)
    for idx, token in enumerate(tokens):
        word2idx[trimer(token)] = idx + 3
        idx2word[idx + 3] = trimer(token)
    print("   [*] vocab size: %d" % len(word2idx))

    return word2idx, idx2word

def gloveLoad(glove_fpath, vocab):
    embed_path = glove_fpath + '.embed.pkl'
    word2vec_path = glove_fpath + '.word2vec.pkl'

    embed = None
    word2vec = {}

    if os.path.exists(word2vec_path):
        embed = load_pkl(embed_path)
        word2vec = load_pkl(word2vec)
    else:
        with io.open(glove_fpath + "/snli_vectors.txt", 'r') as f:
            for idx, l in it(enumerate(f), "  [*] Load glove"):
                tokens = l.split(' ')
                word = tokens[0]
                vecs = tokens[1:]

                if vocab.has_key(word):
                    word2vec[word] = np.array(vecs, dtype="float32")

    return embed, word2vec

def add_oov(word2idx, word2vec):
    for word in word2idx.keys():
        if not word2vec.has_key(word) and word != 'PAD':
            word2vec[word] = np.random.uniform(-0.5, 0.5, 300) # edim = 300
        elif word == 'PAD':
            word2vec[word] = np.zeros((300), np.float32)
    return word2vec # get return or not, same result.

def make_embed(word2idx, word2vec):
    embed_dim = 300 # from GloVe config

    embed = np.zeros(shape=(len(word2idx), embed_dim), dtype='float32')
    for idx, word in enumerate(word2vec.keys()):
        embed[word2idx[word]] = word2vec[word]

    return embed

class SNLILoader(object):
    def __init__(self, data_dir=".data/snli/", glove_fpath="./data/glove"):
        fname = "snli_1.0/snli_1.0"

        train_fname = os.path.join(data_dir, '%s_train.txt' % fname)
        test_fname = os.path.join(data_dir, '%s_test.txt' % fname)
        
        train_txt = os.path.join(data_dir, 'snli_train.txt')
        test_txt = os.path.join(data_dir, 'snli_test.txt')

        if os.path.exists(train_txt):
            self.word2idx, self.idx2word = build_vocab(train_txt, test_txt)
        else:
            snli_tokenize(data_dir = data_dir)
            self.word2idx, self.idx2word = build_vocab(train_txt, test_txt)

        self.train_data = self.file_to_sentence_pair(train_fname)
        self.test_data = self.file_to_sentence_pair(test_fname)

        self.embed, self.word2vec = gloveLoad(glove_fpath, self.word2idx)
        add_oov(self.word2idx, self.word2vec)
        self.embed = make_embed(self.word2idx, self.word2vec)

    def read_words(self, fname):
        pairs = defaultdict(list)

        if not os.path.exists(fname):
            raise IOError("%s not exists. Execute ./download.sh to download data." % fname)

        with open(fname, "r") as f:
            if "train" in fname:
                total = 550153
            elif "dev" in fname:
                total = 10001
            elif "test" in fname:
                total = 10001

            for idx, l in it(enumerate(f), "Read %s" % fname):
                if idx == 0: continue

                splits = l.split("\t")
                s1, s2, s1_idx, label = splits[5].strip('.'), splits[6].strip('.'), splits[7], splits[0]

                pairs[s1].append({label:s2})

        return pairs

    def file_to_sentence_pair(self, path):
        snli_path = path + ".entail.pkl"

        if os.path.exists(snli_path):
            snli_data = load_pkl(snli_path)

            return snli_data

        print(" [*] Processing %s into indices..." % path)
        pairs = self.read_words(path)

        if "_train.txt" in path:
            s1_words = [w for s1 in pairs.keys() for w in s1.split()]
            s2_words = [w for l2s in pairs.values() for s in l2s for k_v in l2s for w in k_v.values()]

        labels = ['entailment', 'neutral', 'contradiction']
        snli_data = [ (s.lower(), e.lower()) for s, paraphs in pairs.items() for paraph in paraphs for label, e in paraph.items() if label == 'entailment' ]

        save_pkl(snli_data, snli_path)

        return snli_data

def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj


