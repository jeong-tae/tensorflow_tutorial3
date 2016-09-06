#################################################
# To use GloVe, make txt corpus as a input file #
# All words separated by a single space         #
# ###############################################

import os
from tqdm import tqdm as it
from collections import defaultdict

def trimer(word):
    return word.replace(".", "").replace("?", "").replace(",", "").replace("'", "").replace('"', "").replace("!", "")

def read_words(fname):
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

def tokenization(data_file = None, fout_path = "./data/SNLI_train.txt"):

    pairs = read_words(data_file)

    snli_data = [ (s.lower(), e.lower()) for s, paraphs in pairs.items() for paraph in paraphs for label, e in paraph.items() if label == 'entailment' ]

    fout = open(fout_path, 'w')
    for s1, s2 in snli_data:
        splits = s1.split() + s2.split()
        splits = [trimer(word) for word in splits]
        tokens = ' '.join(splits)
        fout.write(tokens + ' ')
    fout.close()

def snli_tokenize(data_dir):
    train_fname = data_dir + "snli_1.0/snli_1.0_train.txt"
    valid_fname = data_dir + "snli_1.0/snli_1.0_dev.txt"
    test_fname = data_dir + "snli_1.0/snli_1.0_test.txt"

    train_fout = data_dir + "snli_train.txt"
    valid_fout = data_dir + "snli_dev.txt"
    test_fout = data_dir + "snli_test.txt"

    print(" [*] Tokenization... snli...")
    tokenization(train_fname, train_fout)
    tokenization(valid_fname, valid_fout)
    tokenization(test_fname, test_fout)
    print(" [*] Tokenize Done!")
    
if __name__ == '__main__':
    snli_tokenize()







    
