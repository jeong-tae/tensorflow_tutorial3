import numpy as np

class lang_config(object):
    def __init__(self, snli_loader, train_data, test_data):
        self.word2idx = snli_loader.word2idx
        self.idx2word = snli_loader.idx2word
        self.voca_size = len(self.word2idx)
        self.embed = snli_loader.embed
        self.word2vec = snli_loader.word2vec

        self.max_words      = 0
        for s1, s2 in train_data:
            length1, length2 = len(s1.split()), len(s2.split())
            self.max_words = max(self.max_words, length1, length2)

        for s1, s2 in test_data:
            length1, length2 = len(s1.split()), len(s2.split())
            self.max_words = max(self.max_words, length1, length2)
        self.max_words += 1

        self.batch_size     = 32
        self.std_init       = 0.02
        self.edim           = 300
        self.max_epoch      = 15
        self.train_range    = np.array(range(len(train_data)))
        self.test_range     = np.array(range(len(test_data)))
        self.lr             = 0.05
        self.max_grad_norm  = 40

class con_config(object):
    def __init__(self, cornell_loader, sources, targets):
        # no seperated train, test
        self.word2idx = cornell_loader.word2idx
        self.idx2word = cornell_loader.idx2word
        self.voca_size = len(self.word2idx)
        
        self.max_words      = 0
        for s in sources:
            length = len(s.split())
            self.max_words = max(self.max_words, length)

        for t in targets:
            length = len(t.split())
            self.max_words = max(self.max_words, length)
        self.max_words += 1

        if len(sources) != len(targets):
            raise "Sources and targets has not same length"

        self.batch_size     = 32
        self.std_init       = 0.02
        self.edim           = 300
        self.max_epoch      = 3
        self.data_range     = np.array(range(len(sources)))
        self.lr             = 0.05
        self.max_grad_norm  = 40
