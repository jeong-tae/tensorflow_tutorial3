
class Conversation(object):
    def __init__(self, config, sess, sources, targets):
        self.sess = sess
        self.source_data = sources
        self.target_data = targets

        self.word2idx = config.word2idx
        self.idx2word = config.idx2word
        self.voca_size = config.voca_size
        self.max_words = config.max_words
        self.data_range = config.data_range
        
        self.batch_size = config.batch_size
        self.edim = config.edim
        self.max_epoch = config.max_epoch
        self.lr = tf.Variable(config.lr, trainable = False)
        self.max_grad_norm = config.max_grad_norm

        self.inputs = tf.placeholder(tf.float32, [None, self.max_words])
        self.targets = tf.placeholder(tf.int32, [None, self.max_words])
        self.target_weights = tf.placeholder(tf.float32, [None, self.max_words])

    def build_model(self, feed_previous = False):
        

    def data_iteration(self):
        data_range = self.data_range
        random.shuffle(data_range)
        
        batch_len = len(data_range) // self.batch_size

        for l in xrange(batch_len):
            batch = data_range[self.batch_size * l:self.batch_size *(l+1)]

            batch_input = np.zeros((self.batch_size, self.max_words), np.int32)
            batch_targets = np.zeros((self.batch_size, self.max_words), np.int32)
