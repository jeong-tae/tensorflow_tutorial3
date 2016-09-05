import random
import math
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .util import *

random.seed(1003)
np.random.seed(1003)

class Language(object):
    def __init__(self, config, sess, train, test):
        self.sess = sess
        self.train_data = train
        self.test_data = test
        self.train_range = config.train_range
        self.test_range = config.test_range

        self.word2idx = config.word2idx
        self.idx2word = config.idx2word
        self.voca_size = config.voca_size
        self.max_words = config.max_words
        self.word2vec = config.word2vec

        self.batch_size = config.batch_size
        self.edim = config.edim
        self.max_epoch = config.max_epoch
        self.lr = tf.Variable(config.lr, trainable = False)
        self.max_grad_norm = config.max_grad_norm

        self.sources = tf.placeholder(tf.float32, [self.batch_size, self.max_words, self.edim])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.max_words])
        self.target_weights = tf.placeholder(tf.float32, [self.batch_size, self.max_words])

        self.demo_source = tf.placeholder(tf.float32, [1, self.edim])

    def build_model(self):
        print(" [*] model building...")

        with tf.variable_scope("projection"):
            self.p = tf.get_variable('weights', shape = [self.edim, self.voca_size], initializer = tf.contrib.layers.xavier_initializer())
            self.p_b = tf.get_variable('biases', shape = [self.voca_size], initializer = tf.constant_initializer(0.0))

        self.gru_cell = tf.nn.rnn_cell.GRUCell(self.edim)
        self.init_state = self.gru_cell.zero_state(self.batch_size, tf.float32)

        state = self.init_state
        outputs = []
        with tf.variable_scope("language"):
            for t in range(self.max_words):
                if t > 0: tf.get_variable_scope().reuse_variables()
                output, state = self.gru_cell(self.sources[:, t, :], state)
                outputs.append(output)
        outputs = tf.reshape(tf.concat(1, outputs), [-1, self.edim])

        self.logits = tf.matmul(outputs, self.p) + self.p_b
        self.preds = tf.reshape(tf.nn.softmax(self.logits), [-1, self.max_words, self.voca_size])

        self.loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.targets, [-1])], [tf.reshape(self.target_weights, [-1])])
        self.cost = tf.reduce_sum(self.loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.op = optimizer.apply_gradients(zip(grads, tvars))

    def demo_model(self):
        print(" [*] model building for demo")
        
        with tf.variable_scope("projection"):
            self.p = tf.get_variable('weights', shape = [self.edim, self.voca_size], initializer = tf.contrib.layers.xavier_initializer())
            self.p_b = tf.get_variable('biases', shape = [self.voca_size], initializer = tf.constant_initializer(0.0))

        self.gru_cell = tf.nn.rnn_cell.GRUCell(self.edim)
        self.init_state = self.gru_cell.zero_state(1, tf.float32)

        state = self.init_state
        outputs = []
        with tf.variable_scope("language"):
            output, state = self.gru_cell(self.demo_source, state)
            outputs.append(output)
        self.last_state = state
        outputs = tf.reshape(tf.concat(1, outputs), [-1, self.edim])

        self.logits = tf.matmul(outputs, self.p) + self.p_b
        self.preds = tf.reshape(tf.nn.softmax(self.logits), [-1, 1, self.voca_size])


    def train(self):
        N = len(self.train_range) // self.batch_size
        costs = 0.0
        self.init_state.eval()

        for step, (batch_source, batch_target, batch_target_weights) in tqdm(enumerate(self.data_iteration(self.train_data, True)), desc = 'train', total = N):
            _, loss =  self.sess.run([self.op, self.loss],
                                feed_dict = {   self.sources: batch_source,
                                                self.targets: batch_target,
                                                self.target_weights: batch_target_weights
                                            })
            costs += np.sum(loss)
        return costs/N/self.batch_size

    def test(self):
        N = len(self.test_range) // self.batch_size
        costs = 0.0
        self.init_state.eval()

        for step, (batch_source, batch_target, batch_target_weights) in tqdm(enumerate(self.data_iteration(self.test_data, False)), desc = 'test', total = N):
            loss =  self.sess.run([self.loss],
                                feed_dict = {   self.sources: batch_source,
                                                self.targets: batch_target,
                                                self.target_weights: batch_target_weights
                                            })
            costs += np.sum(loss)
        return costs/N/self.batch_size

    def run(self, task_name = ''):
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()

        for i in range(self.max_epoch):
            train_cost = self.train()
            test_cost = self.test()

            print("Epoch: %d, Train perplexity: %.3f" % (i+1, math.exp(train_cost)))
            print("Epoch: %d, Test perplexity: %.3f" % (i+1, math.exp(test_cost)))
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints/")
            save_path = self.saver.save(self.sess, "checkpoints/lang.ckpt")
        test_cost = self.test()
        print("Test perplexity: %.3f" % math.exp(test_cost))
    
    def demo(self, args):

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "checkpoints/lang.ckpt")

        # language generation
        def softmax(softmax_inputs, temp):
            shifted_inputs = softmax_inputs - softmax_inputs.max()
            exp_outputs = np.exp(temp * shifted_inputs)
            exp_outputs_sum = exp_outputs.sum()
            if np.isnan(exp_outputs_sum):
                return exp_outputs * float('nan')
            assert exp_outputs_sum > 0
            if np.isinf(exp_outputs_sum):
                return np.zeros_like(exp_outputs)
            eps_sum = 1e-20
            return exp_outputs / max(exp_outputs_sum, eps_sum)

        def random_choice_from_probs(pred, temp = 1.0):
            if temp == float('inf'):
                return np.argmax(softmax_inputs)
            probs = softmax(pred[0], temp)
            r = random.random()
            cum_sum = 0.
            for i, p in enumerate(probs):
                cum_sum += p
                if cum_sum >= r: return i
            return 0   

        def generate_sentence(temp):
            word_input = np.zeros((1, self.edim), np.float32)
            word_input[0, :] = self.word2vec["<GO>"]
            state = self.init_state.eval(session = self.sess)
            sentence = []
            while( len(sentence) < self.max_words and (not sentence or sentence[-1] != self.word2idx["<EOS>"]) ):
                pred, state = self.sess.run([self.preds, self.last_state], 
                        feed_dict = { self.demo_source: word_input,
                                        self.init_state: state })
                #import pdb
                #pdb.set_trace()
                #sentence.append(random_choice_from_probs(pred[0], temp))
                sentence.append(np.argmax(pred[0]))
                word_input[0, :] = self.word2vec[self.idx2word[sentence[-1]]]
            return sentence

        while(True):
            temp = float(raw_input(" [*] Input seed positive number(exit:-1):"))
            if temp == -1:
                break
            indices = generate_sentence(temp)
            sentence = indices_to_sentence(indices, self.idx2word, self.max_words)
            print(" [*] generated sentence: %s" % sentence)

    def data_iteration(self, data, is_train = True):
        data_range = None
        if is_train:
            data_range = self.train_range
            random.shuffle(data_range)
        else:
            data_range = self.test_range

        batch_len = len(data_range) // self.batch_size

        for l in xrange(batch_len):
            batch = data_range[self.batch_size * l:self.batch_size * (l+1)]

            batch_source = np.zeros((self.batch_size, self.max_words, self.edim), np.float32)
            batch_target = np.zeros((self.batch_size, self.max_words), np.int32)
            batch_target_weights = np.zeros((self.batch_size, self.max_words), np.int32)

            for b in range(self.batch_size):
                s1, s2 = data[batch[b]]

                source = sentence_to_wordvecs(s1, self.word2vec, self.max_words, self.edim)
                target = sentence_to_indices(s1, self.word2idx, self.max_words)
                target_weight = np.sign(target)

                batch_source[b, :, :] = source
                batch_target[b, :] = target
                batch_target_weights[b, :] = target_weight

            yield batch_source, batch_target, batch_target_weights









