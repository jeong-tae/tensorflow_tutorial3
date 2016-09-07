import tensorflow as tf
import os
import numpy as np
import random
from tqdm import tqdm

from .util import *

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

        self.encode_inputs = tf.placeholder(tf.int32, [None, self.max_words])
        self.decode_inputs = tf.placeholder(tf.int32, [None, self.max_words])
        self.decode_targets = tf.placeholder(tf.int32, [None, self.max_words])
        self.target_weights = tf.placeholder(tf.float32, [None, self.max_words])

    def build_model(self, feed_previous = False):
        print(" [*] model build...")
        with tf.variable_scope("embedding"):
            self.embedding = tf.get_variable("w", [self.voca_size, self.edim])
        embedded = tf.nn.embedding_lookup(self.embedding, self.encode_inputs)

        self.gru_cell = tf.nn.rnn_cell.GRUCell(self.edim)
        e_state = None
        if feed_previous:
            e_state = self.init_state = self.gru_cell.zero_state(1, tf.float32)
        else:
            e_state = self.init_state = self.gru_cell.zero_state(self.batch_size, tf.float32)
        def length(data):
            used = tf.sign(data)
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(self.gru_cell, embedded, dtype = tf.float32, sequence_length=length(self.encode_inputs))

        def decoder(decoder_input, init_state, cell, feed_previous = False):
            with tf.variable_scope("embedding", reuse = True):
                embedding = tf.get_variable("w", [self.voca_size, self.edim])
            embedded = tf.nn.embedding_lookup(embedding, decoder_input)
            state = tf.identity(init_state)        

            outputs = []
            with tf.variable_scope("decoding"):
                for t in range(self.max_words):
                    if t > 0: tf.get_variable_scope().reuse_variables()
                    inp = embedded[:, t, :]
                    if outputs and feed_previous:
                        inp = outputs[-1]
                    output, state = cell(inp, state)
                    outputs.append(output)
            return tf.concat(1, outputs), state

        decoder_outputs, decoder_state = decoder(self.decode_inputs, encoder_state, self.gru_cell, feed_previous)

        self.W = tf.get_variable("softmax_w", [self.edim, self.voca_size])
        self.b = tf.get_variable("softmax_b", [self.voca_size])
        
        outputs = tf.reshape(decoder_outputs, [-1, self.edim])
        logits = tf.matmul(outputs, self.W) + self.b
        self.preds = tf.reshape(tf.nn.softmax(logits), [-1, self.max_words, self.voca_size])

        self.loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.decode_targets, [-1])], [tf.reshape(self.target_weights, [-1])])
        self.cost = tf.reduce_sum(self.loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads = optimizer.compute_gradients(self.cost, tvars)
        self.op = optimizer.apply_gradients(grads)
        print(" [*] build done")

    def train(self):
        total_costs = []

        batch_len = len(self.data_range) // self.batch_size
        for step, (batch_e_input, batch_d_input, batch_d_target, batch_target_weights) in tqdm(enumerate(self.data_iteration()), desc = 'train', total = batch_len):

            _, cost, pred = self.sess.run([self.op, self.cost, self.preds],
                            feed_dict = {   self.encode_inputs: batch_e_input,
                                            self.decode_inputs: batch_d_input,
                                            self.decode_targets: batch_d_target,
                                            self.target_weights: batch_target_weights
                                        })

            #if step % 100 == 0:
            #    indices = [ np.argmax(p) for p in pred[0] ]
            #    sentence = indices_to_sentence(indices, self.idx2word, self.max_words)
            #    print(" [*] sample: %s" % sentence)                

            total_costs.append(cost)
        return sum(total_costs)

    def run(self, argv):
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        
        for i in range(self.max_epoch):
            train_cost = self.train()

            print("Epoch: %d, Train cost: %.3f" % (i+1, train_cost))
    
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints/")
            save_path = self.saver.save(self.sess, "checkpoints/conv.ckpt")

    def demo(self):
        print("")

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "checkpoints/conv.ckpt")

        while(True):
            encode = raw_input("You(exit=-1)>> ").lower()
            if encode == '-1':
                break
            encode = sentence_to_indices(encode, self.word2idx, self.max_words, False)
            decode = sentence_to_indices("<GO>", self.word2idx, self.max_words, False)
            batch_e_input = np.zeros((1, self.max_words), np.int32)
            batch_d_input = np.zeros((1, self.max_words), np.int32)
            batch_e_input[0, :] = encode
            batch_d_input[0, :] = decode

            pred = self.sess.run([self.preds],
                            feed_dict = {   self.encode_inputs: batch_e_input,
                                            self.decode_inputs: batch_d_input
                                        })
            indices = [np.argmax(p) for p in pred[0][0]]
        
            sentence = indices_to_sentence(indices, self.idx2word, self.max_words)
            print("bot>> %s" % sentence)

    def data_iteration(self):
        data_range = self.data_range
        random.shuffle(data_range)
        
        batch_len = len(data_range) // self.batch_size

        for l in xrange(batch_len):
            batch = data_range[self.batch_size * l:self.batch_size *(l+1)]

            batch_e_input = np.zeros((self.batch_size, self.max_words), np.int32)
            batch_d_input = np.zeros((self.batch_size, self.max_words), np.int32)
            batch_d_target = np.zeros((self.batch_size, self.max_words), np.int32)
            batch_d_target_weights = np.zeros((self.batch_size, self.max_words), np.float32)
            

            for b in range(self.batch_size):
                s, t = self.source_data[batch[b]], self.target_data[batch[b]]

                encode = sentence_to_indices(s, self.word2idx, self.max_words, False)
                decode = sentence_to_indices("<GO> "+t, self.word2idx, self.max_words, False)
                decode_target = sentence_to_indices(t, self.word2idx, self.max_words, True)
                target_weight = np.sign(decode_target)

                batch_e_input[b, :] = encode
                batch_d_input[b, :] = decode
                batch_d_target[b, :] = decode_target
                batch_d_target_weights[b, :] = target_weight

            yield batch_e_input, batch_d_input, batch_d_target, batch_d_target_weights
