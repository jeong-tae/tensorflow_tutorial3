# cornell loader

import sys
import os

def read_file_cornell(fileName):
    a=[]
    input_file = open(fileName, "r")
    for line in input_file:
        a.append(line.strip().split(' +++$+++ '))
    input_file.close()
    return a

def read_conversation_order(fileName):
    ls = read_file_cornell(fileName)
    a=[]
    for i in ls:
        L = i[3].split("', '")
        temp = []
        last = len(L)-1
        temp.append(L[0][2:])
        for j in range(1,last):
            temp.append(L[j])
        temp.append(L[last][:len(L[last])-2])
        a.append(temp)
    return a

def conversation_corpus(line_fname = 'data/dialog/cornell movie-dialogs corpus/movie_lines.txt', con_fname = 'data/dialog/cornell movie-dialogs corpus/movie_conversations.txt'):
    text = read_file_cornell(line_fname)
    order = read_conversation_order(con_fname)
    
    lineID = []
    for i in text:
        lineID.append(i[0])

    corpus = []
    count = 0.0
    for od in order:
        count = count + 1
        if count % 500 == 0.0:
            print count/len(order)
        first=lineID.index(od[0])
        temp = []
        for j in range(len(od)):
            temp.append(text[first-j])
        corpus.append(temp)
    return corpus

def conversation_corpus_50():
    text = read_file_cornell('movie_lines.txt')
    order = read_conversation_order('movie_conversations.txt')

    lineID = []
    for i in text:
        lineID.append(i[0])

    corpus = []
    count = 0.0
    itr = 0
    for od in order:
        count = count + 1
        itr = itr + 1
        if itr == 2500:
            break
        if count % 500 == 0.0:
            print count/len(order)
        first=lineID.index(od[0])
        temp = []
        for j in range(len(od)):
            temp.append(text[first-j])
        corpus.append(temp)
    return corpus

def make_data():
    corpus = conversation_corpus()
    
    src = open("source.txt",'w')
    trg = open("target.txt",'w')
    
    deg = 0
    for line in corpus:
        noUtt = False
        itr = 0
        for i in range(len(line)):
            if len(line[i])<5:
                noUtt = True
                itr = i
                print "There is no Utterance on %s" % line[i][0]
                print "iter : %d" % deg

        for i in range(0,len(line)-1):
            if noUtt:
                if itr-i == 1 or itr-i == 0:
                    continue
            src.write(line[i][4]+'\n')
        for i in range(1,len(line)):
            if noUtt:
                if itr-i == 0 or itr-i == -1:
                    continue
            trg.write(line[i][4]+'\n')
        deg = deg + 1
    src.close()
    trg.close()

def make_data_cut(dir_path = 'data/dialog/cornell movie-dialogs corpus/'):
    line_fname = dir_path + 'movie_lines.txt'
    con_fname = dir_path + 'movie_converstations.txt'
    corpus = conversation_corpus(line_fname, con_fname)
    
    src = open(dir_path + "source.txt",'w')
    trg = open(dir_path + "target.txt",'w')
    
    deg = 0
    for line in corpus:
        noUtt = False
        itr = 0
        for i in range(len(line)):
            if len(line[i])<5:
                noUtt = True
                itr = i
                print "There is no Utterance on %s" % line[i][0]
                print "iter : %d" % deg

        for i in range(0,len(line)-1):
            if noUtt:
                continue
            src.write(line[i][4]+'\n')
        for i in range(1,len(line)):
            if noUtt:
                continue
            trg.write(line[i][4]+'\n')
        deg = deg + 1
    src.close()
    trg.close()

def check_no_utterance():
    corpus = conversation_corpus()
    deg = 0
    for line in corpus:
        for i in range(len(line)):
            if len(line[i])<5:
                print"There is no Utterance on %s" % line[i][0]
                print "iter : %d" % deg
        deg = deg + 1

def build_vocab(sources, targets):
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

def load_file_by_lines(fname):
    return open(fname, 'r').readlines()

class CornellLoader(object):
    def __init__(self, data_dir = '.data/dialog/cornell movie-dialogs corpus/'):
        source_fname = os.path.join(data_dir, 'source.txt')
        target_fname = os.path.join(data_dir, 'target.txt')        

        if os.path.exists(source_fname):
            print(" [*] Load source and target...")
            self.sources = load_file_by_lines(source_fname)
            self.targets = load_file_by_lines(target_fname)
            self.word2idx, self.idx2word = build_vocab(source_fname, target_fname)
        else:
            print(" [*] Preprocessing conversation into source and target...")
            make_data_cut()
            self.sources = load_file_by_lines(source_fname)
            self.targets = load_file_by_lines(target_fname)
            self.word2idx, self.idx2word = build_vocab(source_fname, target_fname)


