# for language task

import tensorflow as tf
import numpy as np
import sys

from config import lang_config as config
from models import Language
from snli_loader import SNLILoader

argvs = sys.argv

def run_task(data_dir, task_name = 'Language modeling'):
    snli_loader = SNLILoader(data_dir)
    
    train_data = snli_loader.train_data
    test_data = snli_loader.test_data

    lang_config = config(snli_loader, train_data, test_data)
    
    with tf.Session() as sess:
        model = Language(lang_config, sess, train_data, test_data)
        if len(argvs) < 2:
            model.build_model()
            model.run(task_name)
        else:
            model.demo_model()
            model.demo(argvs[1])

def main(_):
    run_task('./data/snli/')

if __name__ == '__main__':
    tf.app.run()
