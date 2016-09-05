# for conversation task

import tensorflow as tf
import numpy as np
import sys

from config import con_config as config
from models import Conversation
from cornell_loader import CornellLoader

argvs = sys.argv

def run_task(data_dir, task_name = 'Conversation'):
    data_loader = CornellLoader(data_dir)
    con_config = config(data_loader)

    with tf.Session() as sess:
        model = Conversation(con_config, sess, data_loader.sources, data_loader.targets)
        model.build_model()
        if len(argvs) < 2:
            model.run(task_name)
        elif argvs[1] == 'demo':
            model.demo(argvs[1])

def main(_):
    run_task('data/dialog/cornell movie-dialogs corpus/', 'Conversation')

if __name__ == '__main__':
    tf.app.run()
