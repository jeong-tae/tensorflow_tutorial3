# for conversation task

import tensorflow as tf
import numpy as np
import sys

from config import con_config as config
from models import Conversation
from cornell_loader import CornellLoader

argvs = sys.argv

def run_task(data_dir, task_name = 'Conversation'):
    data_loader = CornellLoader(data_dir, length = 5000)
    sources = data_loader.sources
    targets = data_loader.targets
    con_config = config(data_loader, sources, targets)

    with tf.Session() as sess:
        model = Conversation(con_config, sess, sources, targets)
        if len(argvs) < 2:
            model.build_model(False)
            model.run(task_name)
        elif argvs[1] == 'demo':
            model.build_model(True)
            model.demo()

def main(_):
    run_task('data/dialog/cornell movie-dialogs corpus/', 'Conversation')

if __name__ == '__main__':
    tf.app.run()
