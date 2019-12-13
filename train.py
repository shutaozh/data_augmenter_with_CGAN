import os
from model import CGAN
from utils import *

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("data_size", 524, "The size of data [524]")
flags.DEFINE_integer("condition_size", 524, "The size of the sketch [524]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise z. [100]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        cgan = CGAN(
            sess,
            data_size=FLAGS.data_size,
            condition_size=FLAGS.condition_size,
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            checkpoint_dir=FLAGS.checkpoint_dir,
            data_dir=FLAGS.data_dir)

        show_all_variables()

        cgan.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
