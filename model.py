import os
import time

from ops import *
from utils import *
import csv


class CGAN(object):
    def __init__(self, sess, data_size=524, condition_size=524,
                 batch_size=64, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 checkpoint_dir=None, data_dir='./data'):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess

        self.batch_size = batch_size

        self.data_size = data_size
        self.condition_size = condition_size

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir

        self.c_dim = 1

        self.build_model()

    def build_model(self):

        data_dims = [1, self.data_size, 1]
        condition_dims = [self.condition_size]

        self.data_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + data_dims, name='real_data')

        self.condition_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + condition_dims, name='real_condition')

        condition = self.condition_inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, condition)
        self.D, self.D_logits = self.discriminator(self.data_inputs, condition, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, condition, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        penalty = tf.reduce_sum(tf.nn.relu(-(self.G * self.data_inputs))) / self.batch_size
        penalty2 = tf.reduce_sum(tf.abs((self.G - tf.abs(self.G)) - (self.data_inputs - tf.abs(self.data_inputs)))) / 30
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) + penalty + penalty2

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data, min_item, max_item = load_data(os.path.join(config.data_dir, 'TrainingDataReal.csv'))
        condition, min_item, max_item = load_condition(os.path.join(config.data_dir, 'TrainingDataSketch.csv'))

        # percent of APs
        for i in range(len(condition)):
            index = sum(np.array(np.where(condition[i][0:520] == 0)).tolist(), [])
            new_index = random.sample(index, round(1.0 * len(index)))
            condition[i][new_index] = 1

        batch_idxs = len(data) // config.batch_size
        is_training_G = False

        for epoch in xrange(config.epoch):

            for idx in xrange(0, int(batch_idxs)):
                batch1 = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch2 = condition[idx * config.batch_size:(idx + 1) * config.batch_size]

                batch_data1 = np.expand_dims(batch1, axis=1)
                batch_data = np.expand_dims(batch_data1, axis=-1)

                batch_condition = np.array(batch2)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # cross training
                if is_training_G:
                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.condition_inputs: batch_condition,
                                                              self.data_inputs: batch_data})
                    self.writer.add_summary(summary_str, counter)
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.data_inputs: batch_data, self.z: batch_z,
                                                              self.condition_inputs: batch_condition})
                    self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str, fake, conditions = self.sess.run([g_optim, self.g_sum, self.G, self.condition_inputs],
                                                                 feed_dict={self.z: batch_z,
                                                                            self.condition_inputs: batch_condition,
                                                                            self.data_inputs: batch_data})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.condition_inputs: batch_condition})
                errD_real = self.d_loss_real.eval(
                    {self.data_inputs: batch_data, self.condition_inputs: batch_condition})
                errG = self.g_loss.eval(
                    {self.z: batch_z, self.condition_inputs: batch_condition, self.data_inputs: batch_data})

                if errD_fake + errD_real < errG:
                    is_training_G = True
                else:
                    is_training_G = False

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 1000) == 0:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, data, condition, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            condition = tf.expand_dims(condition, 1)
            condition = tf.expand_dims(condition, -1)

            data = tf.concat([data, condition], 2)

            h0 = lrelu(conv2d(data, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, condition):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = 1, self.data_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z = tf.concat([z, condition], 1)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        model_name = "CGAN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
