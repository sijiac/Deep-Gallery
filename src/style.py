import tensorflow as tf
import vgg
import genr_net
import functools
import numpy as np
import os
from tools import load_image, save_image

STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER = 'relu4_2'

SAVE_PERIOD = 5


def _vggnet(data_path, input_image):
    weights, mean_pixel = vgg.load_net(data_path)
    image = vgg.preprocess(input_image, mean_pixel)
    net = vgg.net_preloaded(weights, image)
    return net


# To be used in Style.train()
def gram_matrix(tensor):
    shape = tensor.get_shape()
    numOfBatch, h, w, numOfChannels = shape[0].value, shape[1].value, shape[2].value, shape[3].value
    single_size = h * w * numOfChannels
    matrix = tf.reshape(tensor, (numOfBatch, -1, numOfChannels))
    matrixT = tf.transpose(matrix, perm=[0, 2, 1])
    gram = tf.matmul(matrixT, matrix) / single_size
    return gram


def _target_grams(vgg_path, style_image):
    style_grams = {}
    style_shape = (1,) + style_image.shape

    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        _holder = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        net = _vggnet(vgg_path, _holder)
        style_pre = np.array([style_image])
        for _id in STYLE_LAYERS:
            gram_layer = gram_matrix(net[_id])
            gram = gram_layer.eval(feed_dict={_holder:style_pre})
            style_grams[_id] = gram

    return style_grams


class Style(object):

    def __init__(self, content_images, style_image, content_weight, style_weight, denoise_weight,
                 vgg_path, ck_dir, test_image, test_out_dir, log_path, batch_size=4, alpha=1e-3):
        self.ct_images = content_images
        self.st_image = style_image
        self.test_image = test_image

        self.ct_weight = content_weight
        self.st_weight = style_weight
        self.de_weight = denoise_weight

        # directory paths
        self.vgg_path = vgg_path
        self.ck_dir = ck_dir
        self.test_out_dir = test_out_dir
        self.log_path = log_path

        # parameters
        self.batch_size = batch_size
        self.alpha = alpha

        # shape of batch content
        self.bc_shape = batch_content_shape = (batch_size, 256, 256, 3)
        tails = len(content_images) % batch_size
        if tails != 0:
            self.ct_images = self.ct_images[:-tails]

        assert len(self.st_image.shape) == 3 and len(self.test_image.shape) == 3
        assert self.test_image.shape[0] == 256 and self.test_image.shape[1] == 256 and self.test_image.shape[2] == 3

    def create_content_loss(self):
        x_content = tf.placeholder(tf.float32, shape=self.bc_shape, name="X_content")
        content_net = _vggnet(self.vgg_path, x_content)

        _genr_net = genr_net.net(x_content / 255.0)
        genrAndvgg_net = _vggnet(self.vgg_path, _genr_net)

        loss = self.ct_weight * tf.nn.l2_loss(genrAndvgg_net[CONTENT_LAYER] - content_net[CONTENT_LAYER]) / self.batch_size
        return loss, genrAndvgg_net, _genr_net, x_content

    def create_style_loss(self, genrAndvgg_net, style_target_grams):
        # with tf.Graph().as_default():
        losses = []
        for _id in STYLE_LAYERS:
            gram = gram_matrix(genrAndvgg_net[_id])
            losses.append(tf.nn.l2_loss(gram - style_target_grams[_id]))
        loss = self.st_weight * reduce(lambda x, y: x + y, losses) / self.batch_size
        return loss

    def create_denoise_loss(self, generated_net):
        loss = tf.reduce_sum(tf.abs(generated_net[:, 1:, :, :] - generated_net[:, :self.bc_shape[1]-1, :, :])) + \
               tf.reduce_sum(tf.abs(generated_net[:, :, 1:, :] - generated_net[:, :, :self.bc_shape[2]-1, :]))
        loss = self.de_weight * loss / self.batch_size
        return loss

    def train(self):
        target_grams = _target_grams(self.vgg_path, self.st_image)

        batch_test_image = [self.test_image] * self.batch_size
        batch_test_image = np.array(batch_test_image)

        with tf.Graph().as_default(), tf.Session() as sess:
            content_loss, genrAndvgg_net, _genr_net, x_content = self.create_content_loss()
            style_loss = self.create_style_loss(genrAndvgg_net, target_grams)
            denoise_loss = self.create_denoise_loss(_genr_net)

            total_loss = content_loss + style_loss + denoise_loss
            print("total_loss type=", type(total_loss))
            my_optimizer = tf.train.AdamOptimizer(self.alpha).minimize(total_loss)
            sess.run(tf.global_variables_initializer())

            numOfExamples = len(self.ct_images)
            cur_iter = 0
            st_idx, ed_idx = 0, self.batch_size
            while ed_idx < numOfExamples:
                batch_data = self.ct_images[st_idx: ed_idx]
                x_batch = list(map(lambda x: load_image(x, (256, 256)).astype(np.float32), batch_data))
                x_batch = np.array(x_batch)
                my_optimizer.run(feed_dict={x_content: x_batch})

                st_idx += self.batch_size
                ed_idx += self.batch_size

                if not cur_iter % SAVE_PERIOD:
                    file_name = "model_{}.ckpt".format(cur_iter)
                    tf.train.Saver().save(sess, os.path.join(self.ck_dir, file_name))

                    # use test image to do pre-propagation
                    pre_prop = [content_loss, style_loss, denoise_loss, total_loss, _genr_net]
                    ret = sess.run(pre_prop, feed_dict={x_content: batch_test_image})

                    save_image(os.path.join(self.test_out_dir, "{}.png".format(cur_iter)), ret[4][0])
                    with open(self.log_path, 'a') as f:
                        print("{}   content_loss:{:.3f} style_loss:{:.3f}  denoise_loss:{:.3f} total_loss:{:.3f}"
                              .format(cur_iter, ret[0], ret[1], ret[2], ret[3]))
                        f.write("{}   content_loss:{:.3f} style_loss:{:.3f}  denoise_loss:{:.3f} total_loss:{:.3f}\n"
                                .format(cur_iter, ret[0], ret[1], ret[2], ret[3]))

                cur_iter += 1