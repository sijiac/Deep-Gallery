import sys, os, pdb
import genr_net
import numpy as np
import vgg
import scipy.misc
import tensorflow as tf
from tools import load_image, save_image

DEVICE_ID = '/gpu:0'


def transfer(content_image, output_path, model_path):
    """

    :param content_image: height * weight * channel
    :param output_path:
    :param model_path:
    :param batch_size:
    :return:
    """

    if not os.path.isdir(output_path):
        raise ValueError(output_path + " doesn't exist.")
    if not os.path.isfile(model_path):
        raise ValueError(model_path + " doesn't exist.")

    g = tf.Graph()
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(DEVICE_ID), tf.Session(config=tf_config) as sess:
        batch_content_shape = (1, ) + content_image.shape

        X_content = tf.placeholder(tf.float32, shape=batch_content_shape, name='transfer_X_content')
        _genr = genr_net.net(X_content)

        tf.train.Saver.restore(sess, model_path)

        genrd_image = sess.run(_genr, feed_dict={X_content:content_image})[0]
        save_image(output_path, genrd_image)
