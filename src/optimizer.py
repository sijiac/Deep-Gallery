import tensorflow as tf
import vgg
import generate_net
import functools
import numpy as np
from tools import load_image

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

def _vggnet(data_path, input_image):
    weights, mean_pixel = vgg.load_net(data_path)
    image = vgg.preprocess(input_image, mean_pixel)
    net = vgg.net_preloaded(weights, input_image)
    return net, mean_pixel

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def create_content_loss(sess, batch_content_shape, vgg_path, batch_size):
    with tf.Graph().as_default():
        X_content = tf.placeholder(tf.float32, shape=batch_content_shape, name="X_content")
        content_net = _vggnet(vgg_path, X_content)

        X_generated = generate_net.net(X_content/255.0)
        generated_net = _vggnet(vgg_path, X_generated)

        loss = tf.nn.l2_loss(generate_net[CONTENT_LAYER] - content_net[CONTENT_LAYER])
        content_size = _tensor_size(content_net[CONTENT_LAYER]) * batch_size
    return loss, generated_net, X_content

def create_style_loss(sess, generated_net, style_target_grams, batch_size):
    with tf.Graph().as_default():
        losses = []
        for id in STYLE_LAYERS:
            layer = generated_net[id]
            numOfBatch, h, w, numOFilter = layer.get_shape()
            single_size = h * w * numOFilter
            fs = tf.reshape(layer, (numOfBatch, h * w, numOFilter))
            fsT = tf.transpose(fs, perm=[0, 2, 1])
            gram = tf.matmul(fsT, fs) / single_size
            losses.append(tf.nn.l2_loss(gram - style_target_grams[id]))
    loss = reduce(lambda x, y: x + y, losses) / batch_size
    return loss


def create_denoise_loss(sess, generated_net, batch_content_shape, batch_size):
    Y_size = _tensor_size(generated_net[:, 1:, :, :])
    X_size = _tensor_size(generated_net[:, :, 1:, :])
    Y = tf.nn.l2_loss(generated_net[:, 1:, :, :] - generated_net[:, :, batch_content_shape[1]-1:, :])
    X = tf.nn.l2_loss(generated_net[:, :, 1:, :] - generated_net[:, :, :batch_content_shape[2]-1, :])

    loss = (X / X_size + Y / Y_size) / batch_size
    return loss


def _target_grams(vgg_path, style_image):
    style_grams = {}
    style_shape = (1,) + style_image.shape

    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        net = _vggnet(vgg_path, style_image)
        style_pre = np.array([style_image])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_grams[layer] = gram

    return style_grams


def optimizer(content_images, style_image, content_weight, style_weight, denoise_weight,
              vgg_path, numOfEpochs=2, batch_size=4, alpha=1e-3):

    mod = len(content_images) % batch_size
    if mod > 0:
        content_images = content_images[:-mod]

    batch_content_shape = (batch_size, 256, 256, 3)

    target_grams = _target_grams(vgg_path, style_image)

    with tf.Graph().as_default(), tf.Session() as sess:
        content_loss, genrd_net, X_content = create_content_loss(sess, batch_content_shape, vgg_path, batch_size)
        style_loss = create_style_loss(sess, genrd_net, batch_size)
        denoise_loss = create_denoise_loss(sess, genrd_net, batch_content_shape, batch_size)

        total_loss = content_weight * content_loss + style_weight * style_loss + denoise_loss * denoise_loss
        my_optimizer = tf.train.AdamOptimizer(alpha).minimize(total_loss)
        sess.run(tf.global_variables_initializer())

        numOfExamples = len(content_images)
        cur_iter = 0
        st_idx, ed_idx = 0, batch_size
        while ed_idx < numOfExamples:
            batch_data = content_images[st_idx: ed_idx]
            X_batch = list(map(lambda x: load_image(x, (256,256).astype(np.float32)), batch_size))
            X_batch = np.array(X_batch)
            my_optimizer.run(feed_dict={X_content:X_batch})
            cur_iter += 1