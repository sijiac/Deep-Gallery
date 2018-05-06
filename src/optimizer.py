import tensorflow as tf
import vgg
import genr_net
import functools
import numpy as np
import os
from tools import load_image, save_image

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

SAVE_PERIOD = 5


def _vggnet(data_path, input_image):
    weights, mean_pixel = vgg.load_net(data_path)
    image = vgg.preprocess(input_image, mean_pixel)
    net = vgg.net_preloaded(weights, image)
    return net


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def create_content_loss(sess, batch_content_shape, vgg_path, batch_size):
    # with tf.Graph().as_default():
    x_content = tf.placeholder(tf.float32, shape=batch_content_shape, name="X_content")
    content_net = _vggnet(vgg_path, x_content)

    _genr_net = genr_net.net(x_content/255.0)
    genrAndvgg_net = _vggnet(vgg_path, _genr_net)

    loss = tf.nn.l2_loss(genrAndvgg_net[CONTENT_LAYER] - content_net[CONTENT_LAYER]) / batch_size
    content_size = _tensor_size(content_net[CONTENT_LAYER]) * batch_size
    return loss, genrAndvgg_net, _genr_net, x_content


def create_style_loss(sess, genrAndvgg_net, style_target_grams, batch_size):
    # with tf.Graph().as_default():
    losses = []
    for _id in STYLE_LAYERS:
        layer = genrAndvgg_net[_id]
        tensorShape = layer.get_shape()
        numOfBatch, h, w, numOfFilter = (tensorShape[0].value,
                                         tensorShape[1].value,
                                         tensorShape[2].value,
                                         tensorShape[3].value)
        single_size = h * w * numOfFilter
        fs = tf.reshape(layer, (numOfBatch, h * w, numOfFilter))
        fsT = tf.transpose(fs, perm=[0, 2, 1])
        gram = tf.matmul(fsT, fs) / single_size
        losses.append(tf.nn.l2_loss(gram - style_target_grams[_id]))
    loss = reduce(lambda x, y: x + y, losses) / batch_size
    return loss


def create_denoise_loss(sess, generated_net, batch_content_shape, batch_size):
    Y_size = _tensor_size(generated_net[:, 1:, :, :])
    X_size = _tensor_size(generated_net[:, :, 1:, :])
    Y = tf.nn.l2_loss(generated_net[:, 1:, :, :] - generated_net[:, :batch_content_shape[1]-1, :, :])
    X = tf.nn.l2_loss(generated_net[:, :, 1:, :] - generated_net[:, :, :batch_content_shape[2]-1, :])

    loss = (X / X_size + Y / Y_size) / batch_size
    return loss


def _target_grams(vgg_path, style_image):
    style_grams = {}
    style_shape = (1,) + style_image.shape

    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        _holder = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        net = _vggnet(vgg_path, _holder)
        style_pre = np.array([style_image])
        for _id in STYLE_LAYERS:
            features = net[_id].eval(feed_dict={_holder:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_grams[_id] = gram

    return style_grams


def optimize(content_images, style_image, content_weight, style_weight, denoise_weight,
              vgg_path, ck_dir, test_image, test_out_dir, log_path, batch_size=4, alpha=1e-3):

    assert len(style_image.shape) == 3 and len(test_image.shape) == 3
    assert test_image.shape[0] == 256 and test_image.shape[1] == 256 and test_image.shape[2] == 3

    mod = len(content_images) % batch_size
    if mod > 0:
        content_images = content_images[:-mod]

    batch_content_shape = (batch_size, 256, 256, 3)

    target_grams = _target_grams(vgg_path, style_image)

    batch_test_image = [test_image] * batch_size
    batch_test_image = np.array(batch_test_image)

    with tf.Graph().as_default(), tf.Session() as sess:
        content_loss, genrAndvgg_net, _genr_net, x_content = create_content_loss(sess, batch_content_shape, vgg_path, batch_size)
        style_loss = create_style_loss(sess, genrAndvgg_net, target_grams, batch_size)
        denoise_loss = create_denoise_loss(sess, _genr_net, batch_content_shape, batch_size)

        total_loss = content_weight * content_loss + style_weight * style_loss + denoise_weight * denoise_loss
        print("total_loss type=", type(total_loss))
        my_optimizer = tf.train.AdamOptimizer(alpha).minimize(total_loss)
        sess.run(tf.global_variables_initializer())

        numOfExamples = len(content_images)
        cur_iter = 0
        st_idx, ed_idx = 0, batch_size
        while ed_idx < numOfExamples:
            batch_data = content_images[st_idx: ed_idx]
            x_batch = list(map(lambda x: load_image(x, (256,256)).astype(np.float32), batch_data))
            x_batch = np.array(x_batch)
            my_optimizer.run(feed_dict={x_content:x_batch})

            st_idx += batch_size
            ed_idx += batch_size

            if not cur_iter % SAVE_PERIOD:
                file_name = "model_{}.ckpt".format(cur_iter)
                tf.train.Saver().save(sess, os.path.join(ck_dir, file_name))

                # use test image to do pre-propagation
                pre_prop = [content_loss, style_loss, denoise_loss, total_loss, _genr_net]
                ret = sess.run(pre_prop, feed_dict={x_content:batch_test_image})

                save_image(os.path.join(test_out_dir, "{}.png".format(cur_iter)), ret[4][0])
                with open(log_path, 'a') as f:
                    print("{}   content_loss:{:.3f} style_loss:{:.3f}  denoise_loss:{:.3f} total_loss:{:.3f}"
                          .format(cur_iter, ret[0], ret[1], ret[2], ret[3]))
                    f.write("{}   content_loss:{:.3f} style_loss:{:.3f}  denoise_loss:{:.3f} total_loss:{:.3f}\n"
                          .format(cur_iter, ret[0], ret[1], ret[2], ret[3]))

            cur_iter += 1