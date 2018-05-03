import tensorflow as tf
import vgg

def _vggnet(data_path, input_image):
    weights, mean_pixel = vgg.load_net(data_path)
    image = vgg.preprocess(input_image, mean_pixel)
    net = vgg.net_preloaded(weights, input_image)
    return net, mean_pixel


def content_loss(sess, net, content_image, content_layers):
    with tf.Graph().as_default():
