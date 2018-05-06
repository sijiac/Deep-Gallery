import sys, os, shutil
sys.path.insert(0, "src")
from tools import load_image
from argparse import ArgumentParser
from style import Style
import neural_transfer
import numpy as np, scipy.misc
import uuid

UUID = uuid.uuid4().hex[:8]
UUID_PRFIX = './' + UUID
CHECKPOINT_DIR = UUID_PRFIX + '/ck_dir'
OUTPUT_DIR = UUID_PRFIX + '/output'
DATA_PATH = './val2017'
VGG_PATH = './imagenet-vgg-verydeep-19.mat'

# build_model() needed
STYLE_PATH = './wave.jpg'
TEST_PATH = './small_artist.jpeg'
LOG_PATH = UUID_PRFIX + '/log.txt'

# tranfer() needed
CONTENT_PATH = './input/small_artist.jpeg'
MODEL_PATH = './8ca14295/ck_dir/model_2500.ckpt'
GENRD_PATH = './output/small_artist.jpeg'


def build_model():

    if os.path.exists(UUID_PRFIX):
        shutil.rmtree(UUID_PRFIX)
    os.makedirs(UUID_PRFIX)
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if os.path.isfile(LOG_PATH):
        os.remove(LOG_PATH)
    with open(LOG_PATH, 'w') as f:
        pass

    # Check directory path
    if not os.path.isdir(CHECKPOINT_DIR):
        raise ValueError(CHECKPOINT_DIR + " doesn't exist.")
    if not os.path.isdir(OUTPUT_DIR):
        raise ValueError(OUTPUT_DIR + " doesn't exist.")
    if not os.path.isdir(DATA_PATH):
        raise ValueError(DATA_PATH + " doesn't exist.")

    # Check file path
    if not os.path.exists(VGG_PATH):
        raise ValueError(VGG_PATH + " doesn't exist.")
    if not os.path.exists(VGG_PATH):
        raise ValueError(STYLE_PATH + " doesn't exist.")
    if not os.path.exists(TEST_PATH):
        raise ValueError(TEST_PATH + " doesn't exist.")

    style_image = load_image(STYLE_PATH)
    test_image = load_image(TEST_PATH, (256,256))

    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    files = []
    for (dirpath, dirnames, filenames) in os.walk(DATA_PATH):
        files.extend(filenames)
        break
    files = [os.path.join(DATA_PATH, x) for x in files]

    new_style = Style(content_images=files,
                       style_image=style_image,
                       content_weight=7.5,
                       style_weight=100,
                       denoise_weight=100,
                       vgg_path=VGG_PATH,
                       ck_dir=CHECKPOINT_DIR,
                       test_image=test_image,
                       test_out_dir=OUTPUT_DIR,
                       log_path=LOG_PATH,
                       batch_size=32,
                       alpha=1e-3)
    new_style.train()


def transfer(reserve=False):
    if not os.path.isfile(CONTENT_PATH):
        raise ValueError(CONTENT_PATH + " doesn't exist.")

    content_image = load_image(CONTENT_PATH)
    neural_transfer.transfer(content_image=content_image,
                             output_path=GENRD_PATH,
                             model_path=MODEL_PATH,
                             reserve_color=reserve)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--function',
                        type=str,
                        dest='func',
                        help="transfer or train",
                        metavar='FUNC',
                        required=True)
    parser.add_argument('--reserve',
                        dest='reserve',
                        action='store_true',
                        help="means reserve original color",
                        default=False)
    opts = parser.parse_args()

    if opts.func == 'transfer':
        transfer(reserve=opts.reserve)
    elif opts.func == 'train':
        print("uuid: {}".format(UUID))
        build_model()
    else:
        raise ValueError("--function " + opts.func + " not found!")