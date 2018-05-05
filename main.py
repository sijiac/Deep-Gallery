import sys, os, shutil
sys.path.append('src')
from tools import load_image
from argparse import ArgumentParser
import optimizer
import neural_transfer
import numpy as np, scipy.misc
import uuid


UUID_PRFIX = uuid.uuid4().hex[:8]
CHECKPOINT_DIR = './' + UUID_PRFIX + '_ck'
OUTPUT_DIR = './' + UUID_PRFIX + '_output'
DATA_PATH = './train2014'
VGG_PATH = './imagenet-vgg-verydeep-19.mat'


STYLE_PATH = './wave.jpeg'
TEST_PATH = './artist.jpeg'


CONTENT_PATH = './input/small_artist.jpeg'
MODEL_PATH = './wave.ckt'
GENRD_PATH = './output/small_artist.jpeg'


def build_model():

    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    os.mkdirs(CHECKPOINT_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdirs(OUTPUT_DIR)

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

    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    files = []
    for (dirpath, dirnames, filenames) in os.walk(DATA_PATH):
        files.extend(filenames)
        break
    files = [os.path.join(DATA_PATH, x) for x in files]

    optimizer.optimize(content_images=files,
                       style_image=style_image,
                       content_weight=7.5,
                       style_weight=100,
                       denoise_weight=200,
                       vgg_path=VGG_PATH,
                       batch_size=4,
                       alpha=1e-3,
                       ck_dir=CHECKPOINT_DIR)


def transfer():
    if not os.path.isfile(CONTENT_PATH):
        raise ValueError(CONTENT_PATH + "")

    content_image = load_image(CONTENT_PATH)
    neural_transfer.transfer(content_image=content_image,
                             output_path=OUTPUT_DIR,
                             model_path=MODEL_PATH)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--function',
                        type=str,
                        dest='func',
                        help="transfer or train",
                        metavar='FUNC',
                        required=True)
    opts = parser.parse_args()

    if opts.func == 'transfer':
        transfer()
    elif opts.func == 'train':
        build_model()
    else:
        raise ValueError("--function " + opts.func + " not found!")