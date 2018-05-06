# Deep-Gallery

Deep Gallery: A Convolutional Neural Network Algorithm of Artistic Style Transfer

## Create conda environment

This project requires:

- numpy, scipy
- tensorflow
- cv2
- python2.7

To create an Deep-Gallery conda environment:

```
conda env create -f conda_env.yml
```

## Build a style model

1. Download pre-trained vgg19 and coco train2014 dataset:

```
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
```

Make sure `imagenet-vgg-verydeep-19.mat` and `train2014` are under in `Deep-Gallery/`.

2. Set parameters in main.py:

```
STYLE_PATH = './wave.jpg'
TEST_PATH = './artist.jpg'
```

3. Start to build a model:

```
python main.py --function train
```

## Transfer a picture

1. Set parameters in main.py:

```
CONTENT_PATH = './input/artist.jpg'
MODEL_PATH = './8ca14295/ck_dir/model_2500.ckpt'
GENRD_PATH = './output/artist.jpg'
```

2. Start to transfer a picture:

```
python main.py --function transfer
```

## Transfer with perserving original data

```
python main.py --function transfer --reserve
```


## Thanks

This the last project assignment in 10701. As the saying goes, *Keep calm and trust the process*. All assigments in this class are very struggle, but after mindful thinking and continuous trials, we have learnt a lot. So thank all of the faculty members, this is the best Machine Learning courses we have token in CMU!