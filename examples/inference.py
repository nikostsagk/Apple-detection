# show images inline
# %matplotlib inline

# automatically reload modules when they have changed
# %load_ext autoreload
# %autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse
import ntpath

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

#def parse_args(args):
#	""" Parse the arguments. """
#	parser  = argparse.ArgumentParser(description='Infer an image and visualize detections.')
#	parser.add_argument('--model', help='The path to model weights', dest='model', default=None)
#	parser.add_argument('--convert-model', help='Convert the model to inference model', action='store_true')
#	parser.add_argument('--image', help='The path to the image to infer', default=None)
#
#	return paser.parse_args(args)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())



# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#if args.model is not None:
#	model_path = args.model
#else:
#	print('Model to path is missing.ß')

# load retinanet model
model = models.load_model('../results/test_02/after_10_init_epochs/vgg16_csv_08.h5', backbone_name='vgg16')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)
print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'apple'}


# load image
image_name = '20130320T012856.619229_62.png'
image = read_image_bgr(image_name)

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
plt.savefig('{}.pdf'.format(image_name, bbox_inches='tight'))

