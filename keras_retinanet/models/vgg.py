"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import keras
from keras.utils import get_file

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class VGGBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return vgg_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg16' or self.backbone == 'vgg13' or self.backbone == 'vgg11' :
            resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif self.backbone == 'vgg19':
            resource = keras.applications.vgg19.vgg19.WEIGHTS_PATH_NO_TOP
            checksum = '253f8cb515780f3b799900260a226db6'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'vgg13', 'vgg11']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs, mode='caffe'):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode=mode)


def vgg_retinanet(num_classes, backbone='vgg16', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
    else:
        inputs = keras.layers.Input(shape=inputs)

    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'vgg19':
        vgg = keras.applications.VGG19(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'vgg13':
        vgg = vgg13(input_tensor=inputs, weights=None)
    elif backbone == 'vgg11':
        vgg = vgg11(input_tensor=inputs, weights=None)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_names = ["block3_pool", "block4_pool", "block5_pool"]
    layer_outputs = [vgg.get_layer(name).get_output_at(0) for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)


def vgg13(input_tensor=None, weights=None):
    """
    Returns
        A lighter version of VGG16.
    """
    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return keras.Model(input_tensor, x, name='VGG13')

def vgg11(input_tensor=None, weights=None):
    """
    Returns
        A lighter version of VGG16.
    """
    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return keras.Model(input_tensor, x, name='VGG11')

