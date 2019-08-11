"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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


def freeze(model):
    """ Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    VGG13_LAYERS = ['block1_conv2', 'block2_conv2']
    VGG16_LAYERS = ['block3_conv3', 'block4_conv3', 'block5_conv3']
    VGG19_LAYERS = ['block3_conv4', 'block4_conv4', 'block5_conv4']


    for layer in model.layers:
    	layer.trainable = False
    	if layer.name is in VGG13_LAYERS:
        	layer.trainable = True
    return model
