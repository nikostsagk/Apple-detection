Dataset properties:

Apple data image properties:
# Items: 1120
# Properties: PNG image data, 308 x 202, 8-bit/color RGB, non-interlaced
# Annotations: Circles: #item,c-x,c-y,radius,label
# Rectangular annotations: #item, x0, y0, x1, y1, class
# Classes: class, index
Dataset Structure:

keras-retinanet
└─ data
   ├── annotations
   ├── rectangular_annotations
   ├── classes
   ├── images
   ├── labelmap.json
   ├── readme.txt
   └── sets
       ├── all.txt
       ├── test.txt
       ├── train.txt
       ├── train_val.txt
       └── val.txt

Dataset info:
-The images contained within are random crops extracted from data otherwise densely collected over the farms. The image names are typically appended with the ij co-ordinate or image sections from which the particular crop was extracted from. Refer to [1] for further information on data gathering.
-The Annotations folder contains corresponding .csv files with the same names as the images, with fruit annotation information per line.
-Rectangular annotation folder contains the circular annotations converted to rectangular. Boxes from objects that exceed image spatial dimensions, are fixed on the image boundaries.
-The sets folder contains the dataset splits (list of image names) as used in [1] for training, testing and validation.

[1] Bargoti, S., & Underwood, J. (2016). Deep Fruit Detection in Orchards. arXiv preprint arXiv:1610.03677. [Submitted to ICRA (2017)]


