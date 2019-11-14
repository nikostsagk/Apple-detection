# Improving Apple Detection and Counting Using RetinaNet

This work aims to investigate the apple detection problem through the deployment of the RetinaNet object detection framework in conjunction with the VGG architecture. Following hyper-parameters’ optimisation, the performance scaling with the backbone’s network depth is examined through four different proposed deployments for the side-network. Analysis of the relationship between performance and training size establishes that 10 samples are enough to achieve adequate performance, while 200 samples are enough to achieve state-of-the-art performance. Moreover, a novel lightweight model is proposed that achieves an F1-score of 0.908 and inference time of nearly 70FPS. These results outperform previous state-of-the-art models in both performance and detection rates. Finally, the results are discussed regarding the model’s limitations, and insights for future work are provided.

# Dataset
The dataset used for this project is the [ACFR dataset](http://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/) and can be downloaded [here](https://github.com/nikostsagk/Apple-detection/releases/download/dataset/Archive.zip). It consists of images of three different fruits (apples, mangoes & almonds), but only the apple set was used. The original train/val/test set was preserved in order to make comparisons with previous studies.

The dataset contains 1120 308x202 samples with apples. The annotations are given in `#item, x0, y0, x1, y1, class` format (circular) and can be converted to square with the `examples/convert_annotations.py` file. More info in the `readme.txt` file in the dataset folder.

<div class="row">
  <div class="column">
    <img src="https://github.com/nikostsagk/Apple-detection/blob/master/examples/20130320T004433.135911.Cam6_52.png" alt="example 1" width="180">
  </div>
  <div class="column">
    <img src="https://github.com/nikostsagk/Apple-detection/blob/master/examples/20130320T004433.707425.Cam6_54.png" alt="example 2" width="180">
  </div>
  <div class="column">
    <img src="https://github.com/nikostsagk/Apple-detection/blob/master/examples/20130320T004443.802883.Cam6_42.png" alt="example 3" width="180">
  </div>
   <div class="column">
    <img src="https://github.com/nikostsagk/Apple-detection/blob/master/examples/20130320T004445.136158.Cam6_23.png" alt="example 4" width="180">
  </div>
</div>

# Architectures

The repository consists of four side-network architectures, each one implemented on the four repo branches.

* `master` : The original side-network architecture.

<p align="center">
  <img src="https://github.com/nikostsagk/msc-project-report/blob/master/figures/ch4/original.jpg" width="520">
</p>

* `retinanet_p3p4p5` : The original side-network architecture without the strided convolutional filters right after the VGG network.

<p align="center">
  <img src="https://github.com/nikostsagk/msc-project-report/blob/master/figures/ch4/retinanet_p3p4p5.jpg" width="500">
</p>

* `retinanet_ci_multiclassifiers` : The `retinanet_p3p4p5` implementation with separate classification regression heads for the predictions.

<p align="center">
  <img src="https://github.com/nikostsagk/msc-project-report/blob/master/figures/ch4/retinanet_ci_multi.jpg" width="500">
</p>

* `retinanet_ci` : A lightweight implementation where common classification and regression heads make predictions right after the C<sub>i</sub> reduced blocks, without the upsampling-merging technique.

<p align="center">
  <img src="https://github.com/nikostsagk/msc-project-report/blob/master/figures/ch4/retinanet_ci_reduced.jpg" width="500">
</p>

# Installation
Clone the repo and follow the instructions in: [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)

# Sources
1) [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
2) [Martin Zlocha](https://github.com/martinzlocha/anchor-optimization)
3) [ACFR FRUIT DATASET](http://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/)
