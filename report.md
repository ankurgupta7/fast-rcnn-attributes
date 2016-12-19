# Learnings
## Deep Neural Networks: Resources
Since I had to study Deep Neural Networks, I followed a few resources on the internet. They were very beneficial

* `https://www.udacity.com/course/deep-learning--ud730`
* `http://cs231n.stanford.edu/`
* `https://www.coursera.org/learn/machine-learning`
* `http://deeplearning.net/tutorial/logreg.html#logreg`
* `http://advancedmachinelearning.weebly.com/slides.html`
* Research Papers by YL Cun, G Hinton, R Girshick

## Fast-RCNN
The novelty in this technique is to use pre-computed (using selective search\cite{selsearch}) ROI for every training image and learn the bounding boxes by regression on their coordinates. SPPNet\cite{sppnet} also achieved the result by the same method but Girshick etal \cite{fastrcnn} employed clever techniques to convert the pooling pyramid layer into a differentiable set of layers. Doing this unfroze this layers and back propagation could learn these weights too. 
<br/> Fast-RCNN takes as input
* training images
* the corresponding selective-search bounding boxes
* annotations
* train-test split as input. 

The train.prototxt and test.prototxt files must very carefully modified to adapt to a new dataset. 

A very good resource on understanding this flow can be found at \cite{`https://github.com/zeyuanxy/fast-rcnn/blob/master/help/train/README.md`}. I have often gone back to this page, whenever I have plugged in new datasets.

## Multi Attribute Fast-RCNN
* In terms of the architecture of the network, the major change is in Loss calculation as compared to Fast-RCNN. 
* We are using SigmoidLossLayer instead of softmaxloss layer. This is because, now we want the probability of each class independent of probability of other classes. 
* Softmax essentially mimics a probabilistic distribution where sum of probabilities of all classes sum to 1. This is not the suitable behavior for us. Instead we require that the output should indicate the network's "confidence" in predicting multiple labels for the same bounding box.
## Importance of Runtime Debugging 
* This codebase was built on top of Fast-RCNN code. The original code is written to enhance performance. Therefore, at places it may seem convoluted and the variables are not always aptly named. Also, debugging has helped me immensely in finding the intent of some code snippets by constantly looking at values of variables at runtime and see how these values evolve. Please see Productivity Hacks Section below for more details. 

# Challenges
* Getting familiar with Deep Learning, Python, PyCaffe, Caffe-C++
* Understanding Loss Calculations in SmoothL1lossLayer in Fast-RCNN
* Learning about different loss layers and finding the most suitable choice for our problem
* Understanding the codebase
* Plugging in new attributes dataset
* Figuring out the use of roidb and its relationship with minibatch class
* Dimension mismatch due to multi-label classification
* intelligently using cfg(configure) class
* Debugging Python code at Runtime (Productivity Impact)
* Debugging Hidden and Loss Layers at Caffe-C++ level <br/>
* Changing logic Testing with new image: We can no more print all the bounding box with the maximum class likelihood. The logic now changes to finding intersection of bounding boxes with all desired class probabilities above a certain threshold and taking their intersection.

# Knowledge Transfer
## GitHub: 
The repository is hosted at GitHub and can be cloned at \cite{`git@github.com:ankurgupta7/fast-rcnn-attributes.git`}
Fast-RCNN is also using a custom build of caffe. Girshick et al have added their custom loss layer for bounding box regression, roi pooling layer and dropout scaling at test time. the SHA1 of relevant commit in caffe-rcnn log is 0dcd397b29507b8314e252e850518c5695efbb83. Any changes that I made are on top of the mentioned commit.

It may be noted that I have edited a few files(below) in the c++ caffe src and therefore I have included caffe-fast-rcnn directory as part of my repository. It was initially submoduled (which is the correct way) but it became difficult to keep updating and raising Pull Request with every change. Future student can folk the caffe-Berkley repo from commit 33f2445b47fddd260ac59901c0920f2bb671d579 and host it on his/her own GitHub and submodule caffe-fast-rcnn to that repo. I have not taken that step. 

## Source Files Changed
the entire file list of changed files is under Appendix 1. However, below is a list of the most important files that changed. I have tried that wherever I am changing caffe c++ src files, I am adding "experimental debug- ankur" in comments so that those lines can be easily "grep-ed" by future student not comfortable with git workflow. 

* `.gitmodules`  <br/> 
 removed the caffe-fast-rcnn from submodule
* `caffe-fast-rcnn/CMakeLists.txt`  <br/> 
 changed this to load cudnn locally and not from system wide installation. because there are computability issues with cudnn>4.x
* `caffe-fast-rcnn/Makefile.config.example`  <br/> 
* `src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp`  <br/> 
Added relevant stdout redirections statements to easily debug change is layer shapes. also, these statements help in debugging what is happening in the loss layers in runtime directly from c++. 
* `src/caffe/layers/smooth_L1_loss_layer.cpp`  <br/> 
Added relevant stdout redirections statements to easily debug change is layer shapes for the bounding box loss calculation. also, these statements help in debugging what is happening in the loss layers in runtime directly from c++. 
It may be noted that this layer does not have a cpu implementation.
* `src/caffe/layers/loss_layer.cpp`  <br/> 
This is the parent class of all loss layers. had to add stdout redirection here because sometimes, reshape of parent loss_layer class broke due to a few changes. 
* `---other caffe files`  <br/> 
* `data/attributes/Annotations/00001.txt - 04013.txt`  <br/> 
These are generated annotations from People Attributes Dataset (Dataset section below). This dataset is suitable for multilabel classification.
* `data/attributes/Annotations_old/01.txt - 07.txt`  <br/> 
Debug Annotations. Just a set of 7 images with full image as the bounding box. and hand-made true labels
* `data/attributes/Images/01 - 07.jpg`  <br/> 
Corresponding image data files for debugging.
* `data/attributes/ImageSets/train_people_all_valid.txt`  <br/> 
In the dataset we are using, not all images have valid annotations. As a result naive list of all images will crash the network and load annotation routines in our code. This txt file contains the list of all images with "valid" annotations.
* `data/attributes/Images/people_labels_parser_fcnn.py`  <br/> 
This script produces annotations in the desired format of multiple labels that we are using in our implementation. 
* `data/attributes/ImageSets/make_traintxt.py`  <br/> 
This script produces train.txt for all valid annotations.
* `models/attributes/vgg/fast_rcnn/solver.prototxt`  <br/> 
solver configurations for Multiclass FRCNN. We have reduced the learning rate to 10e-7 and increased the interations to 2e6 for our task. 
* `models/attributes/vgg/fast_rcnn/test.prototxt`  <br/> 
Network architecture for testing pipeline. The change is in last layer. From SoftMax to Sigmoid layer
* `models/attributes/vgg/fast_rcnn/train.prototxt`  <br/> 
Network architecture for training. The change is in loss layer. From SoftMaxLoss to SigmoidCrossEntropyLoss layer.

* `tools/test_net.py`  <br/> 
The only change in this file is to change the path of trained_model to match the directory structure.
* `tools/train_net.py`  <br/> 
train_net has been modified to accomodate PyCharm integration for easier debugging. There are some argument parameters hard-coded. This is primarily because I wanted to test different arguments and it was much easier this ways. arguments can be moved to configurations in PyCharm. 

* `experiments/scripts/fast_rcnn.sh`  <br/> 
added support for attributes dataset in the command line invocation of fast-rcnn.
* `lib/datasets/attributes.py`  <br/> 
The dataset specific class used to load annotations for attributes dataset in F-RCNN environment. this parses all the anotations file and converts them to 'boxes', 'gt_classes', 'overlaps' objects in ROI-DB used in FRCNN. It also defines the hot encoding representation of multiclass labels consumed by CrossEntropy Loss function. This class also holds the set of all possible labels. This class in important and its load_annotataion method has to be re-implemented for each dataset.
* `lib/roi_data_layer/layer.py`  <br/> 
Layer Python Data class. Here, I have changed the labels blob to comply with sigmoid loss layer. labels_sigmoid variable in get_minibatch method is the key change. other changes were made to accomodate the dimensional shape of this variable. 
There will be a few pdb.set_trace(). Ignore them. they were used to debug things on runtime. 
* `lib/roi_data_layer/minibatch.py`  <br/> 
This file was heavily changed to accomodate for change in shapes of gt_classes variable. But near the end of my independent research, I realized a new approach rendered all these changes useless. Therefore I have reverted all the changes in this file.
* `lib/roi_data_layer/roidb.py`  <br/> 
In the method _compute_targets, i have added a few checks to the variable ex_gt_overlaps so that if there are corrupt entries in annotations, it gets handled and the network does not break.
* `lib/datasets/factory.py`  <br/> 
abstract factory for abstracting the creation of dataset object based on dataset name. I have added support for attributes dataset. 
* `lib/fast_rcnn/test.py`  <br/> 
removed support for flipping images. the imdb helper with flipped images shuffles the order of test dataset. it became terribly difficult to debug because of that. Its left as a comment. Can be uncommented later.
* `lib/fast_rcnn/train.py`  <br/> 
removed support for flipping images. the imdb helper with flipped images shuffles the order of train dataset. it became terribly difficult to debug because of that. Its left as a comment. Can be uncommented later.

* `tools/demo.py`  <br/> 
We can no more print all the bounding box with the maximum class likelihood. The logic now changes to finding intersection of bounding boxes with all desired class probabilities above a certain threshold and taking their intersection. Earlier non-maximal suppression was applied to bounding boxes and the max probability of class per that bounding box as input. Now we need a heuristic to predict the best score for nms.
for testing the network with a new image, we cannot use the same calculations that came with Fast-RCNN. This is because
* `network-arch.png`  <br/> 
visual depiction of all network layers
* `lib/datasets/VOCdevkit-matlab-wrapper/get_voc_opts.m`  <br/> 
Part of Matlab eval script. Edited this to include attributes dataset.
* `lib/datasets/VOCdevkit-matlab-wrapper/voc_eval.m`  <br/> 
Part of Matlab eval script. Edited this to include attributes dataset.

## Source Diff
The entire diff can be viewed simply by `git diff 90e75082f087596f28173546cba615d41f0d38fe` <br/>
diff of important files described above carefully selected to include the minimal content is attached as Appendix 2

## Network Visualization
 caffe/python/draw_net.py <netprototxt_filename> <out_img_filename>
 This was very helpful to me in finding the position of labels layer, especially in faster-rcnn.

## Debugging on CPU: 
 In train_net.py: <br/> 
comment `line# 83: caffe.set_device(args.gpu_id)`
 Please note that smoothl1 layer is not implemented for cpu so we cannot test the full network. but is nevertheless useful to debug first few layers locally without nvidia graphic card
## build steps: cudnn and caffe build. 
Build the Cython modules
    ```
    cd $FRCN_ROOT/lib
    make
    ```
Build Caffe and pycaffe
    ```
    cd $FRCN_ROOT/caffe-fast-rcnn
    mkdir build
    cd build
    cmake ..
    # Here change the include path to $FRCN_ROOT/cudnn
     make -j8 && make pycaffe
    ```
## PyCharm Integration & installation on cluster
* `ssh -X address/of/cluster`
* Portable PyCharm can be setup without root permission.
* required python packages can be installed locally with `--user` flag in `pip` or use virtualenv
## VNC
* I used TigerVNC post tunneling vnc traffic on port 5901 locally. <br/> 
`ssh -L 5901:localhost:5901 /address/of/cluster`

## Using pdb to debug
* pdb.set_trace for command line debugging

# Future Work
## Make the predictions accurately
In the current state of the network, it predicts the exact probability for a particular label, across thousands of bounding boxes. For example, if selection search found 4000 boxes, and CONF_THRESHOLD is 0.2, then there can be about 1300 bounding boxes with the same probability of 0.24256681 of it being is_man class. This behavior is not expected and is in fact erroneous. 

Earlier, the network output only the expected probability of just the first label i.e. is_man and was throwing garbage for other labels. But I found a bug in my implementation that was passing the ground truth label of just the first label. I fixed it and now all the labels are getting some probability but it’s now low (~ 0.2) for all the labels.

## Decide on a way to evaluate the results

Currently, I am evaluating for a few images by hand. But this needs to change. The reason it has not changed so far, is that evaluation if Fast-RCNN is done with some MATLAB scripts with many network dependent parameters being fed independently. Like the list of all labels for the current dataset must be put in back again in MATLAB and similarly all the paths of model files, image files etc. has to be changed again MATLAB. <br/>
One of the ways to counter this is to move python port of evaluation from Faster-RCNN to this code and plug them together.
## shift to faster-rcnn if everything else works. 
If the network starts predicting the expected class labels, then this should be ported to work with Faster-RCNN. It should be straightforward. The labels blob in `minibatch.py` should be changed accordingly.
# Dataset
People Attributes Dataset developed by Lubomir Bourdev, Jitendra Malik as part of Poselets project at UC Berkley. They provide bounding boxes around people and multiple attributes per bounding box like is_male, has_trousers, has_jeans etc

Download Link: <br/> 
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/attributes_dataset.tgz

# Sample Result
<br/>
markdown
![Result](https://github.com/ankurgupta7/fast-rcnn-attributes/raw/master/results.png?raw=true)

There are multiple observations to be made about the result <br/>
* There are just three bounding boxes visible but the labels seems to be running in thousands. This is happening because the nonmaximal suppression is not able to suppress minima, precisely because there is no maxima. All the probabilities are exactly equal. This needs to be careflly looked into

* The probabilities are low (~0.2) for these detected classes.

* The results are actualy wrong. these bounding boxes do not localize the given attributes

<br/> It could be the case the dataset itself is not marked accurately. I have not looked into this possibility. 



# Appendix 1: All Files Changed

 * `.gitmodules`
 
 * `caffe-fast-rcnn/CMakeLists.txt`
 * `caffe-fast-rcnn/LICENSE`
 * `caffe-fast-rcnn/include/caffe/layer.hpp`
 * `caffe-fast-rcnn/include/caffe/layers/dropout_layer.hpp`
 * `caffe-fast-rcnn/include/caffe/layers/python_layer.hpp`
 * `caffe-fast-rcnn/python/caffe/__init__.py`
 * `caffe-fast-rcnn/python/caffe/_caffe.cpp`
 * `caffe-fast-rcnn/python/classify.py`
 * `caffe-fast-rcnn/src/caffe/layers/dropout_layer.cpp`
 * `caffe-fast-rcnn/src/caffe/layers/dropout_layer.cu`
 * `caffe-fast-rcnn/src/caffe/layers/lrn_layer.cpp`
 * `caffe-fast-rcnn/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp`
 * `caffe-fast-rcnn/src/caffe/proto/caffe.proto`
 
 * `cudnn/cuda/include/cudnn.h`
 * `cudnn/cuda/lib64/libcudnn.so.4`
 * `cudnn/cuda/lib64/libcudnn.so.4.0.7`
 * `cudnn/include/cudnn.h`
 * `cudnn/lib/x86_64-linux-gnu/libcudnn.so.4`
 * `cudnn/lib/x86_64-linux-gnu/libcudnn.so.4.0.7`
 
 * `data/attributes/Annotations/00001.txt - 04013.txt`
 * `data/attributes/ImageSets/make_traintxt.py`
 * `data/attributes/ImageSets/train.txt`
 * `data/attributes/ImageSets/train_people_all_valid.txt`
 * `data/attributes/Images/labels.txt`
 * `data/attributes/Images/people_labels_parser_fcnn.py`
 * `data/demo/000004_boxes.mat`
 * `data/demo/001551_boxes.mat`
 
 * `experiments/scripts/fast_rcnn.sh`
 
 * `lib/datasets/VOCdevkit-matlab-wrapper/get_voc_opts.m`
 * `lib/datasets/VOCdevkit-matlab-wrapper/voc_eval.m`
 * `lib/datasets/attributes.py`
 * `lib/datasets/factory.py`
 * `lib/fast_rcnn/config.py`
 * `lib/fast_rcnn/test.py`
 * `lib/fast_rcnn/train.py`
 * `lib/roi_data_layer/layer.py`
 * `lib/roi_data_layer/minibatch.py`
 * `lib/roi_data_layer/roidb.py`
 
 * `models/.gitignore`
 * `models/attributes/vgg/.gitignore`
 * `models/attributes/vgg/fast-rcnn.tgz`
 * `models/attributes/vgg/fast_rcnn/solver.prototxt`
 * `models/attributes/vgg/fast_rcnn/test.prototxt`
 * `models/attributes/vgg/fast_rcnn/train.prototxt`
 
 * `tools/demo.py`
 * `tools/test_net.py`
 * `tools/train_net.py`

# Appendix 2
