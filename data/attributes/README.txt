Attributes of People Dataset
Version 1.0

The dataset contains 4013 training and 4022 test examples of people.
Each example consists of an image centered at the person and nine boolean attributes, each of which is optionally specified. The images are taken from the H3D dataset as well as the PASCAL 2010 trainval set for the person category. If available, the highest resolution version of the PASCAL images is used from Flickr.
The split in train is random, with the following constraints:
- the training set does not contain any images from H3D-test
- the test set does not contain any images from H3D-train

The images are saved as jpeg files and the attributes and bounding boxes are specified in a file labels.txt

Each line in labels.txt corresponds to a separate annotation and has the following format:

<img_name> <xmin> <ymin> <width> <height> a1 a2 a3 a4 a5 a6 a7 a8 a9

where:
   <img_name> is the name of the corresponding image, such as 00022.jpg
   <xmin> <ymin> <width> and <height> specify the visible bounds of the person within the image.
   a1..a9 specify the value of the attribute. The value is 1 if the attribute is present, -1 if it is absent and 0 if it is unspecified.

The nine attributes are:
  a1: is_male
  a2: has_long_hair
  a3: has_glasses
  a4: has_hat
  a5: has_t-shirt
  a6: has_long_sleeves
  a7: has_shorts
  a8: has_jeans
  a9: has_long_pants

In addition, for each image we specify the visible bounds of each other person in the image, other than the target. The information is in the file other_bounds.txt in the following format:

<img_bame> <xmin> <ymin> <width> <height>

where
   <img_name> is the name of the corresponding image, such as 00022.jpg
   <xmin> <ymin> <width> <height>  is a bounding box of a background person in the image.

Notice that a given image might be skipped or listed multiple times depending on the number of background people in it. Notice also that the bounding box of the background person may span outside the image.

If you use this dataset please cite the following paper:

@InProceedings{BourdevAttributesICCV11,
  author       = "Lubomir Bourdev and Subhransu Maji and Jitendra Malik",
  title        = "Describing People: Poselet-Based Attribute Classification",
  booktitle    = "International Conference on Computer Vision (ICCV)",
  year         = "2011",
  url          = "http://www.eecs.berkeley.edu/~lbourdev/poselets"
}

If you have questions about this dataset please contact Lubomir at lubomir.bourdev@gmail.com
