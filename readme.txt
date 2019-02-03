pytorch_image_synthesis_hypothesis.py 
----
classifies the cifar10 images
From the confusion matrix we found out that there were classes where the confusion was comparatively very high e.g. - ship - place, cat - dog, deer-bird, car-truck etc.
We tried to reduce the confusion by synthesizing images from these two classes using dcgan and then introducing it as a new class in the training data set.(Not in the
test data set).
We took 1000 synthesized images and then appended them to the individual data bin files with 200 images to each of them.

dcgan_cifar_ship_truck_synthesis.py
----
this code was used to generate the synthesized images from the two classes
the path to the folder need to be changed.
the images get generated in the current folder
One more image named progress.jpeg gets generated which is a collation of 64 images.

custom_cifar10.py
----
Wrote own cifar10 dataloader as we had to suppress the integrity check for the dataset.
We modified the md5 checksums of the individual data bin in these folders as well as that
of the batches.meta file.

bacthes.meta
----
The number of examples need to be changed and the new class name like ship_plane should be added
The CIFAR10 class from this file is used.

cifar10_write.py
----
Appending the newly synthesized images to the databins

The images needed to be resized before appending because the cifar10 images are 32x32.
Whereas the newly generated images are 32x32

IMGP tool was used to resize the images
