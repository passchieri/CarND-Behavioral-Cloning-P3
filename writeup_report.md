# Behavioral Cloning 
Author: **Igor Passchier**

---

## Introduction 

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The following sections will address the [rubric points](https://review.udacity.com/#!/rubrics/432/view) 

[//]: # (Image References)

[model]: ./pictures/network.png "Model Visualization"
[training1]: ./pictures/training_nodropout.png "Training performance without Dropout"
[training2]: ./pictures/training_withdropout.png "Training performance with Dropout"
[center]: ./pictures/center.png "Center camera image"
[left]: ./pictures/left.png "Left camera image"
[right]: ./pictures/right.png "Right camera image"
[cropped]: ./pictures/cropped.png "Cropped camera image"
[flipped]: ./pictures/flipped.png "Flipped camera image"

## Required Files

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following required files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode. This file is unmodified from the supplied version.
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results (this document)

In addition, I have supplied the following additional files:
* epoch1.h5 Trained version of the model after 1 epoch, without Dropout
* epoch3.h5 Trained version of the model after 3 epochs, without Dropout
* output_speed15.mp4 Video of the result, driving at 15 mph instead of 9 mph

## Quality of Code
### Code is functional
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
./drive.py model.h5
```
The training data is collected on a mac, using the mac version of the simulator, while the training has been done on an AWS g2x2large Ubuntu virtual machine.

### Usable and Readable code

The model.py file contains the code for training and saving the convolution neural network. It contains the code to augment the images, and the model definition, and the code to perform the training of the model. The support functions are all documented with docstrings, the
main process to create,train, and save the results are documented with normal comments.

### Model Architecture and Training Strategy

#### An appropriate model architecture has been employed
After several modifications and trials (see later), I finally decided to use the [Nvidia network](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars). It consists of 5 convolutional layers with Relu activation. The first 3 also have a MaxPooling layer. This is followed by a flattening and 3 Fully connected layers. The final output is the stearing angle for the vehicle. The model is represented in the figure below:

![Model][model]

#### Reducing overfitting in the model
A dropout layer is added after every convolution layer. The images below show the training performance with and without dropout. Due to the early termination that has been implemented, the number of epochs is different for the 2 training sessions

| ![Training without Dropout][training1] | ![Training with Dropout][training2] |
| *Training without Dropout* | *Training with Dropout implemented* |

#### Model parameter tuning
I tuned the batch size, to match the memory of the graphics card. With the final version of the model, a batch size of 64 was the optimum.

The model used an adam optimizer, so the learning rate was not tuned.

Another parameter in the model is to determine how much the stearing angle of left and right camera images is changed. I have tried 0.1, 
0.2, and 0.4. Although difficult to really see the effect systematically, I decided to fix it to 0.4. This seems to make the corrections a 
bit larger when moving off the center of the track.

#### Appropriate training data
I created several training sets. The first set is from several laps trying to stay in the center. This is more difficult than it seems, 
which results in some bad training data. I have made provisions to remove those, by just deleting the center images and skipping them 
while reading in the images, but in the end did not have to use this. A second training set contains a lap in the opposite direction, and 
short intervals of steering back to the center from either the left or the right.

The code is capable of reading training data from multiple directories, so I can easily combine them to see the effect. In this way, I 
could also include the sample training data, combined with my own training data.

I have augmented the training data by using also the left and right camera images, with a modified stearing angle (see previous section). 
Futhermore, I have flipped all images. 

![Left image][left]

*Image from the left camera*

![Center image][center]

*Image from the cneter camera*

![Right image][right]

*Image from the Right camera*

The full paths, (modified) steering angle, and need to flip are all stored in seperate arrays. These are then split in training and 
validation sets. Futher, generators are used to actually read the images, flip if required, and provide the X and y values used in 
training and validation.

The images are also cropped (part of top and bottom is removed), but this is not done in the image pre-processing, but in the model. In this way, no preprocessing needs to be added to the drive.py code.

![Cropped image][cropped]

*Cropped image from the center camera*

![Flipped image][flipped]

*Flipped image from the center camera*


#### Solution Design 

I have build up the model in several steps, to understand what was actually hapening. The first serious model I used was based on Lenet. 
This did not give a stable driving experience. I modified the size of the first convolution kernel, as the images we are using and the 
features in the images are much larger than 32x32. However, this did not really solved the instabilities in the driving. Therefore, I 
switched to a network inspired on the NVidia network, suggested in the lesson and also used in our own autonomous test vehicle with 
DrivePX platform. 

With the Dropout implemented as well and sufficient training data, this resulted in a smooth training and also good driving result for 
track 1. I also tried on track 2, but if the model is only trained on track 1, this fails already at the first corner. Without track 2 
training data, this will not work. Furthermore, maybe the cropping of the image need to be modified, because when going up hill, the 
cropping will likely remove relevant features. I have not worked this out in detail.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## Simulation
### Test of the trained model
The provided video shows that the final model keeps the vehicle nicely on the road at 9 Mph. 

I have tried to incease the driving speed to 15 Mph, see additional video. What can be observed, is that the vehicle sometimes start to 
*oscilate*, i.e. going from left to right and back again on the track. This means that the stearing is to aggresive. To prevent this, the 
training and control should either include both stearing and set_speed, or a more advanced model has to be added after the output of the 
neural network. I have tried to multiply the stearing output with a constant factor, but that is clearly too simple: the factor need to be 
applied to the stearing rate, not to the stearing angle itself. I have not investigated this further.
