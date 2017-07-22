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

![Training without Dropout][training1]
*Training without Dropout*

![Training with Dropout][training2]
*Training with Dropout implemented*

#### Model parameter tuning
I tuned the batch size, to match the memory of the graphics card. With the final version of the model, a batch size of 64 was the optimum.

The model used an adam optimizer, so the learning rate was not tuned.

Another parameter in the model is to determine how much the stearing angle of left and right camera images is changed. I have tried 0.1, 
0.2, and 0.4. Although difficult to really see the effect systematically, I decided to fix it to 0.4. This seems to make the corrections a 
bit larger when moving off the center of the track.

#### Appropriate training data


### Architecture and Training Strategy

#### Solution Design 

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Simulation
### Test of the trained model
