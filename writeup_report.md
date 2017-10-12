**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/placeholder.png "Model Visualization"
[image2]: ./images/placeholder.png "Grayscaling"
[image3]: ./images/placeholder_small.png "Recovery Image"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image"
[imagecnn]: ./images/nvidia-cnn.png "Nvidia CNN Architecture"



#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model to use is [Nvidia Architecture for self driving car](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This model consists of a Cropping layer to ignore top part of the picture and a lambda layer to normalize the images in front of the convolution neural network which includes convolution feature maps with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 92-100), each layer includes rectified linear unit(relu) and a Dropout layer is added following each convolution layer.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers following almost all layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 152).

#### 4. Appropriate training data

The training data were recorded from multiple runs under training mode. Include two laps and one reverse lap, and some special training runs focused on the three turns and recovering from road sides.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reuse the a

My first step was to use a convolution neural network model similar to the Nividia's self driving model. I thought this model might be appropriate because it is quite small and simple, while it 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-111) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![nVidia Model][imagecnn]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded one lap driving on opposite direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when the car is too close to one road side. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

The data sets include 2 laps on anti-clock-wise direction, one lap on close-wise direction, two sets for recovering from left and right, and three sets for the three turns in the lap.

The brightness of images were randomly adjusted so the data of continuous images could be very different thus the model could focus on really important characteristics of the images. 

I also use images from left camera and right camera with adjusted steering as training datasets. This prove to be very helpful to teach the car to stay in the track.

To augment the data set, I also flipped images and angles thinking that this would provide more diversity to training data.


![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
