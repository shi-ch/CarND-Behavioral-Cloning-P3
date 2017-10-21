**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[imgcenter]: ./images/center-lap2.png "Center Camera"
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

The overall strategy for deriving a model architecture was to reuse the nVidia's self driving model, then focus on preparing preprocessing training datasets.

My first step was to use a convolution neural network model similar to the Nividia's self driving model. I thought this model might be appropriate because it is quite small and simple, thus can be trained with limited resource of a laptop.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The models had low mean squared error on the training set and also quite low mean squared error on the validation set. This implied the the training data might not be diversified enough to teach the model to capture the characteristics of images that leads to the steering decisions.

To add more data points for training, I added left-right flipped images; then added images from left camera and right camera -- this step was proved to be very helpful to stablize the cars' movement and the car seemed to capture the idea of staying on track with these sets. I also re-ran recovery runs multiple times to get better record starting points and better recovering courses.

However the model still could not finish one lap and always ran out the turns on the right side, which means the models are not alertive and change the steering angle drastically enough to a larger angle. This implied the model might feel itself was trained to do only tiny angle adjustments. So I randomly dropped 70% images where the steering angle less than 0.1 [(mode.py, 37)](https://github.com/shi-ch/CarND-Behavioral-Cloning-P3/blob/master/model.py#L37) to reduce the percentage of running straight in the hope the model could learn better about making steering angle changes and the I did see the improvements.

Another issue was overfitting, while the loss in training dropped steadily after epochs, the validation loss stayed in a certain range. This implied overfitting of the model. To combat the overfitting, I modified the model by injecting drop out layers following almost every layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-111) consists of a convolution neural network with the following layers and layer sizes.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 38, 158, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 77, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 17, 77, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 37, 48)     43248       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 7, 37, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 35, 64)     27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 5, 35, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 33, 64)     36928       dropout_4[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 3, 33, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6336)          0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 6336)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           633700      dropout_6[0][0]                  
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_7[0][0]                  
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_8[0][0]                  
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_9[0][0]                  
====================================================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
____________________________________________________________________________________________________
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![nVidia Model][imagecnn]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Camera][imgcenter]

I then recorded one lap driving on opposite direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when the car is too close to one road side. These images show what a recovery looks like starting from positions and directions that need immediate steering changes :

The recorded video of recovering from right:
[![](./images/recover-right.mp4)](https://github.com/chunhui-shi/CarND-Behavioral-Cloning-P3/blob/master/images/recover-right.mp4)

The recorded video of recovering for turn 2:
[![](./images/recover-turn2.mp4)](https://github.com/chunhui-shi/CarND-Behavioral-Cloning-P3/blob/master/images/recover-turn2.mp4)


The data sets include 2 laps on anti-clock-wise direction, one lap on close-wise direction, two sets for recovering from left and right, and three sets for the three turns in the lap.

I also use images from left camera and right camera with adjusted steering as training datasets. This prove to be very helpful to teach the car to stay in the track.

To augment the data set, I also flipped images and angles thinking that this would provide more diversity to training data.

After the collection process, I had more than 20K number of data points. I then preprocessed this data by randomly adjusting the brightness so that the data of continuous images could be very different from each other. Adding more randomness could help the model to focus on really important characteristics of the images that are related to staying on track. 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the validation loss reach a value and no more improvement could acheived. I used an adam optimizer so that manually training the learning rate wasn't necessary.
