# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/project3_image1.png "Model Visualization"
[image2]: ./images/center_track_one.jpg "Center Track One"
[image3]: ./images/center_track_one_opposite.jpg "Center Track One opposite"
[image4]: ./images/center_track_two.jpg "Center Track Two"
[image5]: ./images/center_track_two_opposite.jpg "Center Track Two opposite"
[image6]: ./images/center_right1.jpg "Recovery One"
[image7]: ./images/center_right2.jpg "Recovery Two"
[image8]: ./images/center_right3.jpg "Recovery Three"
[image9]: ./images/center_left1.jpg "Recovery Four"
[image10]: ./images/center_left2.jpg "Recovery Five"
[image11]: ./images/center_left3.jpg "Recovery Six"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavioral_Cloning.ipynb containing the jupyter notebook used to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showcasing a successful lap around the first circuit in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing (My model requires keras-2.0.6, which can be downloaded from [here](https://pypi.python.org/pypi/Keras/2.0.6))
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The Behavioral_Cloning.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the nvidia end to end learning for self driving cars [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Also data from both the tracks was used for training to reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first gather a good diverese dataset followed by a good newtork to be trained.

My first step was to use a convolution neural network model similar to the nvidia end to end deep learning. I thought this model might be appropriate because this was used for a very similar purpose by nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding some dropout and then their wasn't any overfitting.

Then I also forgot to convert the cv read image from BGR to RGB, after fixing this the model worked seamlessly. 


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture


Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I repeated the same doing 2 laps in the ooposite direction.

![alt text][image3]

Then I recorded one lap on track 2 using center lane driving.

![alt text][image4]

Then I recorded one lap on track 2 in opposite direction using center lane driving.

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from right :

![alt text][image6]
![alt text][image7]
![alt text][image8]

These images show what a recovery looks like starting from left :

![alt text][image9]
![alt text][image10]
![alt text][image11]




After the collection process, I had 7096 number of data points. I then preprocessed this data by normalization and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4-5 as evidenced by the training output. I used an adam optimizer so that manually training the learning rate wasn't necessary.
