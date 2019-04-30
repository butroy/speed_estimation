# Speed Challenge  

### Introduction  

In this project, I estimated vehicle speed from a car dash cam. After research, I found that to tackle this problem, using optical flow analysis and convolutional neural network is a optimal choice. In this report. I would show how I realize a excellent result step by step.

### Data exploration
The data set includes two videos, one for train and one for test. Both of videos are shot at 20fps and speed of car at each frame in train video is also provided. 

First, I decode the video into pictures. Below is a sample frame of the video.

<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>

The track of speed in train video is

<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>


### Optical Flow
The key part of solving this motion problem is optical flow. Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movemement of object or camera. It is 2D vector field where each vector is a displacement vector showing the movement of points from first frame to second. 

Optical flow works on several assumptions:

1. The pixel intensities of an object do not change between consecutive frames.
2. Neighbouring pixels have similar motion.

Our problem fits for optical flow assumptions such that while a car moving forward, the road lines around the car and the passing cars provide good information for estimating car speed.   

<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>

<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>

<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>

Above set of images is a sample of two consecutive frames and their optical flow output. I crop out the sky and the engine cover parts of input frames since those parts mostly don't change and don't have too much information and will also waste computing resource in neural network training stage. As we can see in the optical flow image, the road lines' and the nearby cars' movements are recorded and we could use these information to train a CNN to estimate car speed. 

To be specific, I used two consecutive images to generate a optical flow image, and the average of two frames as the label 

## CNN structure  

I use the Keras API to create the neural network structure and I used an end to end learning stucture from [Nvidia](https://arxiv.org/pdf/1604.07316v1.pdf)

<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>

Before training, I shuffled the data randomly and choose 0.7/0.2/0.1 as train/valid/test ratio. I also use batch generator to generate images as need to save memory. 


## Training result


<p align="center">
 <img src= "https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" />
</p>

The mse loss for train/valid/test are 0.5432/0.9480/0.9 

And I took a evaluation on the whole training video and mse is 0.577

# Further improvement

Due to limited time and computing resources, there is a lot of possible improvement I could further try.

1. The train video has unbalance information on highway and local road. If we could collect more video under various traffic conditions, it should increase the robustness and the generability of the model. 

2. I only tried two NN structures and the nvidia model out performs the other one, however, there might be other NN structures better fit this problem

