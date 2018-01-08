# AnsweringQuestionsOnGivenImages

README:

Here, I try to answer questions based on real world images. 
As an input the model is given images and questions associated 
with that image and the model is trained to predict the most 
logical answer after passing the images with a CNN and questions separtely in a LSTM.




Answering questions based on real world image is
a challenging task that has emerged as an active research topic in
the recent years. This problem marks the intersection of Natural
language processing and Computer Vision. The challenge lies in
not only incorporating the techniques that are best for the image
processing and natural language processing but integrating them
efficiently for the task of Visual Question ansIring and also
how to best utilize the features from the image and questions.
Since most of the current approaches are trained on 2D images
they are not robust in dealing with questions that require 3D
representation of the scene. My approach D-VQA(Depth Vqa) and DS-VQA(Depth Switched)
provides a novel way of utilizing the spatial features that increases
the performance compared to the earlier methodologies. My
model D-VQA achieves a WUPS(0.0) score of 81.57% on the
reduced DAQUAR dataset, which is a significant improvement.

I have incorporated a seperate segment of Depth features which are extracted using ResNet and have combined them with global representation features of the image to predict
 the answers of the input questions even more accurately incorporating depth based insight.


I have proposed 2 new models namely D-VQA and DS-VQA for thee task of visual question ansIring.
I have experimented my models accross multiple settings and have reported the results in the report.

My main contribution in this work is to effectively improve the existing VQA methods by incorporating and learning the 
depth related parameters which is of high signifance in the VQA setting. The existing methods, perform Ill on ansIring the non-depth related questions but dont achieve the same performance with repect to the Depth related questions. 

Dataset Links: 
DAQUAR: https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/
NYU Depth Dataset V2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

IDE: Python 2.7, MATLAB 2017 
FrameWorks Used: Theano 0.9.0
				 Keras 0.3.3
				 Kraino library(developed for VQA specifically)

I have used MATLAB 2017 only for generating the Depth CNN features using MATLAB-17s Neural Network Toolbox for ResNet50
I have presented the basic framework of my code, whehrein I had changed parameters(mentioned in comments) to experiment with all the parameters mentioned in the report.

Steps to Run the D-VQA Model:
1) Make sure the "DAQUAR Triple" dataset is included in the folder.
2) Include the pre-trained CNN features for Global Image representation for evaluating competing algorithm Marinkowski et al. [9]
3) With the Depth Images of the NYU Depth Dataset V2, run the "ResNetFeatures.m" file after placing the filelists appropriately. (Done separately for test and train subsets).
4) In order to map the questions and image features, run the "Order_Duplicates.m" file. Now the extraced features and questions correspond to the same image.
5) To run the D-VQA model, run "D_VQA main.py" with the dependencies "Estimate_Frequencies.py", "Model_Definition.py" to generate the WUPS scores.

Steps to Run the DS-VQA Model:
1) Make sure the "DAQUAR Triple" dataset is included in the folder.
2) Include the pre-trained CNN features for Global Image representation for evaluating competing algorithm Minkowski et al. [9]
3) With the Depth Images of the NYU Depth Dataset V2, run the "ResNetFeatures.m" file after placing the filelists appropriately. (Done separately for test and train subsets).
4) In order to map the questions and image features, run the "Order_Duplicates.m" file. Now the extraced features and questions correspond to the same image.
5) To run the D-VQA model, run "DS_VQA main.py" with the dependencies "Estimate_Frequencies.py", "Model_Definition.py" which employs the switching automatically and reports the final Accuracy and WUPS scores.

