clear all;
close all;
clc;
%%
% This code takes in the features as input and arranges the features into 
% a matrix corresponding to the order of the questions in dataset.

Order=csvread('test_filenames.csv');
load('Complete_Features_Depth_ResNet.mat');
load('Features_IMDS.mat');
for index=1:size(Order,1)
    filename{index}=strcat('C:\Users\Dell user\Documents\MATLAB\Depth_Images\','image',num2str(Order(index)),'.jpg');
    for inner_index=1:size(imds.Files,1)
        Flag=strcmp(imds.Files(inner_index),filename(index));
        if(Flag)
            Final_Test_Feature_Set(index,:)=trainingFeatures(:,:,:,inner_index);
        end
    end
end
