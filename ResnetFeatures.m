clear all;
close all;
clc;
%% Generate Depth Features
% This code uses MATLAB-17's resnet model, to generate the Depth features 
% of a depth image from the fc 1000 layer of the resnet.

load('File_List.mat');
rootFolder = fullfile('C:\Users\Dell user\Desktop\Custom Images');
imds = imageDatastore(fullfile(rootFolder), 'LabelSource', 'foldernames');
net = resnet50;
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
featureLayer = 'fc1000';
trainingFeatures = activations(net, imds, featureLayer, ...
'MiniBatchSize', 32);
