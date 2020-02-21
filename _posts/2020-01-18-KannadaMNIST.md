---
layout: post
title: KannadNet Detecting Numbers Written In Kannada Numerical System
subtitle: Detecting Numbers Written In Kannada Numerical System 
bigimg: https://i.ibb.co/yWcPzNd/number123.png
tags: [CNN, Kannda]
---


# KannadNet: Detecting Numbers Written In Kannada Numerical System


* Detecting Numbers Written In Kannada Numerical System using a  ResNet-34, a **Deep Residual Learning Neural Network for Image Recognition.**

* The **DataSet** was in a pixel information **csv** file containing an image of **28x28 pixel**.

* First, I tried to train it using **RandomForest Classifier** which gave an accuracy score of **97.98 %.**

* Then I switched to **ResNet34.**

* I have achieved an accuracy of **99.4 %.**

* To see the project jupyter Notebook click [**Here**](https://shadab4150.github.io/KannadaNet/Kannada_digit_recognizer_randomForest_resnet34.html)

## Kannada Numbers Covnverted to image from pixel data using matrix. 28x28
![kd](https://i.ibb.co/yWcPzNd/number123.png)

## Top Losses after the KannadNet was trained on a 34 layer conv Neural network.
![kd](https://i.ibb.co/8PMQ2RH/loss.png)

## Confusion Matrix
![kd](https://i.ibb.co/X4YtHmz/confu.png)

## Actual and Predicted images:
![kd](https://i.ibb.co/BBtqy5Q/pred2.png)
![kd](https://i.ibb.co/YcWRr89/pred1.png)
