---
layout: post
title: Comparing activation function ReLU vs Mish
subtitle: Comparing ReLU vs Mish activation function on classification accuracy of MNIST dataset. 
bigimg: "/img/msh1.jpg"
tags: [CNN,Mish,ReLU]
---

# Activation functions:

* To put in simple terms, an artificial neuron calculates the **weighted sum**  of its inputs and adds a bias, as shown in the figure below by the net input.


<center><img src="https://www.geeksforgeeks.org/wp-content/uploads/33-1-1-660x306.png"></center>

* Now the value of **net input** can be any anything from **-inf** to **+inf**. The neuron doesn’t really know how to bound to value and thus is not able to decide the firing pattern. Thus the activation function is an important part of an artificial neural network. They **basically decide whether a neuron should be activated or not.**

* The activation function is a non-linear transformation that we do over the input before sending it to the next layer of neurons or finalizing it as output.



# Comparing activation function ReLU vs Mish

##  **ReLU** ( Rectified Linear Unit )

* ReLU is a type of activation function. Mathematically, it is defined as ***y = max(0, x).***

<center>Visually, it looks like the following:</center>


<center><img src="https://miro.medium.com/max/1026/1*DfMRHwxY1gyyDmrIAd-gjQ.png"></center>

ReLU is the most commonly used activation function in neural networks, especially in **CNNs**. If you are unsure what activation function to use in your network, ReLU is usually a good first choice.
ReLU is linear (identity) for all positive values, and zero for all negative values. This means that:
* It’s cheap to compute as there is no complicated math. The model can therefore take less time to train or run.
* It converges faster. Linearity means that the slope doesn’t plateau, or “saturate,” when x gets large. It doesn’t have the vanishing gradient problem suffered by other activation functions like sigmoid or tanh.
* Since ReLU is zero for all negative inputs, it’s likely for any given unit to not activate at all.


##  Mish is Self Regularized Non-Monotonic Activation Function

A new paper by [**Diganta Misra**](https://github.com/digantamisra98/Mish) titled **Mish: A Self Regularized Non-Monotonic Neural Activation Function** introduces the AI world to a new deep learning activation function that shows improvements over both **Swish (+.494%) and ReLU** (+ 1.671%) on final accuracy.
* It is modified verion of swish activation function. Mathematically, it is defined as:

<center><img src="https://i.ibb.co/TK0LPcD/mishmath.jpg"></center>


<center>Visually, it looks like the following:</center>


<center><img src="https://miro.medium.com/max/512/1*S9xYzBLjOd4JrrGC-U2Zhg.jpeg"></center>


* Here is graph of six different activation functions:


<center><img src="/img/activation fucntion.png"></center>


I downloaded the pytorch implementation of **Mish** activation function of Diganta Mishra's from a kaggle user [**Iafoos**](https://www.kaggle.com/iafoss/mish-activation/) to compare it with **ReLU** on classification task of classic MNIST dataset.


* Found true that it performs better.
* But one of its down side is it's computationally expensive compared to **ReLU** which just takes **max(0,x)**.


## Below are my findings and the Project [**Notebook:**](https://github.com/shadab4150/Deep-Learning-Classifiers/blob/master/comparing_activation_function_ReLU_vs_Mish.ipynb)

* MNIST dataset has an unevenly distributed set of images.
<img src="/img/count.jpg">
* Metrics I choose was macro average f1_score and accuracy.
* And defined two model one using ReLU activation and the other one using Mish Activation. Both models summary is given below:

<center><h4>Model with Mish.</h4></center>


![kd](https://i.ibb.co/L6NXBjC/modelmish.jpg)


<center><h4>Summary of Model with Mish.</h4></center>


![kd](https://i.ibb.co/L8Zj94f/modelMsh.jpg)


<center><h4>Model with ReLU.</h4></center>


![kd](https://i.ibb.co/6Y5XXJn/model-relu.jpg)


<center><h4>Summary of Model with ReLU.</h4></center>


![kd](https://i.ibb.co/xL4q3XQ/modelRel.jpg)


I ran both the models with a learning rate of **1e-1** for 6 epochs.
Below are the results:


<center><h4>Six epochs for Model with ReLU Activation function.</h4></center>


<center><img src="https://i.ibb.co/sVzb77s/mishlearn.jpg"></center>


<center><h4>Six epochs for Model with Mish Activation function.</h4></center>


<center><img src="https://i.ibb.co/McKPb4Q/learn-mish1313.jpg"></center>


#### As one can see Model with Mish activation function gives better F1_score as well accuracy.


#### Credits and Refrences:
```
https://github.com/digantamisra98/Mish
https://www.kaggle.com/iafoss/mish-activation/
https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7
https://www.geeksforgeeks.org/activation-functions/
```
#### Shoutout to:
* Fast.ai team for there fast.ai easy to use software to test model swiftly


