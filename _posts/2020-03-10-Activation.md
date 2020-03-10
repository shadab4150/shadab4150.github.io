---
layout: post
title: Comparing activation function ReLU vs Mish
subtitle: Comparing ReLU vs Mish activation function on classification accuracy of MNIST dataset. 
bigimg: "/img/msh1.jpg"
tags: [CNN,Mish,ReLU]
---

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


*  Mish is Self Regularized Non-Monotonic Activation Function


