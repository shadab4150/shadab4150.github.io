# Disaster Post Classification on Social media.

* In this repo a natural language model has been trained and developed to classify post as fake or real regarding natural disasters
earthquake, cyclone, etc.

Social media has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. 

* Take this example:

![kd](https://i.ibb.co/5FJPby7/tweet-screenshot.png)
* **Tweet source:** https://twitter.com/AnyOtherAnnaK/status/629195955506708480

The author explicitly uses the word “ABLAZE” but means it metaphorically. 
This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

* In this project I  build a deep learning model that classifies which posts are about real disasters and which one’s aren’t.

* **Disclaimer: The dataset for this model contains text that may be considered profane, vulgar, or offensive.**
### Acknowledgments

* This dataset was created by the company **figure-eight** and originally shared on their [**‘Data For Everyone’**](https://www.figure-eight.com/data-for-everyone/) website.

#### Project Notebook
* Project notebook [**here**](https://github.com/shadab4150/Disaster-post-classification-NLP/blob/master/real_or_not_disaster_tweets.ipynb)

* First step was building a language model which can learn and understand tweets.

* I was able to get an accuracy of 49% with help of learning rate finder.

![kd](https://i.ibb.co/9Z3Gbqc/sha12.jpg)
![kd](https://i.ibb.co/kGrwBRH/sha1212.jpg)

* Then I build a classification model to classify the post as real or fake.
* And was able to achieve an accuracy of **80.3681 %.**

![kd](https://i.ibb.co/FqSKFH4/cl1.jpg)
![kd](https://i.ibb.co/0nRNWLZ/acc.jpg)

* Confusion Matrix:

![kd](https://i.ibb.co/C1gkFyp/confusion.jpg)

* Top losses:

![kd](https://i.ibb.co/8bppzwm/top-losses.jpg)

* The final models has been saved [**here.**](https://drive.google.com/drive/folders/1q0BLeyVlNhkGDtFpaQ1oKwc1A6HDRGCY?usp=sharing)
## For fun lets see how accurate the language learner model can complete sentences and post.

![kd](https://i.ibb.co/Qj8RyPD/tweet-comp.jpg)

* **A great thanks to jeremy howard for fast.ai deep learning libraries.**
