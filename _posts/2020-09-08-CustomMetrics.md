---
layout: post
title: Kaggle Fastai Custom Metrics
subtitle: Custom Metrics for fastai v1 for kaggle competitions.
bigimg: "https://d1m75rqqgidzqn.cloudfront.net/wp-data/2019/11/07200611/shutterstock_1061069282-696x428.jpg"
tags: [kaggle, fastai , metrics]
---

## kaggle fastai custom metrics
***

Custom Metrics for fastai v1 for kaggle competitions.

***
## Disclaimer :

Each Kaggle competition has a unique metrics suited to its, need this package lets you download those custom metrics to be used with fastai library.
Since Metrics are an important part to evaulate your models performance.

## Installation 

```sh
pip install kaggle-fastai-custom-metrics
```

or

```bash
git clone https://github.com/shadab4150/kaggle_fastai_custom_metrics
cd kaggle_fastai_custom_metrics
pip install .
```
## Usage :

```python
from kaggle_fastai_custom_metrics.kfcm import *
metric = weighted_auc();
learn = Learner(data, arch, metrics= [metric] );
```

```
print_all_metrics()
```

<table>
  <tr>
    <th>Competition URL</th>
    <th>Metric</th> 
  </tr>
  <tr>
    <td>https://www.kaggle.com/c/plant-pathology-2020-fgvc7</td>
    <td>column_mean_logloss</td>
  </tr>
  <tr>
    <td>https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge</td>
    <td>column_mean_aucroc</td>
  </tr>
  <tr>
    <td>https://www.kaggle.com/c/trends-assessment-prediction</td>
    <td>weighted_mae</td>
  </tr>
   <tr>
    <td> https://www.kaggle.com/c/alaska2-image-steganalysis</td>
    <td>alaska_weighted_auc</td>
  </tr>
   <tr>
    <td>https://www.kaggle.com/c/google-quest-challenge</td>
    <td>AvgSpearman</td>
  </tr>
  <tr>
    <td>https://www.kaggle.com/c/landmark-recognition-2020</td>
    <td>GAP_vector</td>
  </tr>
  <tr>
  





