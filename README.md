# DeepGamblers-tensorflow
TensorFlow reimplementation of https://github.com/Z-T-WANG/NIPS2019DeepGamblers.  

## MNIST experiment
I provide the MNIST experiment with a simple CNN model which is not provided in the original repository.  
The experiment is provided on [colab notebook](https://colab.research.google.com/drive/1LNZpbevhUQP4gf8K1ciP6I-CvzGxZo6k).

Top 10 high abstetion score images in the test set:

![top_ten_high_abstention_score_images](https://i.imgur.com/Hos7Rss.png)

Rotated 9's experiment (title stands for the abstention score, class 10 is the abstention class):

![rotated_nines_experiment](https://i.imgur.com/2y30f7c.png)

## CIFAR10 experiment
`main.py` includes model (VGG16_bn only) training and evaluation.  
Trained VGG16_bn models are provided in [my Google Drive](https://drive.google.com/open?id=1UxpO133UqP7-h6euGE1ggHrbmsxdr0gt); ckpt-12 is a full-trained model (100 epochs for pretraining with a simple cross entropy loss and 200 epochs for training with the gambler loss).

The result is the following.  
Even though the performance of my trained model looks quite bad (the original paper shows error 6.12% for coverage 1.00, which means my base VGG16_bn model is maybe not constructed well), the proposed abstention mechanism works well.

```
Coverage: 1.00, Error: 12.93%
Coverage: 0.95, Error: 10.59%
Coverage: 0.90, Error: 8.58%
Coverage: 0.85, Error: 7.06%
Coverage: 0.80, Error: 5.93%
Coverage: 0.75, Error: 4.88%
Coverage: 0.70, Error: 4.06%
Coverage: 0.65, Error: 3.32%
Coverage: 0.60, Error: 2.83%
Coverage: 0.55, Error: 2.25%
Coverage: 0.50, Error: 1.80%
```
