# Kaggle Facial Keypoints Detection
The goal of this repository is to store the different models I tried so far for the Kaggle Facial Keypoints Detection competition.

## Models
I took inspiration from the Deep Learning tutorial associated to this competition. So far, I have tried two models (implemented with TensorFlow):

* One Hidden Layer Neural Network
* Convolutional Neural Network

I also did *Data Augmentation* (flipping some images in each batch to create new examples).

## Results

| Model                                     | Epochs | Real time   | Best score achieved |
| ----------------------------------------- |:------:|:-----------:|:-------------------:|
| One Hidden Layer                          | 200    | 1 min 27 s  | 3.78045             |
| ConvNet (with Data Augmentation)          | 200    | 70 min 10 s | 3.07737             |

## References
Competition: https://www.kaggle.com/c/facial-keypoints-detection

Tutorial: http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/