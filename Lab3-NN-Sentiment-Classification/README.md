# Sentiment Classification of tweets Deep Neural Networks
## Project Description
* Text classification is one of the most important tasks in Natural Language Processing. Sentiment Analysis is a term that you must have 
heard if you have been in the Tech field long enough. It is the process of ‘computationally’ determining whether a piece of writing is :
  * positive
  * negative
  * neutral
 
 * Here we build \& test different deep learning models with different configurations to evluate their performances (**accuracy** and **test loss**) 
 using the [Pytorch Library](https://pytorch.org/) \\
 **Note** : [Keras Library](https://keras.io/) can also be used for testing sequential deep learning models.

## Datasets
* **Sentence Polarity Dataset 2 ** : This dataset contains 5331 positive and 5331 negative movie reviews, from Rotten Tomatoes and it is a binary classification problem (positive, negative).
* ** Semeval 2017 Task4-A 3** : This dataset contains tweets that are classified in 3 classes (positive, negative, neutral) with 49570 training examples and 12284 test examples.

## Preparation Lab
A really simple Neural Network (NN) was implemented to get used to Pytorch.

## Lab
In the Lab, we evaluated the following models
* various input representations + RNN : LSTM 
* various input representations + RNN : LSTM  + attention layer
* various input representations + RNN : Bi-LSTM 
