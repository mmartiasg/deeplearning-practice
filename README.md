# Tensorflow practice
A set of problems for practice for the tensorflow developer certificate exam

# vae_experiments

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-1.21-blue?logo=numpy&logoColor=white)
![License](https://img.shields.io/github/license/OWNER/REPO)
![Version](https://img.shields.io/github/v/release/OWNER/REPO)

## Overview
- [reuters](#reuters)
- [housing-prices](#housing-prices)
- [License](#license)

# Reuters
Type: Multiclassification problem
Dataset: Reuters datasets with 46 topics at least 10 samples for each one.
Baseline: random selection topics will give an accuracy around 19% on validation, thus this is the metric to beat

## Future experiments:
- How does the number of words affect the algorithm's capacity to learn
- Or maybe increasing or decreasing the max sequence has a negative impact at some point
- Network sizeL: is it better to have a bigger one. How will that overfit the dataset?
- So far I'm using hot encoding I wonder if using a TF-IDF will help because I only have if that token appears or not but all of them weigh the same!
Maybe embeddings will make an improvement?
- Another models like LSTM, GRU or Transformers could be better for this task and if that is how much better?
- It is interesting to understand the power of the representation it is possible to achieve by increasing the size of the network o by some restrictions or increasing the limits of the inductive bias the model introduces by limiting the amount of data it sees at once, in cases such as fully connected layers or sequence models.

## Parameters:
- NUM_WORDS: 10000
- 1 layer with 64 units to avoid information bottleneck
- Give us an accuracy around 79.8% on validation but it will start to overfit after 10 or so epochs.
- By reducing the NUM_WORDS reduce ofcourse the overfit but if reduce too much (lets say to 100) the accuracy drops to 68%.
- The same happens fi the units are reduced.
## Conclussion:
It seems better to stop training after 8 epochs. The model could be smaller, or num of words reduced not too much; it could drop to 3000 without harming the metric too much.


# Housing-prices
Type: Scalar regression problem

Observation: The main focus is how to handle a scenario where you have little data to work with, like in this case where the validation set will be under-represented and thus this will lead to variations in the metric just by the way the samples that will fall in the validation set might not be the best representatives of the training distribution we could end up with just the highest house values only and for training the lowest values and that evaluation will not be significant to make a decision about the qualities of that model.

The main point here is how to apply a K-fold method of evaluation as in this case to solve that constraint.

Metrics is MAE Loss is MSE

After that, the baseline performance on MAE is 6.533 on the test set. The model with k=4 folds achieve 2.5 on the test set.

## Implementation will be carried on
- Pythorch
- Keras
- JAX

## License

Distributed under the MIT License. See `LICENSE` for more information.

