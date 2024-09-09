![Build Status](https://img.shields.io/github/workflow/status/OWNER/REPO/CI)
![License](https://img.shields.io/github/license/OWNER/REPO)
![Version](https://img.shields.io/github/v/release/OWNER/REPO)

# deeplearning-practice
A set of problems for practice for the tensorflow developer certificate exam

## Overview
- [reuters](#reuters)
- [housing-prices](#housing-prices)
- [License](#license)

# Reuters
Type: Multiclassification problem
Dataset: Reuters datasets with 46 topics at least 10 samples for each one.
Baseline: random selection topics will give an accuracy around 19% on validation, thus this is the metric to beat

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


## License

Distributed under the MIT License. See `LICENSE` for more information.

