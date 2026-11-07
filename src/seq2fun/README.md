# file notations

- baseline_models.py: This file contains the implementation of baseline models used for sequence-to-function prediction tasks including: DeepSEA, DanQ, and DeepFormer.

- train_baseline.py: This file contains the training loop and related functions for training the baseline models on sequence-to-function prediction tasks optimized for AUC metric with early stopping.

models.py: This file includes the definitions of a seq2fun model architecture can adapt to various input sequence lengths, the desgin is carefully desgined to able testing performance on different input sequence lengths. The CNN components are inspired by DeepSEA but 1st layers are used with exp activation function (motivated by a study by Peter Koo). The transformer components are simply a BERT-like transformer encoder (standard cos-sin positional encoder) with hf transformers library implementation.

- train2.py: This file contains the training loop and related functions for training the seq2fun models on sequence-to-function prediction tasks optimized for AUC metric with early stopping.


- data_utils.py: This file provides utility functions for loading and preprocessing sequence data for training and evaluation of the models.