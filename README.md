# storytestimony

The following repository contains the data and the code for the paper "Reports of personal experiences and stories in argumentation:
datasets and analysis". [1]

## Datasets

The data used to train and test the models is stored in `datasets`. All information about the datasets, annotation and structure can be found the `datasets/readme.md`

## Training and testing a model

The simple classifiers (bag of words, random forest) can be found in `baselines`.
To train and test a new model with BERT you can use the script `bert_classifier.py`

```
python bert_classifier source_folder result_folder epochs gpu model_path --test
```
- source_folder: loads *train.csv* and *val.csv* from the given path
- result_folder: saves the trained model (*metrics.pt* and *model.pt*), results (*classification_report.csv*) and the predictions (*predictions.csv* with gold label, predicted label and class probabilities)
- epochs: the number of epochs to train
- gpu: the GPU (no) to train on if trained on a GPU
- model_path: the LM name to load the pretrained LM to fine-tune
  - e.g. *bert-base-uncased*
- --test: add this flag to create predictions for the test data