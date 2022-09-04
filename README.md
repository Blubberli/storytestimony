# storytestimony

The following repository contains the data and the code for the paper "Reports of personal experiences and stories in argumentation:
datasets and analysis". [1]

## Datasets

The data used to train and test the models is stored in `datasets`. All information about the datasets, annotation and structure can be found the `datasets/readme.md`

## Training and testing a model

The simple classifiers (bag of words, random forest) can be found in `baselines`.
To train and test a new model with BERT you can use the script `bert_classifier.py`

```
python bert_classifier source_folder result_folder temp_folder epochs gpu model_path --test
```
- source_folder: loads *train.csv* and *val.csv* from the given path
- result_folder: saves the trained model (*metrics.pt* and *model.pt*), results (*classification_report.csv*) and the predictions (*predictions.csv* with gold label, predicted label and class probabilities)
- temp_folder: saves a minimal file (train, val, test) as a tsv with only text and label
- max_seqlen: the maximum sequence length (default is 512)
- epochs: the number of epochs to train
- gpu: the GPU (no) to train on if trained on a GPU
- model_path: the LM name to load the pretrained LM to fine-tune
  - bert-based transfomer model, e.g. *bert-base-uncased*
- --test: add this flag to create predictions for the test data

The domain-adapted versions of the language model are also available. For these, the underlying language model have been tuned on domain-specific data. 
Note that if you load these, you still need to fine-tune them on the classification task. 
To load them use the following names as *model_path*:
- falkne/bert-europarl-en [link to hub](https://huggingface.co/falkne/bert-europarl-en)
- falkne/bert-online-discussions-en [link to hub](https://huggingface.co/falkne/bert-online-discussions-en)
- falkne/bert-discussions-online-parliament-en [link to hub](https://huggingface.co/falkne/bert-mixed-discussions-europarl-online-en)
```
python bert_classifier.py datasets/splits/cmv_10splits/split0/ datasets/splits/cmv_10splits/split0/ tmp/ 125 tmp/ 5 0 "falkne/bert-europarl-en" --test
```
## Regression Analysis

The data and the code for the linear regression analysis can be found in `regression_analysis`.





[1] [Reports of personal experiences and stories in argumentation: datasets and analysis Neele Falk and Gabriella Lapesa.
Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022.](https://aclanthology.org/2022.acl-long.379/)