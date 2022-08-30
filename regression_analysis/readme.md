## Regression Analysis

This directory conains the results and code to analyze with linear regression.
The overall classifiaction results for each experiment is stored in `result_files/aggregated_results.csv`
The predictions from the models for each instance in the test data is stored in `result_files/instance_based_results.csv`

To run the code you need to have R installed and the following packages:

- ggplot2
- dplyr
- effects
- relaimpo
- jtools
- tidyr
- car
- caret
- standardize

## Aggregated Analysis
### Independent Variables
For further details have a look at section C in the Appendix

- trainCorpus :*training data 
  - *indomain*: Europolis, CMV, RegRoom
  - *2vs1*: concatenation of two
  - *mixed*: concatenation of all
- testCorpus: test data
  - *Europolis*, *CMV*, *RegRoom*
- model: which model was used
  - random forest, bag of words, feed forward, BERT, domain-adapted BERT
- split: which split was used for training / testing

### Dependent Variables

- precision
- recall
- f1
- f1macro

The code used to explain the variance of the results (F1 Macro) can be found in `aggregated_analysis.R`.

## Instance Based Results

For simplicity the regression analysis was conducted on a smaller set of results.
For further details have a look at section D in the Appendix
### Independent Variables

- trainCorpus :*training data 
  - *indomain*: Europolis, RegRoom
  - *mixed*: concatenation of all
- testCorpus: test data
  - *Europolis*, *RegRoom*
- model: which model was used
  - BERT and domain-adapted variants
- split: which split was used for training / testing
- linguistic features


### Dependent Variable

- probability of positive class (of a text containing a narrative / experience)

The code used to explain the variance of the results (probability(story)) can be found in `item_based_analysis.R`.
