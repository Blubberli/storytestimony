library(ggplot2)
library(dplyr)
library(effects)
library(relaimpo)
library(jtools)
library(tidyr)
library(car)
library(caret)

# read in aggregated results
data <- read.csv("./result_files/aggregated_results.csv", header=TRUE, sep="\t")

# filter out all single out domain experiments and keep "indomain", 2vs1 and "mixed"
data <- data %>%
  mutate(trainingSetup = case_when(trainCorpus == 'regroom' & testCorpus == 'regroom' ~ 'indomain',
                                   trainCorpus == 'europolis' & testCorpus == 'europolis' ~ 'indomain',
                                   trainCorpus == 'cmv' & testCorpus == 'cmv' ~ 'indomain',
                                   trainCorpus == '2vs1'  ~ '2vs1',
                                   trainCorpus == 'mixed' ~ 'all'))

data <- data %>%
  mutate(testCorpus = case_when(testCorpus == 'regroom' ~ 'RegRoom',
                                testCorpus == 'europolis' ~ 'Europolis',
                                testCorpus == 'cmv'~ 'CMV'))
data <-data %>% filter(!is.na(trainingSetup))
# check whether trainCorpus variable only has three categories

data <- data %>%
  mutate(model = case_when(model == 'features' ~ 'features',
                           model == 'BoW' ~ 'BoW',
                           model == 'feedforwardNN'~ 'feedforwardNN',
                           model == 'BERT'~ 'BERT',
                           model == 'bert-europarl-adapt'~ 'BERTeuroparl',
                           model == 'bert-argue-adapt'~ 'BERTargue',
                           model == 'bert-mixed-adapt'~ 'BERTmixed'))

unique(data$trainingSetup)
# total instances: 630
nrow(data)

# sort factors and relevel. reference level is indomain
data$trainingSetup <- factor(data$trainingSetup, levels = c("indomain" ,"2vs1", "all"))
data$trainingSetup <- relevel(data$trainingSetup, ref="indomain")

# reference level is features
data$model <- factor(data$model, levels = c("features", "BoW", "feedforwardNN", "BERT", "BERTeuroparl", "BERTargue", "BERTmixed"))
data$model <- relevel(data$model, ref="features")

unique(data$testCorpus)
# reference level is Europolis
data$testCorpus <- factor(data$testCorpus, levels = c("Europolis", "CMV", "RegRoom"))
data$testCorpus <- relevel(data$testCorpus, ref="Europolis")

# check that split has no effect
split <- lm(f1macro~split, data=data)
summary(split, center=T)

# variation from test test: 11.6%
testset <- lm(f1macro~testCorpus, data=data)
summary(testset)
# 23.6 % expl var
train_test <- lm(f1macro~testCorpus + trainingSetup, data=data)
summary(train_test)
# 44.1%
train_test_model <- lm(f1macro~testCorpus + trainingSetup + model, data=data)
summary(train_test_model, center=T)

# 65.3 %
two_way <- lm(f1macro~(testCorpus + trainingSetup + model)^2, data=data)
summary(two_way)

three_way <- lm(f1macro~(testCorpus + trainingSetup + model)^3, data=data)
summary(three_way)

# check significance between complex and nested models
# ***
anova(testset, train_test)
# ***
anova(train_test, train_test_model)
# ***
anova(train_test_model, two_way)
# ***
anova(two_way, three_way)

# check for multicollinearity
vif(three_way)
# save fit of the final model
fit <- anova(threeway)
fit

# identify the relative amount of variance explained by each predictor
explained_variance <-(fit[["Sum Sq" ]]/sum(fit[["Sum Sq" ]]))*100
fit$explvar <- explained_variance
fit[,c("Df", "explvar")]

# compute the effect of each term marginalized over the nested terms
eff <- allEffects(threeway)
eff

# create effect plots
plot(allEffects(three_way), multiline=T,main="", x.var="trainingSetup", ylim = c(0,1),cex.lab=1.5, ylab="F1macro",
     axes=list(grid=T,x=list(rotate=15, cex=1.2), y=list(cex=1.2)), ci.style="band",
     lattice=list(key.args=list(x=1.0,y=.65,corner=c(1,1),padding.text=0.5, cex=0.75, cex.title=0.0)))
