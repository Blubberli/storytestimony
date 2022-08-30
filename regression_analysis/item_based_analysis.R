library(dplyr)
library(effects)
library(relaimpo)
library(jtools)
library(tidyr)
library(car)
library(caret)
library(standardize)

# datenset:
data <- read.csv("./result_files/instance_based_results.csv", header=TRUE, sep="\t")
# only bert
data <- data%>% filter(model=="BERT")
# only trained on RegRoom, Europolis or all
data <- data %>%
  mutate(trainingSetup = case_when(trainCorpus == 'regroom'~ 'RegRoom',
                                   trainCorpus == 'europolis' ~ 'Europolis',
                                   trainCorpus == 'mixed'  ~ 'all'))

# remove the other training instances
data <-data %>% filter(!is.na(trainingSetup))
# test corpus: europolis, regroom
data <- data %>%
  mutate(testCorpus = case_when(testCorpus == 'regroom'~ 'RegRoom',
                                testCorpus == 'europolis' ~ 'Europolis'))
data <-data %>% filter(!is.na(testCorpus))
unique(data$trainingSetup)
unique(data$testCorpus)

# false positives:
false_positives <- data%>% filter(predictedLabel == 1 & label == 0)
# 776 instances
nrow(false_positives)
# false negatives:
false_negatives <- data%>% filter(predictedLabel == 0 & label == 1)
# 1213
nrow(false_negatives)

# extract features of false positives
false_positives_feats <- subset(false_positives, select=c(flesch, negative_adjectives_component,
                                                          positive_nouns_component, auxliliary, 
                                                          named_entities, postlength, 
                                                          subordinate_conj, failure_component,
                                                          well_being_component, positive_adjectives_component, 
                                                          trust_verbs_component, affect_friends_and_family_component, 
                                                          polarity_nouns_component, respect_component,
                                                          politeness_component, economy_component, 
                                                          polarity_verbs_component, positive_verbs_component, 
                                                          past_tense, lsa_average_top_three_cosine,
                                                          virtue_adverbs_component, certainty_component,
                                                          social_order_component, action_component,
                                                          hyper_verb_noun_Sav_Pav, All_AWL_Normed,
                                                          McD_CD, content_poly,COCA_spoken_Bigram_Frequency, 
                                                          COCA_spoken_Frequency_AW, Brysbaert_Concreteness_Combined_AW, 
                                                          mattr50_aw, mtld_original_aw, LD_Mean_Accuracy,
                                                          personal_pronouns, adverbs, objects_component, 
                                                          WN_Mean_Accuracy))
# scale and center all IVs
false_positives_feats <- false_positives_feats %>% mutate_if(is.numeric, scale, scale = T)
false_negative_feats <- subset(false_negatives, select=c(flesch, negative_adjectives_component,
                                                         positive_nouns_component, auxliliary, 
                                                         named_entities, postlength, 
                                                         subordinate_conj, failure_component,
                                                         well_being_component, positive_adjectives_component, 
                                                         trust_verbs_component, affect_friends_and_family_component, 
                                                         polarity_nouns_component, respect_component,
                                                         politeness_component, economy_component, 
                                                         polarity_verbs_component, positive_verbs_component, 
                                                         past_tense, lsa_average_top_three_cosine,
                                                         virtue_adverbs_component, certainty_component,
                                                         social_order_component, action_component,
                                                         hyper_verb_noun_Sav_Pav, All_AWL_Normed,
                                                         McD_CD, content_poly,COCA_spoken_Bigram_Frequency, 
                                                         COCA_spoken_Frequency_AW, Brysbaert_Concreteness_Combined_AW, 
                                                         mattr50_aw, mtld_original_aw, LD_Mean_Accuracy,
                                                         personal_pronouns, adverbs, objects_component, 
                                                         WN_Mean_Accuracy))
# scale and center all IVs
false_negative_feats <- false_negative_feats %>% mutate_if(is.numeric, scale, scale = T)
# add the categorical IV to the DF of false positives
false_positives_feats$trainingSetup <- false_positives$trainingSetup
false_positives_feats$testCorpus <- false_positives$testCorpus
false_positives_feats$storyProb <- false_positives$storyProb

# map the scores to the range 0-0.5 but inverse them. meaning that a very high probability for stories 
# will now be close to 0. then log transform the probability. do the same for false negatives
# the log will spread the many datapoints that are berts confident predictions. in theory then the regression
# should benefit from that. the borderline cases are not much affected by the log.
false_positives_feats$logStoryProb <- log(1-false_positives$storyProb)

# add the categorical IV to the DF of false negatives
false_negative_feats$trainingSetup <- false_negatives$trainingSetup
false_negative_feats$testCorpus <- false_negatives$testCorpus
false_negative_feats$storyProb <- false_negatives$storyProb
false_negative_feats$logStoryProb <- log(false_negatives$storyProb)
summary(false_negative_feats$logStoryProb)
summary(false_positives_feats$logStoryProb)

# set reference level to Europolis
false_positives_feats$trainingSetup <- factor(false_positives_feats$trainingSetup, levels = c("Europolis", "RegRoom", "all"))
false_positives_feats$trainingSetup <- relevel(false_positives_feats$trainingSetup, ref="Europolis")
false_positives_feats$testCorpus <- factor(false_positives_feats$testCorpus, levels = c("Europolis","RegRoom"))
false_positives_feats$testCorpus <- relevel(false_positives_feats$testCorpus, ref="Europolis")


false_negative_feats$trainingSetup <- factor(false_negative_feats$trainingSetup, levels = c("Europolis", "RegRoom", "all"))
false_negative_feats$trainingSetup <- relevel(false_negative_feats$trainingSetup, ref="Europolis")
false_negative_feats$testCorpus <- factor(false_negative_feats$testCorpus, levels = c("Europolis","RegRoom"))
false_negative_feats$testCorpus <- relevel(false_negative_feats$testCorpus, ref="Europolis")

### ---- step-wise regression for false positives:

# simples model only contains features
fp_feats <- lm(logStoryProb~flesch+ negative_adjectives_component+
                    positive_nouns_component+ auxliliary+ 
                    named_entities+ postlength+ 
                    subordinate_conj+ failure_component+
                    well_being_component+ positive_adjectives_component+ 
                    trust_verbs_component+ affect_friends_and_family_component+ 
                    polarity_nouns_component+ respect_component+
                    politeness_component+ economy_component+ 
                    polarity_verbs_component+ positive_verbs_component+ 
                    past_tense+ lsa_average_top_three_cosine+
                    virtue_adverbs_component+ certainty_component+
                    social_order_component+ action_component+
                    hyper_verb_noun_Sav_Pav+ All_AWL_Normed+
                    McD_CD+ content_poly+COCA_spoken_Bigram_Frequency+ 
                    COCA_spoken_Frequency_AW+ Brysbaert_Concreteness_Combined_AW+ 
                    mattr50_aw+ mtld_original_aw+ LD_Mean_Accuracy+
                    personal_pronouns+ adverbs+ objects_component+ 
                    WN_Mean_Accuracy, data=false_positives_feats)

# 3.3% expl var
summary(fp_feats)

# run stepAIC to reduce features
stepAIC(fp_feats)

# simple model with reduced features
fp_feats<- lm(logStoryProb ~ auxliliary + subordinate_conj + respect_component + 
                     politeness_component + economy_component + past_tense + lsa_average_top_three_cosine + 
                     certainty_component + All_AWL_Normed + COCA_spoken_Bigram_Frequency + 
                     mattr50_aw + personal_pronouns + adverbs, data = false_positives_feats)
# 3.3% expl var
summary(fp_feats)
# add testCorpus
feats_test_fp <- lm(logStoryProb ~ (auxliliary + subordinate_conj + respect_component + 
                                      politeness_component + economy_component + past_tense + lsa_average_top_three_cosine + 
                                      certainty_component + All_AWL_Normed + COCA_spoken_Bigram_Frequency + 
                                      mattr50_aw + personal_pronouns + adverbs) + testCorpus, data = false_positives_feats)
# 5.1 % expl var
summary(feats_test_fp)

# add train Corpus
feats_test_train_fp <- lm(logStoryProb ~ (auxliliary + subordinate_conj + respect_component + 
                                            politeness_component + economy_component + past_tense + lsa_average_top_three_cosine + 
                                            certainty_component + All_AWL_Normed + COCA_spoken_Bigram_Frequency + 
                                            mattr50_aw + personal_pronouns + adverbs) +testCorpus+trainingSetup , data = false_positives_feats)
# 5.3% expl var
summary(feats_test_train_fp)

fp_two_way <- lm(logStoryProb ~  (auxliliary + subordinate_conj + respect_component + 
                                    politeness_component + economy_component + past_tense + lsa_average_top_three_cosine + 
                                    certainty_component + All_AWL_Normed + COCA_spoken_Bigram_Frequency + 
                                    mattr50_aw + personal_pronouns + adverbs)*testCorpus +
                   (auxliliary + subordinate_conj + respect_component + 
                      politeness_component + economy_component + past_tense + lsa_average_top_three_cosine + 
                      certainty_component + All_AWL_Normed + COCA_spoken_Bigram_Frequency + 
                      mattr50_aw + personal_pronouns + adverbs)*trainingSetup + testCorpus*trainingSetup, data=false_positives_feats)
# 8.3%
summary(fp_two_way)
# add interactions
fp_interactions <- lm(logStoryProb ~ (auxliliary + subordinate_conj + respect_component + 
                                             politeness_component + economy_component + past_tense + lsa_average_top_three_cosine + 
                                             certainty_component + All_AWL_Normed + COCA_spoken_Bigram_Frequency + 
                                             mattr50_aw + personal_pronouns + adverbs)*trainingSetup*testCorpus, data = false_positives_feats)

#8.2% expl var
summary(fp_interactions)

# run stepAIC to reduce number of IVs
stepAIC(fp_interactions)

fp_final_model <-lm(logStoryProb ~ auxliliary + subordinate_conj + respect_component + 
                                            politeness_component + economy_component + lsa_average_top_three_cosine + 
                                            All_AWL_Normed + COCA_spoken_Bigram_Frequency + mattr50_aw + 
                                            personal_pronouns + adverbs + trainingSetup + testCorpus + 
                                            respect_component:trainingSetup + politeness_component:trainingSetup + 
                                            COCA_spoken_Bigram_Frequency:trainingSetup + mattr50_aw:trainingSetup + 
                                            auxliliary:testCorpus + subordinate_conj:testCorpus + respect_component:testCorpus + 
                                            politeness_component:testCorpus + economy_component:testCorpus + 
                                            lsa_average_top_three_cosine:testCorpus + COCA_spoken_Bigram_Frequency:testCorpus + 
                                            trainingSetup:testCorpus + respect_component:trainingSetup:testCorpus + 
                                            politeness_component:trainingSetup:testCorpus, data = false_positives_feats)
#9.7% expl var
summary(fp_final_model)
#normalized GVIF ok
vif(fp_final_model)


# check all models and their previous model for signifiance
# no sign. improvement
anova(fp_feats, feats_test_fp)
# no sign improvement
anova(feats_test_fp, feats_test_train_fp)
# *** : sign. improvement
anova(feats_test_train_fp, fp_two_way)
anova(fp_two_way, fp_final_model)
# display most sign. predictors
anova(fp_final_model)

# ----------------false negatives-------------

fn_feats <- lm(logStoryProb~flesch+ negative_adjectives_component+
                    positive_nouns_component+ auxliliary+ 
                    named_entities+ postlength+ 
                    subordinate_conj+ failure_component+
                    well_being_component+ positive_adjectives_component+ 
                    trust_verbs_component+ affect_friends_and_family_component+ 
                    polarity_nouns_component+ respect_component+
                    politeness_component+ economy_component+ 
                    polarity_verbs_component+ positive_verbs_component+ 
                    past_tense+ lsa_average_top_three_cosine+
                    virtue_adverbs_component+ certainty_component+
                    social_order_component+ action_component+
                    hyper_verb_noun_Sav_Pav+ All_AWL_Normed+
                    McD_CD+ content_poly+COCA_spoken_Bigram_Frequency+ 
                    COCA_spoken_Frequency_AW+ Brysbaert_Concreteness_Combined_AW+ 
                    mattr50_aw+ mtld_original_aw+ LD_Mean_Accuracy+
                    personal_pronouns+ adverbs+ objects_component+ 
                    WN_Mean_Accuracy, data=false_negative_feats)
# 18.3% expl var
summary(fn_feats)

# reduce features with stepAIC

stepAIC(fn_feats)
# model with reduced features
feats_fn <- lm(logStoryProb ~ positive_nouns_component + auxliliary + 
                 postlength + subordinate_conj + failure_component + well_being_component + 
                 respect_component + economy_component + past_tense + certainty_component + 
                 All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                 Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                 personal_pronouns + adverbs + objects_component, data = false_negative_feats)
# 18.9 %
summary(feats_fn)

# add test corpus
feats_test_fn <- lm(logStoryProb ~ (positive_nouns_component + auxliliary + 
                                      postlength + subordinate_conj + failure_component + well_being_component + 
                                      respect_component + economy_component + past_tense + certainty_component + 
                                      All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                                      Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                                      personal_pronouns + adverbs + objects_component)+testCorpus, data = false_negative_feats)
# 23.3 %
summary(feats_test_fn)
# add train setup
feats_test_train_fn <- lm(logStoryProb ~ (positive_nouns_component + auxliliary + 
                                            postlength + subordinate_conj + failure_component + well_being_component + 
                                            respect_component + economy_component + past_tense + certainty_component + 
                                            All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                                            Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                                            personal_pronouns + adverbs + objects_component)+testCorpus+trainingSetup, data = false_negative_feats)
# 30.1 % expl var
summary(feats_test_train_fn)
# add interactions
two_way_fn <- lm(logStoryProb ~ (positive_nouns_component + auxliliary + 
                                   postlength + subordinate_conj + failure_component + well_being_component + 
                                   respect_component + economy_component + past_tense + certainty_component + 
                                   All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                                   Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                                   personal_pronouns + adverbs + objects_component)*testCorpus + 
                                  (positive_nouns_component + auxliliary + postlength + subordinate_conj + failure_component + well_being_component + 
                                  respect_component + economy_component + past_tense + certainty_component + 
                                  All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                                 Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                                personal_pronouns + adverbs + objects_component)*trainingSetup + 
                                trainingSetup*testCorpus, data = false_negative_feats)

# 36 % expl var
summary(two_way_fn)
# add three-way interaction
interactions_fn <- lm(logStoryProb ~ (positive_nouns_component + auxliliary + 
                                                postlength + subordinate_conj + failure_component + well_being_component + 
                                                respect_component + economy_component + past_tense + certainty_component + 
                                                All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                                                Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                                                personal_pronouns + adverbs + objects_component)*testCorpus*trainingSetup, data = false_negative_feats)
stepAIC(interactions_fn)
fn_final_model <- lm(logStoryProb ~ positive_nouns_component + auxliliary + 
                                             postlength + subordinate_conj + failure_component + well_being_component + 
                                             respect_component + economy_component + past_tense + certainty_component + 
                                             All_AWL_Normed + McD_CD + content_poly + COCA_spoken_Bigram_Frequency + 
                                             Brysbaert_Concreteness_Combined_AW + mtld_original_aw + LD_Mean_Accuracy + 
                                             personal_pronouns + adverbs + testCorpus + trainingSetup + 
                                             positive_nouns_component:testCorpus + auxliliary:testCorpus + 
                                             postlength:testCorpus + subordinate_conj:testCorpus + failure_component:testCorpus + 
                                             respect_component:testCorpus + economy_component:testCorpus + 
                                             certainty_component:testCorpus + All_AWL_Normed:testCorpus + 
                                             McD_CD:testCorpus + content_poly:testCorpus + COCA_spoken_Bigram_Frequency:testCorpus + 
                                             Brysbaert_Concreteness_Combined_AW:testCorpus + mtld_original_aw:testCorpus + 
                                             LD_Mean_Accuracy:testCorpus + positive_nouns_component:trainingSetup + 
                                             auxliliary:trainingSetup + postlength:trainingSetup + subordinate_conj:trainingSetup + 
                                             failure_component:trainingSetup + well_being_component:trainingSetup + 
                                             respect_component:trainingSetup + economy_component:trainingSetup + 
                                             past_tense:trainingSetup + certainty_component:trainingSetup + 
                                             All_AWL_Normed:trainingSetup + McD_CD:trainingSetup + content_poly:trainingSetup + 
                                             COCA_spoken_Bigram_Frequency:trainingSetup + Brysbaert_Concreteness_Combined_AW:trainingSetup + 
                                             mtld_original_aw:trainingSetup + LD_Mean_Accuracy:trainingSetup + 
                                             personal_pronouns:trainingSetup + testCorpus:trainingSetup + 
                                             postlength:testCorpus:trainingSetup + subordinate_conj:testCorpus:trainingSetup + 
                                             failure_component:testCorpus:trainingSetup + respect_component:testCorpus:trainingSetup + 
                                             economy_component:testCorpus:trainingSetup + certainty_component:testCorpus:trainingSetup + 
                                             McD_CD:testCorpus:trainingSetup + content_poly:testCorpus:trainingSetup + 
                                             COCA_spoken_Bigram_Frequency:testCorpus:trainingSetup + Brysbaert_Concreteness_Combined_AW:testCorpus:trainingSetup + 
                                             mtld_original_aw:testCorpus:trainingSetup + LD_Mean_Accuracy:testCorpus:trainingSetup, 
                                           data = false_negative_feats)
# 40% expl var
summary(fn_final_model)
# vif ok
vif(fn_final_model)

# test significance between models
anova(feats_fn, feats_test_fn)
#***
anova(feats_test_fn, feats_test_train_fn)
#***
anova(feats_test_train_fn, two_way_fn)
#***
anova(two_way_fn, fn_final_model)

plot(allEffects(fn_final_model), multiline=T,ylim = c(-10.0,0.0),ylab="log(prob(reports))",cex.lab=1.5, main="", axes=list(grid=T,x=list(rotate=15, cex=1.2), y=list(cex=1.2)), ci.style="band", lattice=list(key.args=list(x=.20,y=.25,corner=c(0,0),padding.text=0.9, cex=1.0, cex.title=0.0)))
plot(allEffects(fp_final_model), multiline=T,ylim = c(-10.0,0.0),ylab="log(prob(reports))",cex.lab=1.5, main="", axes=list(grid=T,x=list(rotate=15, cex=1.2), y=list(cex=1.2)), ci.style="band", lattice=list(key.args=list(x=.20,y=.25,corner=c(0,0),padding.text=0.9, cex=1.0, cex.title=0.0)))

