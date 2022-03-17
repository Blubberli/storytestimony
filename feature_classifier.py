from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report
from bow_model import report_average
import random
import argparse

random.seed(1)


def train_run(outfile, train, test):
    """Train a random forest classifier with features and save the predictions to outfile"""
    print("save predictions to %s" % outfile)
    out = open(outfile, "w")
    out.write("gold label\tpredicted label\tprobability\n")
    train_x = filter_data_for_feats(train)
    test_x = filter_data_for_feats(test)
    train_y = train['label']
    test_y = test['label']
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)
    for i in range(len(y_pred)):
        out.write(str(test_y[i]) + "\t" + str(y_pred[i]) + "\t" + str(probabilities[i]) + "\n")
    out.close()
    importance = classifier.feature_importances_
    feature_dic = {}
    # summarize feature importance
    for i, v in enumerate(importance):
        feature_dic[list(train_x.columns)[i]] = v
    feature_dic = dict(sorted(feature_dic.items(), key=lambda item: item[1]))
    for k, v in feature_dic.items():
        print("feature : %s, importance: %.2f" % (k, v))
    forest_report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    return forest_report


def train_features_onefold(training_path, test_path, prediction_path):
    """Train random forest on one dataset and test on 10 (outdomain)"""
    train = pd.read_csv("%s/train.csv" % training_path, sep="\t")
    reports_forest = []
    for i in range(0, 10):
        test = pd.read_csv("%s/split%i/test.csv" % (test_path, i), sep="\t")
        outfile = "%s/split%d/predictions.csv" % (prediction_path, i)
        forest_report = train_run(outfile, train, test)
        reports_forest.append(forest_report)
    print("results...\n\n")
    print("random forest...\n")
    mean_dic = report_average(reports_forest)
    print("precision non-story %.2f\trecall non-story %.2f\tF1 non-story %.2f" % (
        mean_dic["0"]["precision"], mean_dic["0"]["recall"], mean_dic["0"]["f1-score"]))
    print("precision story %.2f\trecall story %.2f\tF1 story %.2f" % (
        mean_dic["1"]["precision"], mean_dic["1"]["recall"], mean_dic["1"]["f1-score"]))


def train_features_10fold(training_path, test_path, prediction_path):
    """Train the random forest classifier on 10 splits and save each result for the test set."""
    reports_forest = []
    for i in range(0, 10):
        train = pd.read_csv("%s/split%d/train.csv" % (training_path, i), sep="\t")
        test = pd.read_csv("%s/split%i/test.csv" % (test_path, i), sep="\t")
        outfile = "%s/split%d/predictions.csv" % (prediction_path, i)
        forest_report = train_run(outfile, train, test)
        reports_forest.append(forest_report)
    print("results...\n\n")
    print("random forest...\n")
    mean_dic = report_average(reports_forest)
    print("precision non-story %.2f\trecall non-story %.2f\tF1 non-story %.2f" % (
        mean_dic["0"]["precision"], mean_dic["0"]["recall"], mean_dic["0"]["f1-score"]))
    print("precision story %.2f\trecall story %.2f\tF1 story %.2f" % (
        mean_dic["1"]["precision"], mean_dic["1"]["recall"], mean_dic["1"]["f1-score"]))


def filter_data_for_feats(dataframe):
    """This defines the 51 features used for the RandomForest. Returns a dataframe with only the features as columns."""
    dataframe = dataframe[
        ['postlength', 'past_tense', 'personal_pronouns', 'adverbs', 'auxliliary', 'subordinate_conj', 'named_entities',
         'negative_adjectives_component', 'social_order_component', 'action_component', 'positive_adjectives_component',
         'joy_component', 'affect_friends_and_family_component', 'fear_and_digust_component', 'politeness_component',
         'polarity_nouns_component', 'polarity_verbs_component', 'virtue_adverbs_component', 'positive_nouns_component',
         'respect_component', 'trust_verbs_component', 'failure_component', 'well_being_component', 'economy_component',
         'certainty_component', 'positive_verbs_component', 'objects_component', 'mattr50_aw', 'mtld_original_aw',
         'hdd42_aw', 'MRC_Familiarity_AW', 'MRC_Imageability_AW', 'Brysbaert_Concreteness_Combined_AW',
         'COCA_spoken_Range_AW', 'COCA_spoken_Frequency_AW', 'COCA_spoken_Bigram_Frequency', 'COCA_spoken_bi_MI2',
         'McD_CD', 'Sem_D', 'All_AWL_Normed', 'LD_Mean_RT', 'LD_Mean_Accuracy', 'WN_Mean_Accuracy',
         'lsa_average_top_three_cosine', 'content_poly', 'hyper_verb_noun_Sav_Pav', 'flesch', 'gunningFog',
         'chars_per_word', 'syllables_per_word', 'long_words']

    ]
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", type=str)
    parser.add_argument("result_folder", type=str)
    parser.add_argument("--outdomain", action="store_true")
    args = parser.parse_args()

    input_data = args.source_folder
    test_data = args.source_folder
    if args.outdomain:
        train_features_onefold(training_path=input_data, test_path=test_data,
                               prediction_path=args.result_folder)
    else:
        train_features_10fold(training_path=input_data, test_path=test_data,
                              prediction_path=args.result_folder)
