import string

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import argparse
from collections import Counter


def build_vocabulary():
    """Read all datasets and build the vocabulary of the top 5000 words using all datasets as corpus."""
    print("retrieve most frequent words ...")
    nlp = spacy.load("en_core_web_sm")
    data1 = pd.read_csv("europolis+/europolis_all_features.csv", sep="\t")
    data2 = pd.read_csv("cmv+/cmv_all_features.csv", sep="\t")
    data3 = pd.read_csv("regroom+/regroom_all_features.csv", sep="\t")
    all_comments = list(data1.post_text) + list(data2.post_text) + list(data3.post_text)
    words = []
    for doc in nlp.pipe(all_comments, disable=["parser"]):
        for token in doc:
            if token.lemma_ in string.punctuation or token.lemma_ == "\n":
                continue
            words.append(token.lemma_)
    freq_dic = Counter(words)
    most_common = freq_dic.most_common(5000)
    file = open("most_frequent_lemmas.txt", "w")
    for el in most_common:
        file.write(str(el[0]) + "\n")
    file.close()
    return freq_dic.most_common(5000)


def get_vocab():
    words = open("most_frequent_lemmas.txt", "r").readlines()
    return list(set([el.strip() for el in words]))


def report_average(reports):
    """Given a list of classifier reports, return a dictionary with the average of each class."""
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()
        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue
        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict


def train_run(train, test, predictions_path):
    """Train a random forest classifier with a bag-of-words input, absed on the word-frequency of unigram and bigrams (top 5000)"""
    # load spacy model
    nlp = spacy.load("en_core_web_sm")
    print("save predictions to %s" % predictions_path)
    out = open(predictions_path, "w")
    out.write("gold label\tpredicted label\tprobability\n")
    y_train = train['label']
    x_train = train['post_text']
    # test
    y_test = test['label']
    x_test = test['post_text']
    processed_train = []
    processed_test = []
    # convert all documents to lemmas
    print("lemmatize text...")
    for doc in nlp.pipe(x_train, disable=["parser"]):
        text = []
        for token in doc:
            text.append(token.lemma_)
        processed_train.append(" ".join(text))
    for doc in nlp.pipe(x_test, disable=["parser"]):
        text = []
        for token in doc:
            text.append(token.lemma_)
        processed_test.append(" ".join(text))
    print("train random forest...")
    # convert each document to a count-vector based on the top 5000 words
    vectorizer = CountVectorizer(max_features=5000, vocabulary=get_vocab(),
                                 ngram_range=(1, 2))
    # train the random forest with the count-vectors
    vectorizer.fit(processed_train)
    x_train = vectorizer.transform(processed_train).toarray()
    x_test = vectorizer.transform(processed_test).toarray()
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    for i in range(len(y_pred)):
        out.write(str(y_test[i]) + "\t" + str(y_pred[i]) + "\t" + str(probabilities[i]) + "\n")
    out.close()
    forest_report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    return forest_report


def train_bow_onefold(training_path, test_path, prediction_path):
    """Train bag-of-words classifier on one training set and evaluate on 10 test sets (outdomain)."""
    train = pd.read_csv("%s/train.csv" % training_path, sep="\t")
    reports_forest = []
    for i in range(0, 10):
        test = pd.read_csv("%s/split%i/test.csv" % (test_path, i), sep="\t")
        outfile = "%s/split%d/predictions.csv" % (prediction_path, i)
        forest_report = train_run(predictions_path=outfile, train=train, test=test)
        reports_forest.append(forest_report)
    print("results...\n\n")
    print("random forest...\n")
    mean_dic = report_average(reports_forest)
    print("precision non-story %.2f\trecall non-story %.2f\tF1 non-story %.2f" % (
        mean_dic["0"]["precision"], mean_dic["0"]["recall"], mean_dic["0"]["f1-score"]))
    print("precision story %.2f\trecall story %.2f\tF1 story %.2f" % (
        mean_dic["1"]["precision"], mean_dic["1"]["recall"], mean_dic["1"]["f1-score"]))


def train_bow_10fold(training_path, test_path, prediction_path):
    """Train bag-of-words classifier on 10 training sets and evaluate on 10 test sets (in-domain, cross-domain)"""
    reports_forest = []
    for i in range(0, 10):
        train = pd.read_csv("%s/split%d/train.csv" % (training_path, i), sep="\t")
        test = pd.read_csv("%s/split%i/test.csv" % (test_path, i), sep="\t")
        outfile = "%s/split%d/predictions.csv" % (prediction_path, i)
        forest_report = train_run(predictions_path=outfile, train=train, test=test)
        reports_forest.append(forest_report)
    print("results...\n\n")
    print("random forest...\n")
    mean_dic = report_average(reports_forest)
    print("precision non-story %.2f\trecall non-story %.2f\tF1 non-story %.2f" % (
        mean_dic["0"]["precision"], mean_dic["0"]["recall"], mean_dic["0"]["f1-score"]))
    print("precision story %.2f\trecall story %.2f\tF1 story %.2f" % (
        mean_dic["1"]["precision"], mean_dic["1"]["recall"], mean_dic["1"]["f1-score"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", type=str)
    parser.add_argument("result_folder", type=str)
    parser.add_argument("--outdomain", action="store_true")
    args = parser.parse_args()

    input_data = args.source_folder
    test_data = args.source_folder
    if args.outdomain:
        train_bow_onefold(training_path=input_data, test_path=test_data,
                          prediction_path=args.result_folder)
    else:
        train_bow_10fold(training_path=input_data, test_path=test_data,
                         prediction_path=args.result_folder)
