import pandas as pd
import torch
import argparse
from tqdm import tqdm
# Preliminaries

from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import random
import numpy as np

# Models

from transformers import AutoTokenizer
from bert_classifier import BERT, load_checkpoint

# Evaluation


random.seed(42)
np.random.seed(42)


def predict(model, test_loader, result_folder, post_texts):
    y_pred = []
    y_scores = []
    predictions_path = result_folder + "/predictions.csv"
    model.eval()
    with torch.no_grad():
        for (text, labels), _ in tqdm(test_loader):
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_scores.extend(torch.softmax(output).tolist())
    with open(predictions_path, "w") as f:
        f.write("post_text\tpredicted_label\tprobability(storytelling)\n")
        for i in range(len(y_pred)):
            f.write(
                str(post_texts[i]) + "\t" + str(y_pred[i]) + "\t" + str(y_scores[i]) + "\n")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data", type=str)
    parser.add_argument("text_col", type=str)
    parser.add_argument("max_seqlen", type=int)
    parser.add_argument("model_path", type=str)
    parser.add_argument("classification_model", type=str)
    parser.add_argument("result_folder", type=str)
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    # GPU if available, otherwise CPU
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    # init the tokenizer that corresponds to the model
    model_path = "%s" % args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("loaded tokenizer from %s" % model_path)

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                       batch_first=True,
                       fix_length=args.max_seqlen, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    testcsv = pd.read_csv(args.test_data, sep="\t")
    testcsv = testcsv[[args.text_col]]
    testcsv["label"] = [0] * len(testcsv)
    testcsv.to_csv("tmp_test.tsv", sep="\t", index=False)

    fields = [('post_text', text_field), ('label', label_field)]

    test = TabularDataset(path="tmp_test.tsv", format='TSV', fields=fields, skip_header=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    model = BERT(args.classification_model).to(device)
    print("loaded best model from %s" % args.classification_model)
    post_texts = list(testcsv[args.text_col])
    predict(model=model, test_loader=test_iter, result_folder=args.result_folder, post_texts=post_texts)
