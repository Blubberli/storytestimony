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
import torch.nn.functional as F
from bert_classifier import BERT, load_checkpoint

# Evaluation


random.seed(42)
np.random.seed(42)


def predict(model, test_loader, result_folder, test_set):
    predicted_labels = []
    predicted_probabilities = []
    model.eval()
    with torch.no_grad():
        for (text), _ in tqdm(test_loader):
            current_batchsize = len(text)
            labels = torch.ones(current_batchsize)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)

            _, output = output
            current_predicted_probs = F.softmax(torch.tensor(output).detach(), dim=-1).tolist()
            current_predicted_labels = torch.argmax(output, 1).tolist()

            predicted_labels.extend(current_predicted_labels)
            predicted_probabilities.extend(current_predicted_probs)

    predicted_probabilities = [el[1] for el in predicted_probabilities]
    test_set["story_prob"] = predicted_probabilities
    test_set["predicted_label"] = predicted_labels

    test_set.to_csv("%s/predictions.csv" % result_folder, index=False, sep="\t")


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
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                       batch_first=True,
                       fix_length=512, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [(args.text_col, text_field)]
    test_set = pd.read_csv(args.test_data, sep="\t")

    test = TabularDataset(path=args.test_data, format='TSV', fields=fields, skip_header=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    print("loaded test dataset from %s" % args.test_data)

    model = BERT(args.classification_model).to(device)
    print("loaded best model from %s" % args.classification_model)
    post_texts = list(test_set[args.text_col])
    predict(model=model, test_loader=test_iter, result_folder=args.result_folder, test_set=testcsv)
