from sklearn.metrics import f1_score, classification_report
import numpy as np
import pandas as pd
import torch
import argparse

from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

import torch.optim as optim
import random

random.seed(42)
np.random.seed(42)


class BERT(nn.Module):
    """BERT model: returns the prediction and the cross-entropy loss. Is loaded from 'model path'"""

    def __init__(self, model_path):
        super(BERT, self).__init__()

        self.encoder = BertForSequenceClassification.from_pretrained(model_path)
        print("loaded bert model from %s" % model_path)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea


def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    """Save the model and that state_dict"""
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def train(model, optimizer, train_loader, valid_loader, num_epochs, destination_folder,
          best_valid_loss=float("Inf")):
    eval_every = len(train_loader) // 2
    running_loss = 0.0
    valid_running_loss = 0.0
    valid_running_f1 = 0.0
    global_step = 0
    best_valid_f1 = 0.0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (text, labels), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (text, labels), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)
                        text = text.to(device)
                        output = model(text, labels)
                        loss, out = output
                        y_pred = torch.argmax(out, 1).tolist()
                        y_true = labels.tolist()
                        valid_running_f1 += f1_score(y_true=y_true, y_pred=y_pred, average="macro")
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_valid_f1 = valid_running_f1 / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                valid_running_f1 = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid F1: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss, average_valid_f1))

                # checkpoint
                if best_valid_f1 < average_valid_f1:
                    # best_valid_loss = average_valid_loss
                    best_valid_f1 = average_valid_f1
                    save_checkpoint(destination_folder + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(destination_folder + '/' + 'metrics.pt', train_loss_list, valid_loss_list,
                                 global_steps_list)

    save_metrics(destination_folder + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def evaluate(model, test_loader, result_folder):
    y_pred = []
    y_true = []
    y_scores = []
    predictions_path = result_folder + "/predictions.csv"
    report_path = result_folder + "/classification_report.csv"
    model.eval()
    with torch.no_grad():
        for (text, labels), _ in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
            y_scores.extend(torch.sigmoid(output).tolist())
    with open(predictions_path, "w") as f:
        f.write("gold label\tpredicted label\tprobability\n")
        for i in range(len(y_pred)):
            f.write(str(y_true[i]) + "\t" + str(y_pred[i]) + "\t" + str(y_scores[i]) + "\n")
    f.close()
    report = classification_report(y_true, y_pred, labels=[1, 0], digits=2, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(report_path, sep="\t")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", type=str)
    parser.add_argument("target_folder", type=str)
    parser.add_argument("temp_folder", type=str)
    parser.add_argument("max_seqlen", type=int, default=512)
    parser.add_argument("result_folder", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    # GPU if available, otherwise CPU
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    # init the tokenizer that corresponds to the model
    model_path = "%s" % args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("loaded tokenizer from %s" % model_path)

    # encoding can be handled by torchtext
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                       batch_first=True,
                       fix_length=args.max_seqlen, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    # save the converted file to the source folder
    fields = [('post_text', text_field), ('label', label_field)]
    # read training and validation data from source folder
    traincsv = pd.read_csv("%s/train.csv" % args.source_folder, sep="\t")
    valcsv = pd.read_csv("%s/val.csv" % args.source_folder, sep="\t")

    # read test data from target folder
    testcsv = pd.read_csv("%s/test.csv" % args.target_folder, sep="\t")

    # select only text and label column and save to the tsv file in a temporary directory
    traincsv = traincsv[["post_text", "label"]]
    valcsv = valcsv[["post_text", "label"]]
    testcsv = testcsv[["post_text", "label"]]

    traincsv.to_csv("%s/train.tsv" % args.temp_folder, sep="\t", index=False)
    valcsv.to_csv("%s/val.tsv" % args.temp_folder, sep="\t", index=False)

    testcsv.to_csv("%s/test.tsv" % args.temp_folder, sep="\t", index=False)

    # TabularDataset: load from the tsv file from source folder

    train_data, valid, test = TabularDataset.splits(path=args.temp_folder, train='train.tsv',
                                                    validation='val.tsv',
                                                    test='test.tsv', format='TSV', fields=fields,
                                                    skip_header=True)

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Iterators
    train_iter = BucketIterator(train_data, batch_size=16, sort_key=lambda x: len(x.post_text),
                                device=device, shuffle=True, train=True, sort=False, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.post_text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)
    # init bert model
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    # retrieve the list of training labels
    training_labels = list(pd.read_csv("%s/train.tsv" % args.source_folder, sep="\t")["label"])

    # train the model
    train(model=model, optimizer=optimizer, train_loader=train_iter,
          valid_loader=valid_iter, destination_folder=args.result_folder, num_epochs=args.epochs)
    # load the best model after trained for max epochs
    best_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    # load best model from checkpoint
    load_checkpoint(args.result_folder + '/model.pt', best_model)
    if args.test:
        # evaluate the model on the test set
        report = evaluate(best_model, test_iter, args.result_folder)

        print("results...\n")
        print("precision non-story %.2f\trecall non-story %.2f\tF1 non-story %.2f" % (
            report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"]))
        print("precision story %.2f\trecall story %.2f\tF1 story %.2f" % (
            report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"]))
