# IMPORTS
import pandas as pd
import flair
from flair.embeddings import Sentence
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import time
import re

# DEVICE
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
flair.device = device

# UTILS
def load_article(path):
    with open(path) as article:
        raw = article.read()
    lines = raw.split("\n")
    while "" in lines:
        lines.remove("")
    start_indices = []
    idx = 0
    for line in lines:
        start_indices.append(idx)
        idx += len(line)
        char = raw[idx]
        while char == "\n":
            idx += 1
            try:
                char = raw[idx]
            except:
                char = ""

    return lines, start_indices, raw


def get_word_embeddings(sentence, embedding):
    sentence = Sentence(sentence.rstrip("\n"))
    embedding.embed(sentence)
    word_embeds = [sentence[i].embedding for i in range(len(sentence))]
    return word_embeds


def get_token_spans(text):
    token_list = text.split()
    token_spans = []
    start_span = 0
    for i in range(len(token_list)):
        token_spans.append([start_span, start_span + len(token_list[i]) - 1])
        start_span += len(token_list[i]) + 1
    return token_spans


def label_tokens(path, label_df):
    lines, start_indices, raw = load_article(path)
    article_id = int(path[-13:-4])
    df_curr = label_df[label_df["id"] == article_id]
    df_curr.reset_index(inplace=True)

    label_spans = []
    for i in range(len(df_curr)):
        label_spans.append([df_curr["start"][i], df_curr["end"][i]])
    del df_curr

    token_list = []
    label_list = []
    for i in range(len(lines)):
        line = lines[i]
        token_list.append(line.split())
        token_spans = get_token_spans(line)
        token_spans = [
            [span[0] + start_indices[i], span[1] + start_indices[i]]
            for span in token_spans
        ]
        labels = [0] * len(token_spans)
        for i in range(len(token_spans)):
            token_span_curr = list(range(token_spans[i][0], token_spans[i][1] + 1))
            for label_span in label_spans:
                ground_truth_span = list(range(label_span[0], label_span[1] + 1))
                if not set(token_span_curr).isdisjoint(ground_truth_span):
                    labels[i] = 1

        label_list.append(labels)
    assert len(lines) == len(token_list) == len(label_list)
    return lines, token_list, label_list


def get_proximity_score(previous_preds):
    score = 0
    decay = len(previous_preds)
    for pred in previous_preds:
        score += pred / decay
        decay -= 1

    return score


def get_train_dataset(embedding, label_df):
    start_time = time.time()
    train_samples = []
    article_fnames = os.listdir("datasets/train-articles/")
    article_fnames = ["datasets/train-articles/" + f for f in article_fnames]
    count = 0
    for fname in article_fnames:
        lines, token_list, label_list = label_tokens(fname, label_df)
        title_embed = torch.mean(
            torch.stack(get_word_embeddings(lines[0], embedding), dim=1), dim=1
        )
        line_labels = [1 if 1 in labels else 0 for labels in label_list]
        for i in range(len(lines)):
            token_embeds = get_word_embeddings(lines[i], embedding)
            assert len(token_embeds) == len(label_list[i])
            line_embed = torch.mean(torch.stack(token_embeds, dim=1), dim=1)
            for j in range(len(token_embeds)):
                train_samples.append(
                    [
                        title_embed,
                        line_embed,
                        token_embeds[j],
                        label_list[i][j],
                        line_labels[i],
                    ]
                )
    print("Extracted Embeddings in", time.time() - start_time)
    return train_samples


def getWordSpans(text):
    text = re.sub("\w['‘’“”]\w", "aaa", text)
    wordlist = []

    def trans(text, pointer=0):
        if pointer == len(text) - 1:
            return True
        else:
            while text[pointer] == " " and pointer < len(text) - 1:
                pointer = pointer + 1
            s = pointer
            while not text[pointer] == " " and pointer < len(text) - 1:
                pointer = pointer + 1
            wordlist.append([s, pointer])
            return trans(text, pointer)

    try:
        trans(text)
    except:
        return -1
    if wordlist[-1][1] == wordlist[-1][0]:
        wordlist = wordlist[:-1]
    if text[-1].isalpha():
        wordlist[-1][1] += 1
    return wordlist


def getCharSpans(prediction, wordlist):
    charSpans = []

    def getSpan(prediction, wordlist):
        for i in range(len(prediction)):
            if i == 0:
                if prediction[i] == 1:
                    charSpans.append(wordlist[0][0])
            elif prediction[i] == 0 and prediction[i - 1] == 1:
                charSpans.append(wordlist[i - 1][1])
            elif prediction[i] == 1 and prediction[i - 1] == 0:
                charSpans.append(wordlist[i][0])
            if i == len(prediction) - 1 and prediction[i] == 1:
                charSpans.append(wordlist[-1][1])

    getSpan(prediction, wordlist)
    return [[charSpans[i], charSpans[i + 1]] for i in range(0, len(charSpans), 2)]


def process_preds(line_preds):
    if len(line_preds) == 1:
        line_preds[0] = 0
    else:
        for i in range(1, len(line_preds) - 1):
            if (line_preds[i - 1] == 1) & (line_preds[i + 1] == 1):
                line_preds[i] = 1
            if (line_preds[i - 1] == 0) & (line_preds[i + 1] == 0):
                line_preds[i] = 0
        if (line_preds[0] == 1) & (line_preds[1] == 0):
            line_preds[0] = 0
        if (line_preds[-1] == 1) & (line_preds[-2] == 0):
            line_preds[-1] = 0


def predict(model, path):
    article_id = int(path[-13:-4])
    article_spans = []
    model.eval()
    sig = nn.Sigmoid()

    lines, start_indices, raw = load_article(path)
    title_embed = torch.mean(
        torch.stack(get_word_embeddings(lines[0], embedding), dim=1), dim=1
    )

    for i in range(len(lines)):
        line_preds = []
        tokens = lines[i].split()
        token_start_index = []
        cursor = 0
        for j in range(len(tokens)):
            token_start_index.append(cursor)
            cursor += len(tokens[j]) + 1
        token_embeds = get_word_embeddings(lines[i], embedding)
        line_embed = torch.mean(torch.stack(token_embeds, dim=1), dim=1)
        with torch.no_grad():
            for j in range(len(token_embeds)):
                out_line, out_token = model(
                    title_embed, line_embed, token_embeds[j], predict=True
                )
                prediction = round((out_token).item())
                line_preds.append(prediction)

        process_preds(line_preds)
        if len(lines[i]) > 1:
            spans = getCharSpans(line_preds, getWordSpans(lines[i]))
            for span in spans:
                span[0] += start_indices[i]
                span[1] += start_indices[i] + 1
                article_spans.append(span)

    return article_id, article_spans


# DATASET
class Task1Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        title = self.samples[idx][0]
        line = self.samples[idx][1]
        token = self.samples[idx][2]
        token_label = torch.tensor(self.samples[idx][3])
        line_label = torch.tensor(self.samples[idx][4])

        return (title, line, token, token_label, line_label)
