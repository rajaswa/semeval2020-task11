# IMPORTS
import time
import numpy as np
import torch
from torch import nn
from flair.embeddings import (
    Sentence,
    RoBERTaEmbeddings,
    BertEmbeddings,
    XLNetEmbeddings,
    OpenAIGPT2Embeddings,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from span_identification.data import *
from span_identification.models import *

# SEED
seed_val = 0
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DEVICE
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using ", device)

# CONFIG
model_name = "roberta-base"  # BertEmbeddings, XLNetEmbeddings, OpenAIGPT2Embeddings
epochs = 10
lr = 1e-4
bs = 32
alpha = 0.55
fname = "roberta_task_1"
SAVE_PATH = "./"

# DATALOADING
label_df = pd.read_csv("datasets/train-task1-SI.labels", delimiter="\t", header=None)
label_df.columns = ["id", "start", "end"]

embedding = RoBERTaEmbeddings(
    pretrained_model_name_or_path=model_name, layers="-2,-3,-4"
)  # BertEmbeddings, XLNetEmbeddings, OpenAIGPT2Embeddings
samples = get_train_dataset(embedding, label_df)
train_samples, valid_samples = train_test_split(
    samples, test_size=0.03, random_state=seed_val
)

train_dset = Task1Dataset(train_samples)
valid_dset = Task1Dataset(valid_samples)
train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_dset, batch_size=bs, shuffle=False, num_workers=0)

# MODEL
model = Task1Model(768 * 3)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=10e-8, weight_decay=1e-5)
criterion = nn.BCELoss()

# TRAINING
train_loss = []
validation_loss = []
precision_list, f1_list, recall_list, acc_list = [], [], [], []
max_acc = -999
min_loss = 9999999
start_time = time.time()

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_loader):
        title, line, token, token_label, line_label = data
        title, line, token, token_label, line_label = (
            title.to(device),
            line.to(device),
            token.to(device),
            token_label.to(device),
            line_label.to(device),
        )

        optimizer.zero_grad()

        out_line, out_token = model(title, line, token)
        loss = alpha * criterion(out_line.reshape(-1), line_label.float()) + (
            1 - alpha
        ) * criterion(out_token.reshape(-1), token_label.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 1000 == 999:
            print(
                "[%d, %5d] loss: %.3f time: %.3f"
                % (epoch + 1, i + 1, running_loss / 1000, time.time() - start_time)
            )
            running_loss = 0.0

    print("\nEPOCH ", epoch + 1, " TRAIN LOSS = ", epoch_loss / len(train_dset))
    train_loss.append(epoch_loss / len(train_dset))

    model.eval()
    preds = []
    ground_truth = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            title, line, token, token_label, line_label = data
            title, line, token, token_label, line_label = (
                title.to(device),
                line.to(device),
                token.to(device),
                token_label.to(device),
                line_label.to(device),
            )

            out_line, out_token = model(title, line, token)

            predicted = out_token.reshape(-1)
            for pred in predicted:
                preds.append(int(round(pred.item())))
            for label in token_label:
                ground_truth.append(int(label.item()))

    preds = np.array(preds)
    ground_truth = np.array(ground_truth)
    prec, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, preds, average="binary"
    )
    accuracy = accuracy_score(ground_truth, preds)
    print(
        "TRAINING EPOCH",
        epoch + 1,
        " PREC = ",
        prec,
        " RECALL = ",
        recall,
        " F1 = ",
        f1,
        "ACC = ",
        accuracy,
    )

    val_loss = 0.0
    model.eval()
    preds = []
    ground_truth = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            title, line, token, token_label, line_label = data
            title, line, token, token_label, line_label = (
                title.to(device),
                line.to(device),
                token.to(device),
                token_label.to(device),
                line_label.to(device),
            )

            out_line, out_token = model(title, line, token)
            loss = alpha * criterion(out_line.reshape(-1), line_label.float()) + (
                1 - alpha
            ) * criterion(out_token.reshape(-1), token_label.float())
            val_loss += loss.item()

            predicted = out_token.reshape(-1)
            for pred in predicted:
                preds.append(int(round(pred.item())))
            for label in token_label:
                ground_truth.append(int(label.item()))

    print("EPOCH ", epoch + 1, " VAL LOSS = ", val_loss / len(valid_dset))
    validation_loss.append(val_loss / len(valid_dset))
    model.train()

    preds = np.array(preds)
    ground_truth = np.array(ground_truth)
    prec, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, preds, average="binary"
    )
    accuracy = accuracy_score(ground_truth, preds)
    precision_list.append(prec)
    f1_list.append(f1)
    recall_list.append(recall)
    acc_list.append(accuracy)
    print(
        "VALIDATION EPOCH",
        epoch + 1,
        " PREC = ",
        prec,
        " RECALL = ",
        recall,
        " F1 = ",
        f1,
        "ACC = ",
        accuracy,
        "\n",
    )

    if accuracy > max_acc:
        print("Model optimized, saving weights ...\n")
        torch.save(model.state_dict(), SAVE_PATH + fname + ".pt")
        max_acc = accuracy

# PLOTS
fig = plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.legend()
plt.show()
fig.savefig(SAVE_PATH + fname + "loss.png", dpi=400)

fig = plt.figure()
plt.plot(precision_list, label="Precision")
plt.plot(recall_list, label="Recall")
plt.plot(f1_list, label="F1")
plt.plot(acc_list, label="Accuracy")
plt.legend()
plt.show()
fig.savefig(SAVE_PATH + fname + "metrics.png", dpi=400)

# INFERENCE
model.load_state_dict(torch.load(SAVE_PATH + fname + ".pt"))
model.to(device)
model.eval()

# DEV
article_fnames = os.listdir("datasets/dev-articles/")
article_fnames = ["datasets/dev-articles/" + f for f in article_fnames]
article_ids, start, end = [], [], []
for path in article_fnames:
    article_id, spans = predict(model, path, embedding)
    for span in spans:
        article_ids.append(article_id)
        start.append(span[0])
        end.append(span[1])
dev_df = pd.DataFrame()
dev_df["article_id"] = article_ids
dev_df["start"] = start
dev_df["end"] = end
dev_df.to_csv(
    SAVE_PATH + fname + "_dev.txt", header=None, index=None, sep="\t", mode="a"
)

# TEST
article_fnames = os.listdir("datasets/test-articles/")
article_fnames = ["datasets/test-articles/" + f for f in article_fnames]
article_ids, start, end = [], [], []
for path in article_fnames:
    article_id, spans = predict(model, path, embedding)
    for span in spans:
        article_ids.append(article_id)
        start.append(span[0])
        end.append(span[1])
test_df = pd.DataFrame()
test_df["article_id"] = article_ids
test_df["start"] = start
test_df["end"] = end
test_df.to_csv(
    SAVE_PATH + fname + "_test.txt", header=None, index=None, sep="\t", mode="a"
)
