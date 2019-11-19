import os
import warnings
import numpy as np
from torch import nn
import torch
import math
import torch.optim as optim

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def get_class_labels(y):
    return np.unique(y)


def get_class_weights(y):
    """
    Returns the normalized weights for each class
    based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    weights = compute_class_weight('balanced', np.unique(y), y)

    d = {c: w for c, w in zip(np.unique(y), weights)}

    return d
def class_weigths(targets, to_pytorch=False):
    w = get_class_weights(targets)
    labels = get_class_labels(targets)
    if to_pytorch:
        return torch.FloatTensor([w[l] for l in sorted(labels)])
    return labels


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 50
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)


# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
print("###############EX1###################")
y_train = le.fit_transform(y_train)  # EX1
y_test = le.fit_transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

y_train_temp = list(le.inverse_transform(y_train))
y_test_temp = list(le.inverse_transform(y_test))
for i in range(10):
    print(y_train_temp[i], y_train[i])

# Define our PyTorch-based Dataset
print("###############EX2###################")
##Initializing train_set's and test_set's average length to zero, update later
train_set = SentenceDataset(X_train, y_train, word2idx,0)
test_set = SentenceDataset(X_test, y_test, word2idx,0)

for i in range(10):
    print(train_set.data[i], train_set.labels[i])

print("###############EX3###################")
print("calculating average length with and witout outliers...")


#############################################################################
# TRAIN LENGTHS, AVERAGE TRAIN LENGTH, TRAIN WORD EMBEDDINGS
#############################################################################

train_lengths = [len(sentence) for sentence in train_set.data]
train_lengths_without_outliers_1 = list(reject_outliers(np.array(train_lengths)))
train_avg_length = int(np.mean(train_lengths))
train_avg_length_without_outliers_1 = int(np.mean(train_lengths_without_outliers_1))
train_set.avg_length = train_avg_length_without_outliers_1
print("Average length with outliers is: ", train_avg_length)
print("Average length without outliers is: ", train_avg_length_without_outliers_1)
print("printing 5 word embeddings in the original and the transformed form using average legngth...")

train_word_embeddings = [train_set[index] for index in range(len(train_set))]

for i in range(5):
    print("WORD EMBEDDING ",i)
    print("sentence: ", X_train[i],", target: ",y_train_temp[i])
    print("sentence's word embedding: ",train_word_embeddings[i][0], ", label: ",train_word_embeddings[i][1] )
    print()

#############################################################################
# TEST LENGTHS, AVERAGE TEST LENGTH, TEST WORD EMBEDDINGS
#############################################################################
test_lengths = [len(sentence) for sentence in test_set.data]
test_lengths_without_outliers_1 = list(reject_outliers(np.array(test_lengths)))
test_avg_length = int(np.mean(test_lengths))
test_avg_length_without_outliers_1 = int(np.mean(test_lengths_without_outliers_1))
test_set.avg_length = test_avg_length_without_outliers_1

#EX4 - Define our PyTorch-based DataLoader

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)


train_word_embeddings = [x[0] for x in train_word_embeddings]
train_word_embeddings_torched = torch.torch.from_numpy(np.array(train_word_embeddings))
train_lengths_torched = torch.from_numpy(np.asarray(train_lengths))

# move the mode weight to cpu or gpu
model.to(DEVICE)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)

criterion =  nn.CrossEntropyLoss() #nn.CrossEntropyLoss().to(DEVICE) #nn.BCEWithLogitsLoss() , criterion = nn.CrossEntropyLoss().cuda()  # EX8
parameters = [] # EX8
for p in  model.parameters():
    if(p.requires_grad):
        parameters.append(p)


optimizer = optim.Adam(parameters,
                       lr=5e-4,
                       betas=(0.9, 0.999),
                       eps=1e-08,
                       weight_decay=1e-4)  # EX8



#############################################################################
# Training Pipeline
#############################################################################

losses_train = []
losses_test = []
prev_loss = 100
e = 0.1
for epoch in range(EPOCHS):
  
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, criterion)
    losses_train.append(train_loss)
    losses_test.append(test_loss)
    
    prev_loss = test_loss

    print('F1_train: {}'.format(f1_score(y_train_gold, y_train_pred, average="macro")))
    print('Accuracy_train: {}'.format(accuracy_score(y_train_gold, y_train_pred)))
    print('Recall_train: {}'.format(recall_score(y_train_gold, y_train_pred, average="macro")))
    print('F1_test: {}'.format(f1_score(y_test_gold, y_test_pred, average="macro")))
    print('Accuracy_test: {}'.format(accuracy_score(y_test_gold, y_test_pred)))
    print('Recall_test: {}'.format(recall_score(y_test_gold, y_test_pred, average="macro")))

losses_train_arr = np.array(losses_train)
losses_test_arr = np.array(losses_test)

fig = plt.figure()
plt.plot(losses_train,  label="train data")
plt.plot(losses_test,  label="test data")
fig.suptitle('Loss - epochs for both train and test set', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('cummulative running loss', fontsize=16)
plt.legend()
plt.show()