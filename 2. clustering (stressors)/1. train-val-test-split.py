from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time
import numpy as np

import random

# Read the data from the file
print("Loading dump")
with open("../dump.pkl", 'rb') as fid:
     dictionary = pickle.load(fid)
print("finished loading dump")

corpus_embeddings = np.array(dictionary['embeddings'])
corpus_sentences = np.array(dictionary['sentences'])
corpus_ids = np.array(dictionary['ids'])

number = 119510*2
val_test_ids = random.sample(range(0, len(corpus_ids)), number)
print("done generating indices ", len(val_test_ids))

train_embeddings = []
train_sentences = []
train_ids = []

val_embeddings = []
val_sentences = []
val_ids = []

test_embeddings = []
test_sentences = []
test_ids = []


all_ids = list(range(len(corpus_ids)))
train_indices = list(set(all_ids) - set(val_test_ids))
print("done train indices ", len(train_indices))
test_indices = []
val_indices = []
for i in range(0, len(val_test_ids)):
  if i % 2 == 0:
    val_indices.append(val_test_ids[i])
  else:
    test_indices.append(val_test_ids[i])

print("done val incides ", len(val_indices))
print("done test incides ", len(val_indices))

'''for i in range(0, len(corpus_ids)):
  if i not in val_test_ids:
    train_embeddings.append(corpus_embeddings[i])
    train_sentences.append(corpus_sentences[i])
    train_ids.append(corpus_ids[i])

for i in val_test_ids:
  if i % 2 == 0:
    val_embeddings.append(corpus_embeddings[i])
    val_sentences.append(corpus_sentences[i])
    val_ids.append(corpus_ids[i])
  else:
    test_embeddings.append(corpus_embeddings[i])
    test_sentences.append(corpus_sentences[i])
    test_ids.append(corpus_ids[i])'''

print("Store file on disc")

indices = {'train_ids': train_indices, 'val_ids': val_indices, 'test_ids': test_indices}
with open("../indices_dump.pkl", "wb") as fOut:
    pickle.dump(indices, fOut)
print("finished dumping")
























