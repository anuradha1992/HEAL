import csv
import nltk
import pickle
import numpy as np
nltk.download('punkt')
import json
from sentence_transformers import SentenceTransformer, util
import torch

print("Loading embeddings dump...")
with open("../dump.pkl", 'rb') as fid:
     dictionary = pickle.load(fid)
print("finished loading dump")

corpus_embeddings = dictionary['embeddings']
corpus_sentences = dictionary['sentences']
corpus_ids = dictionary['ids']


print("Loading cluster embeddings dump...")
with open("../cluster_dump.pkl", 'rb') as fid:
     cluster_dictionary = pickle.load(fid)
print("finished loading dump")

cluster_embeddings = cluster_dictionary['embeddings']
cluster_sentences = cluster_dictionary['sentences']
cluster_dialog_ids = cluster_dictionary['dialog_ids']
cluster_numbers = cluster_dictionary['cluster_nos']

with open('../topic_dict.txt') as f:
  topic_dict = json.loads(f.read())

print("Done loading topic dict")

print("cluster embeddings: ", len(cluster_embeddings))
print("cluster sentences: ", len(cluster_sentences))
print("cluster dialog ids: ", len(cluster_dialog_ids))
print("cluster numbers: ", len(cluster_numbers))

print("Length of 1 embedding: ", len(cluster_embeddings[0]))

'''
cluster embeddings:  47109
cluster sentences:  47109
cluster dialog ids:  47109
cluster dict:  47092
'''

query_embeddings = []
query_sentences = []
query_ids = []

count = 0
    
with open('../data/test.csv', 'rt', encoding='utf-8') as infile:

  readCSV = csv.reader(infile, delimiter=',')
  next(readCSV)
  #new_row = ['conversation id', 'post title', 'author', 'dialog turn', 'text', 'compound', 'sentiment', 'emotion prediction']

  for row in readCSV:

    author = row[2].strip()
    turn = int(row[3].strip())

    if turn == 1 and 'speaker' in author:
      id_ = row[0].strip() # subreddit + "-" + type_ + "-" + str(id_)
      query = row[1].strip() + " | " + row[4].strip()

      query_embedding = corpus_embeddings[corpus_ids.index(id_)]
      query_embeddings.append(query_embedding)
      query_ids.append(id_)
      query_sentences.append(query)

with open('emb_dump/query_embeddings_red_test.pkl', "wb") as fOut:
    pickle.dump({'query_ids': query_ids, 'query_sentences': query_sentences, 'embeddings': query_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

print("Done writing to query file")

'''print("Loading from file...")

with open('prev_emb_dump/query_embeddings_red_test.pkl', 'rb') as fid:
     data = pickle.load(fid)
     query_embeddings = data['embeddings']
     query_sentences = data['query_sentences']
     query_ids = data['query_ids']'''


print("Converting to cuda ...")

cluster_embeddings = torch.Tensor(cluster_embeddings)
cluster_embeddings = cluster_embeddings.to('cuda')
#cluster_embeddings = util.normalize_embeddings(cluster_embeddings)

query_embeddings = torch.Tensor(query_embeddings)
query_embeddings = query_embeddings.to('cuda')
#query_embeddings = util.normalize_embeddings(query_embeddings)
hits = util.semantic_search(query_embeddings, cluster_embeddings, top_k=1)

print("Done computing hits ...")


with open('./similar-individual/red_test_prev_emb.csv', 'a', encoding='utf-8') as f1:

  writer = csv.writer(f1, delimiter=str(','), lineterminator='\n') 
  new_row = ['query id', 'query text', 'score', 'matching id', 'matching text', 'cluster no', 'cluster keywords']

  writer.writerow(new_row)

  for i in range(0, len(hits)):
    #print(hits[i])  # [{'corpus_id': 6, 'score': 0.5224307775497437}]
    result = hits[i][0]
    idx = result['corpus_id']
    score = result['score']
    cluster_no = cluster_numbers[idx]
    #print("Query: ", query_sentences[i])
    #print(cluster_dialog_ids[idx], cluster_sentences[idx], " (Score: {:.4f})".format(score), " Cluster no: ", cluster_no, " Keywords: ", topic_dict[str(cluster_no-1)])
    #print("---")

    new_row = [query_ids[i], query_sentences[i], round(float(score), 4), cluster_dialog_ids[idx], cluster_sentences[idx], cluster_no, str(topic_dict[str(cluster_no-1)])]
    writer.writerow(new_row)

print("Done writing to file ...")




