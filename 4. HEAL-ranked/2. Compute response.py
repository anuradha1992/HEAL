import boto3
import csv
import smart_open
import nltk
import pickle
import numpy as np
nltk.download('punkt')
import json
from random import randrange
import random

with open('../topic_dict.txt') as f:
  topic_dict_big = json.loads(f.read())

print("Done loading topic dict")

# {"nodes": [{"id": "t1", "label": "commit, killing, death, painless, option", "value": 11856}, 
with open('../HEAL/nodes/topics.txt') as f:
  topic_nodes = json.loads(f.read())["nodes"]

# {"nodes": [{"id": "r1", "label": "I am sorry you are feeling this.", "value": 1025},
with open('../HEAL/edges/responses.txt') as f:
  response_edges = json.loads(f.read())["edges"]

# {"edges": [{"from": "t1", "to": "r1", "value": 273}, {"from": "t2", "to": "r1", "value": 30}, 
with open('../HEAL/nodes/responses.txt') as f:
  response_nodes = json.loads(f.read())["nodes"]

topic_dict = {}
for node in topic_nodes:
  topic_dict[node['id']] = {'label': node['label'], 'value': node['value'], 'response_ids': [], 'response_values': []}

response_dict = {}
for node in response_nodes:
  response_dict[node['id']] = {'label': node['label'], 'value': node['value']}

for edge in response_edges:
  topic_dict[edge['from']]['response_ids'].append(edge['to'])
  topic_dict[edge['from']]['response_values'].append(edge['value'])

'''for key, value in topic_dict.items():
  response_values = value['response_values']
  response_ids = value['response_ids']
  if len(response_ids) > 0:
    response_values, response_ids = zip(*sorted(zip(response_values, response_ids)))
    value['response_values'] = response_values
    value['response_ids'] = response_ids'''

for key, value in topic_dict.items():
  response_values = value['response_values']
  response_ids = value['response_ids']

  d = []

  if len(response_ids) > 0:
    response_sizes = []
    for i in range(len(response_ids)):
      id_ = response_ids[i]
      if id_ in response_dict:
        #response_sizes.append(response_dict[id_]['value'])
        d.append({
          'id': id_,
          'value': response_values[i],
          'size': response_dict[id_]['value']
          })
      else:
        #response_sizes.append(-1)
        d.append({
          'id': id_,
          'value': response_values[i],
          'size': -1
          })

    d=sorted(d, key=lambda i: (i['value'], i['size']),reverse=True)

    response_ids_sorted = []
    response_values_sorted = []

    for ele in d:
      response_ids_sorted.append(ele['id'])
      response_values_sorted.append(ele['value'])

    #response_sizes, response_ids = zip(*sorted(zip(response_sizes, response_ids)))
    #response_sizes, response_values = zip(*sorted(zip(response_sizes, response_values)))
    value['response_values'] = response_values_sorted
    value['response_ids'] = response_ids_sorted

count = 0


with open('./responses/red-test-heal-ranked-top5.csv', 'a', encoding='utf-8') as f1:

  writer = csv.writer(f1, delimiter=str(','), lineterminator='\n')      
  new_row = ['query id', 'query text', 'cluster_keywords', 'response']
  #new_row = ['query id', 'query text', 'score', 'matching id', 'matching text', 'cluster no', 'cluster keywords']
  writer.writerow(new_row)

  with open('./similar-individual/red_test_prev_emb.csv', 'r', encoding='utf-8') as infile: 

    #new_row = ['conversation id', 'post title', 'author', 'dialog turn', 'text', 'compound', 'sentiment', 'emotion prediction']
    #new_row = ['query id', 'query text', 'score', 'matching id', 'matching text', 'cluster no', 'cluster keywords']

    readCSV = csv.reader(infile, delimiter=',')
    next(readCSV)
      
    for row in readCSV:

      score = float(row[2].strip())

      if score >= 0.8:

        cluster_no = int(row[5].strip()) # 1, 2, 3, 4, 5, ...

        #print("Query: ", row[1].strip())
        #print("Cluster no:", cluster_no)
        topic_dict_node = topic_dict["t"+str(cluster_no)]
        #print("Cluster keywords: ", topic_dict_big[str(cluster_no-1)])
        #print("Cluster keywords: ", topic_dict_node['label'])
        #print("Cluster size: ", topic_dict_node['value'])
        #print("Top-5 responses: ")
        
        response_values = topic_dict_node['response_values']
        response_ids = topic_dict_node['response_ids']

        #print("No. of responses: ", len(response_ids))

        c = 0
        for k in range(0, len(response_ids)):
          id_ = response_ids[k]
          if id_ in response_dict:
            #print(id_, response_dict[id_]['label'], response_values[k], response_dict[id_]['value'])
            #if c == 0:
            new_row = [row[0].strip(), row[1].strip(), topic_dict_big[str(cluster_no-1)], response_dict[id_]['label']]
            #else:
            #  new_row = ['', '', '', response_dict[id_]['label']]
            writer.writerow(new_row)
            c += 1
          if c == 5:
            break
        #print()

        count += 1
        print(count)



