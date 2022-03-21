from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time

def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    # Compute cosine similarity scores
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    #print(cos_scores)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)
    #print(top_k_values)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            #print(top_val_large, top_idx_large)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities

# =================================================================== main code ===================================================================

# Read the data from the file
print("Loading dump")
with open("../dump.pkl", 'rb') as fid:
     dictionary = pickle.load(fid)
print("finished loading dump")

with open("../indices_dump.pkl", 'rb') as fid:
     ind_dict = pickle.load(fid)
print("finished loading indices dump")

train_ids = ind_dict['train_ids']

train_embeddings = list(np.array(dictionary['embeddings'])[train_ids])
train_sentences = list(np.array(dictionary['sentences'])[train_ids])
train_ids = list(np.array(dictionary['ids'])[train_ids])


additional_sentences = []
additional_embeddings = []
additional_ids = []


for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:

  for i in range(0, 10):
  #for i in range(0, 1):

    multi = 100000

    if i != 9:

      corpus_embeddings = train_embeddings[i*multi:(i+1)*multi]
      corpus_sentences = train_sentences[i*multi:(i+1)*multi]
      corpus_ids = train_ids[i*multi:(i+1)*multi]

    else:

      corpus_embeddings = train_embeddings[i*multi:]
      corpus_sentences = train_sentences[i*multi:]
      corpus_ids = train_ids[i*multi:]

    if i > 0:
      corpus_embeddings = np.concatenate([additional_embeddings, corpus_embeddings], axis=0).tolist()
      corpus_sentences = additional_sentences + corpus_sentences
      corpus_ids = additional_ids + corpus_ids

    
    print(len(corpus_embeddings[5]))
    print(corpus_sentences[5])
    print(corpus_ids[5])

    print()
    print(len(corpus_embeddings))
    print(len(corpus_sentences))
    print(len(corpus_ids))

    # ======================================== clustering ================================================

    print("Start clustering")
    start_time = time.time()

    

    #Two parameter to tune:
    #min_cluster_size: Only consider cluster that have at least 25 elements (30 similar sentences)
    #threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = community_detection(corpus_embeddings, min_community_size=5, threshold=threshold, init_max_size=1)
    
     
    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    file1 = open("stats.txt","a") 
    file1.write("Threshold = "+str(threshold)+"\n")
    file1.write("Clustering done after {:.2f} sec".format(time.time() - start_time))
    file1.write("\n")

    additional_sentences = []
    additional_embeddings = []
    additional_ids = []

    for i, cluster in enumerate(clusters):
      for sentence_id in cluster:
        additional_sentences.append(corpus_sentences[sentence_id])
        additional_embeddings.append(corpus_embeddings[sentence_id])
        additional_ids.append(corpus_ids[sentence_id])

    #print(clusters)


  with open("../clusters/stressors/clustering-"+str(threshold)+".csv", "a+") as f:
      #fieldnames = ['', 'Title', 'Top 30 tokens', '(Token, Probaility of appearing in topic, Count in topic, Count in corpus)']
      writer = csv.writer(f, delimiter=str(','))        
      #writer.writeheader()

      #Print all cluster / communities
      for i, cluster in enumerate(clusters):
          #print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
          row = ['Cluster: {}'.format(i+1), "{} elements".format(len(cluster))]
          writer.writerow(row)
          row = []
          writer.writerow(row)
          for sentence_id in cluster:
              #print("\t", corpus_sentences[sentence_id])  
              arr = corpus_sentences[sentence_id].split(" | ")
              title = arr[0]
              if len(arr) == 2:
                text = arr[1]
              else:
                text = " ".join(arr[1:])
              row = [corpus_ids[sentence_id], title, text]
              writer.writerow(row)
          row = []
          writer.writerow(row)


  documents_clustered = 0
  labels = []
  values = []
  for i in range(len(clusters)):
    documents_clustered += len(clusters[i])
    labels.append(i+1)
    values.append(len(clusters[i]))

  print("No. of clusters = ", len(clusters))
  if len(clusters) > 0:
    print("Size of the largest cluster = ", len(clusters[0]))
  print("Total no. of documents clustered = ", documents_clustered)
  print("Percentage of documents clustered = ", (documents_clustered/1195092)*100)

  
  file1.write("No. of clusters = "+str(len(clusters))+"\n") 
  if len(clusters) > 0:
    file1.write("Size of the largest cluster = "+str(len(clusters[0]))+"\n") 
  file1.write("Total no. of documents clustered = "+str(documents_clustered)+"\n") 
  file1.write("Percentage of documents clustered = "+str((documents_clustered/1195092)*100)+"\n") 
  file1.write("\n\n") 
  file1.close()

  '''import plotly.graph_objects as go

  # ========================= Categories ==============================

  fig = go.Figure([go.Bar(x=labels, y=values)])
  fig.update_layout(
    height=450,
    width=900,
    title_text='Distribution of clusters for similarity threshold = {}'.format(threshold),
    xaxis_tickangle=-90,
    font=dict(
            family="Times New Roman",
            size=18,
            color="black"
        ),
    xaxis=go.layout.XAxis(
          title=go.layout.xaxis.Title(
              text='Cluster number',
              font=dict(
                  family='Times New Roman',
                  size=18,


                  color='black'
              )
          )
      ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='No. of elements',
            font=dict(
                family='Times New Roman',
                size=18,
                color='black'
            )
        )
    ),
    )
  fig.update_yaxes(type="log")
  fig.write_image(os.getcwd() + "/images/cluster_dist_" + str(threshold) + ".png")'''
























