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

# ==========================================================

tot_responses = 0
response_dict = {}

corpus_sentences = []
corpus_dialog_ids = []
corpus_cluster_ids = []

with open('./all_expectations.csv', 'r', encoding='utf-8') as infile:


      #writer.writerow([i, id_, sent])
  

      readCSV = csv.reader(infile, delimiter=',')
      next(readCSV)

      # offmychest-dyadic-343495  343495  offmychest  My son's mother took him from me and moved to another state 6 hours away. listener_1  2 I am so sorry that this is happening to you. It really bothers me when mothers won’t allow the father to see their child especially when the father hasn’t done any wrong doing. I hope you find a good lawyer and in result I hope you get the time you deserve with your son. Best of luck  0.8971  positive  sympathizing
      for row in readCSV:

        cluster_id = row[0]
        dialog_id = row[1]
        response = row[2]
        corpus_sentences.append(response)
        corpus_cluster_ids.append(cluster_id)
        corpus_dialog_ids.append(dialog_id)


# Model for computing sentence embeddings. We use one trained for similar questions detection
'''model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

corpus_sentences = list(corpus_sentences)
print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

import pickle

print("Writing to file ...")

file = open('expectations.pkl', 'wb')
pickle.dump(corpus_embeddings, file)
file.close()'''

print("finished writing to file ...")

print("Loading embeddings")

import pickle
with open('feedback.pkl', 'rb') as handle:
    corpus_embeddings = pickle.load(handle)

print("Total no. of embeddings: ", len(corpus_embeddings))


print("Start clustering")
start_time = time.time()

for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:

    #Two parameter to tune:
    #min_cluster_size: Only consider cluster that have at least 25 elements (30 similar sentences)
    #threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = community_detection(corpus_embeddings, min_community_size=2, threshold=threshold, init_max_size=1)

     
    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    file1 = open("expectation_stats_2.txt","a") 
    file1.write("Threshold = "+str(threshold)+"\n")
    print("Threshold = "+str(threshold)+"\n")
    file1.write("Clustering done after {:.2f} sec".format(time.time() - start_time))
    file1.write("\n")

    with open("../clusters/expectations/expectation-clustering-"+str(threshold)+".csv", "a+") as f:
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
                row = [corpus_cluster_ids[sentence_id], corpus_dialog_ids[sentence_id], corpus_sentences[sentence_id]]
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
    print("Percentage of documents clustered = ", (documents_clustered/32832)*100)


    file1.write("No. of clusters = "+str(len(clusters))+"\n") 
    if len(clusters) > 0:
      file1.write("Size of the largest cluster = "+str(len(clusters[0]))+"\n") 
    file1.write("Total no. of documents clustered = "+str(documents_clustered)+"\n") 
    file1.write("Percentage of documents clustered = "+str((documents_clustered/32832)*100)+"\n") 
    file1.write("\n\n") 
    file1.close()










