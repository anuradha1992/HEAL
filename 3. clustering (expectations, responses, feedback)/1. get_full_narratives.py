import csv

# ==========================================================

cluster_dict = {}

with open('../clusters/stressors/clustering-0.85.csv', 'r', encoding='utf-8') as infile:
        
  readCSV = csv.reader(infile, delimiter=',')
  cluster_no = 0

  for row in readCSV:

    if len(row) > 0:

      if "Cluster:" in row[0]:
        cluster_no += 1

      else:
        
        id_ = row[0].strip()
        
        cluster_dict[id_] = cluster_no

# ==========================================================

print("Total no. of dialogues clustered: ", len(cluster_dict))

written_count = 0


subreddits = ["mentalhealthsupport", "anxietyhelp", "depressed", "depression", "depression_help", "offmychest", "sad", "suicidewatch"]

# conversation id subreddit post title  author  dialog turn text  compound  sentiment emotion prediction

for subreddit in subreddits:

  for type_ in ["dyadic", "multi"]:

    print(subreddit, " - ", type_)

    count = 0
    summarized_count = 0

    with open('../original/'+subreddit+'_'+type_+'.csv', 'r', encoding='utf-8') as infile:

      readCSV = csv.reader(infile, delimiter=',')
      next(readCSV)

      prev_id_ = -1
      id_ = -1
      dialog_turns = []

      for row in readCSV:

          try:
            id_ = row[0]

            if prev_id_ != -1 and prev_id_ != id_:
            
              full_id = subreddit+"-"+type_+"-"+prev_id_

              if full_id in cluster_dict:

                cluster_no = cluster_dict[full_id]

                with open('./clustered_dialogues/'+str(cluster_no)+'.csv', 'a', encoding='utf-8') as outfile:

                  writer = csv.writer(outfile, delimiter=str(','), lineterminator='\n')        
                  #new_row = ['conversation id', 'subreddit', 'post title', 'author', 'dialog turn', 'text', 'compound', 'sentiment', 'emotion prediction']
                  
                  for turn in dialog_turns:
                    new_row = [full_id] + turn
                    writer.writerow(new_row)

                  written_count += 1

              dialog_turns = []
              dialog_turns.append(row)

            else:

              dialog_turns.append(row)

            prev_id_ = id_

          except Exception as e:
            print("EXCEPTION:", row)
            print(e)


print("Total no. of dialogues written: ", written_count)




