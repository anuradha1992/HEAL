import csv
from nltk.tokenize import sent_tokenize

# ==========================================================

tot_responses = 0
response_dict = {}

with open('./all_listener_responses.csv', 'a', encoding='utf-8') as outfile:

  writer = csv.writer(outfile, delimiter=str(','), lineterminator='\n')        
  new_row = ['cluster_id', 'dialog_id', 'response']
  writer.writerow(new_row)

  tot_clusters = 4363

  for i in range(1, tot_clusters+1):

    sep_res_count = 0

    with open('./clustered_dialogues/'+str(i)+'.csv', 'r', encoding='utf-8') as infile:

      readCSV = csv.reader(infile, delimiter=',')
      #next(readCSV)

      # offmychest-dyadic-343495  343495  offmychest  My son's mother took him from me and moved to another state 6 hours away. listener_1  2 I am so sorry that this is happening to you. It really bothers me when mothers won’t allow the father to see their child especially when the father hasn’t done any wrong doing. I hope you find a good lawyer and in result I hope you get the time you deserve with your son. Best of luck  0.8971  positive  sympathizing
      for row in readCSV:

        try:

          id_ = row[0]
          actor = row[4]
          turn = int(row[5])
          response = row[6]

          if turn == 2 and 'listener' in actor:

            #print(response)

            arr = sent_tokenize(response)

            for sent in arr:

              writer.writerow([i, id_, sent])
              tot_responses += 1
              sep_res_count += 1

        except Exception as e:

          print("Exception: ", row)
          print(e)

      response_dict[i] = sep_res_count


import json

with open('response_counts.txt', 'w') as file:
  file.write(json.dumps(response_dict)) # use `json.loads` to do the reverse

import plotly.graph_objects as go

# ========================= Categories ==============================

fig = go.Figure([go.Bar(x=list(response_dict.keys()), y=list(response_dict.values()))])
fig.update_layout(
  height=450,
  width=900,
  title_text='Distribution of responses per each cluster',
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
          text='No. of listener responses',
          font=dict(
              family='Times New Roman',
              size=18,
              color='black'
          )
      )
  ),
  )
fig.update_yaxes(type="log")
fig.update_traces(marker_color='black')
fig.show()

print("Total no. of responses: ", tot_responses)
print("Average no. of responses per cluster: ", round(tot_responses/tot_clusters,2))
print("Max no. of responses per cluster:", max(list(response_dict.values())))
print("Min no. of responses per cluster:", min(list(response_dict.values())))

print("===")
for key, value in response_dict.items():

  if value == 0:
    print(key, 0)

print("===")

vals = list(response_dict.values())
vals.sort()
print(vals[0:50])

'''Total no. of responses:  245707
Average no. of responses per cluster:  56.32
Max no. of responses per cluster: 60237
Min no. of responses per cluster: 0'''











