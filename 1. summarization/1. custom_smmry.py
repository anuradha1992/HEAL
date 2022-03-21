import csv
import requests 
from pytorch_transformers import RobertaTokenizer
import nltk
import math
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import *
from nltk.tokenize import WhitespaceTokenizer
import numpy as np
from nltk.corpus import stopwords  
import re 
import sys

# Load your usual SpaCy model (one of SpaCy English models)
import spacy
nlp = spacy.load('en_core_web_sm')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

stemmer = PorterStemmer()
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  

transition_phrases = [

'thus', 'for example', 'for instance', 'namely', 'to illustrate', 'in other words', 'in particular', 'specifically', 'such as',

'on the contrary', 'contrarily', 'notwithstanding', 'but', 'however', 'nevertheless', 'in spite of', 'in contrast', 'yet', 'on one hand', 'on the other hand', 'rather', 
'or', 'nor', 'conversely', 'at the same time', 'while this may be true',

'and', 'in addition to', 'furthermore', 'moreover', 'besides', 'than', 'too', 'also', 'both-and', 'another', 'equally important', 'second', 'etc.', 'again', 
'further', 'last', 
'finally', 'not only-but also', 'as well as', 'in the second place', 'next', 'likewise', 'similarly', 'in fact', 'as a result', 'consequently', 'in the same way', 
'for example', 
'for instance', 'however', 'thus', 'therefore', 'otherwise',

'after that', 'afterward', 'then', 'next', 'last', 'at last', 'at length', 'at first', 'formerly', 'another', 'finally', 
'meanwhile', 'at the same time', 
'afterwards', 'subsequently', 'in the meantime', 'eventually', 'concurrently', 'simultaneously',

'although', 'at least', 'still', 'even though', 'granted that', 'while it may be true', 'in spite of', 'of course',

'similarly', 'likewise', 'in like fashion', 'in like manner', 'analogous to',

'above all', 'indeed', 'of course', 'certainly', 'surely', 'in fact', 'really', 'in truth', 'again', 'besides', 'also', 'furthermore', 'in addition',

'specifically', 'especially', 'in particular', 'to explain', 'to list', 'to enumerate', 'in detail', 'namely', 'including',

'for example', 'for instance', 'to illustrate', 'thus', 'in other words', 'as an illustration', 'in particular',

'so that', 'with the result that', 'consequently', 'hence', 'accordingly', 'for this reason', 'therefore', 'because', 'due to', 
'as a result', 'in other words', 'then',

'therefore', 'finally', 'consequently', 'thus', 'in conclusion', 'as a result', 'accordingly',

'for this purpose', 'to this end', 'with this in mind', 'with this purpose in mind', 'therefore']

# SMMRY Algo
# ===================================================================================================================================

def transition_start(first_sent, dialog_turn):

  if dialog_turn == 1:
    for phrase in transition_phrases:
      if first_sent.lower().startswith(phrase):
        return True
    return False
  else:
    return False

def smmry(text, sent_count, dialog_turn):

  # some preprocessing to omit text within brackets and replace u with you. 

  text = re.sub("[\(\[].*?[\)\]]", "", text)
  text = text.replace(' u ', ' you ')

  formatted_text = re.sub('[^a-zA-Z]', ' ', text )
  formatted_text = re.sub(r'\s+', ' ', formatted_text)

  doc = nlp(text)

  fdist = {}
  word_arr = nltk.word_tokenize(formatted_text.lower())

  # preparing a frequency dictionary without considering stop words
  
  for word in word_arr:
    if not word in stop_words:
      word = wnl.lemmatize(word)
      if word not in fdist.keys():
          fdist[word] = 1
      else:
          fdist[word] += 1

  sent_arr = nltk.sent_tokenize(text)
  sent_score_arr = []
  summary_arr = []

  sent_arr_coref_resolved = nltk.sent_tokenize(doc._.coref_resolved)

  # compute scores for each sentence

  for sent in sent_arr:
    score = 0
    token_arr = nltk.word_tokenize(sent.lower())
    for word in token_arr:
      word = wnl.lemmatize(word)
      if word in fdist.keys():
        score += fdist[word]

    sent_score_arr.append(score/len(token_arr))

  sent_score_arr = np.array(sent_score_arr)

  all_ind_arr = sent_score_arr.argsort()[-len(sent_score_arr):][::-1]

  ind_arr_unsorted = sent_score_arr.argsort()[-sent_count:][::-1]

  ind_arr = np.sort(ind_arr_unsorted) 

  summary = ''
  changed_first = False

  if len(ind_arr) > 0:

    try:

      ind = ind_arr[0]
      first_sent = sent_arr[ind]

      while (first_sent != sent_arr_coref_resolved[ind] or transition_start(first_sent, dialog_turn)):
        changed_first = True
        for index in all_ind_arr:
          if index < ind:
            ind = index
            break
        first_sent = sent_arr[ind]
        if ind == 0:
          break
      summary = summary + first_sent + ' '     
      
      if (changed_first):
        first_ind = ind
        sent_score_modified = sent_score_arr[first_ind+1:]
        ind_arr_unsorted = sent_score_modified.argsort()[-(sent_count-1):][::-1]
        ind_arr_next = np.sort(ind_arr_unsorted) 
        
        for i in range(0, len(ind_arr_next)):
          ind = (first_ind+1) + ind_arr_next[i]
          if i == len(ind_arr_next) - 1:
            summary = summary + sent_arr[ind]
          else:
            summary = summary + sent_arr[ind] + ' '
      
      else:
        for i in range(1, len(ind_arr)):
          ind = ind_arr[i]
          if i == len(ind_arr) - 1:
            summary = summary + sent_arr[ind]
          else:
            summary = summary + sent_arr[ind] + ' '

      return summary

    except Exception as e:

      print("EXCEPTION occured")
      return text

  else:
    print("EXCEPTION occured: length of sentence array is not > 0")
    return text


# Run the code for the dataset - processing line by line
# ===================================================================================================================================


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

subreddits = ["mentalhealthsupport", "anxietyhelp", "depressed", "depression", "depression_help", "offmychest", "sad", "suicidewatch"]

for subreddit in subreddits:

  for type_ in ["dyadic", "multi"]:

    print(subreddit, " - ", type_)

    count = 0
    summarized_count = 0


    with open('../original/'+subreddit+'_'+type_+'.csv', 'r', encoding='utf-8') as infile:

      with open('../summarized/'+subreddit+'_'+type_+'.csv', 'a', encoding='utf-8') as outfile:

          readCSV = csv.reader(infile, delimiter=',')
          next(readCSV)

          writer = csv.writer(outfile, delimiter=str(','), lineterminator='\n')        
          new_row = ['conversation id', 'subreddit', 'post title', 'author', 'dialog turn', 'text', 'compound', 'sentiment', 'emotion prediction']
          writer.writerow(new_row)

          for row in readCSV:

              text = row[5]
              dialog_turn = int(row[4])
              
              # preprocessing: replace urls with tag URL
              text = re.sub(r'http\S+', 'URL', text)

              uttr_ids = tokenizer.encode(text)

              if len(uttr_ids) + 2 > 100: # So, the length can fit the input to the transformer

                sent_arr = nltk.sent_tokenize(text)
                avg_tokens_per_sent = len(uttr_ids) / len(sent_arr)
                next_sent_count = -1
                if avg_tokens_per_sent + 2 > 100:
                  next_sent_count = 1
                else:
                  next_sent_count = math.floor(100/avg_tokens_per_sent)
                
                if next_sent_count > 5:
                  next_sent_count = 5 # At maximum we are going to have a 5 sentence summary
                
                summary = smmry(text, next_sent_count, dialog_turn)
                
                uttr_ids = tokenizer.encode(summary)

                while (len(uttr_ids) + 2 > 100 and next_sent_count >= 1):
                  
                  prev_sent_count = next_sent_count
                  avg_tokens_per_sent = len(uttr_ids) / prev_sent_count

                  if avg_tokens_per_sent + 2 > 100:
                    next_sent_count = 1
                  else:
                    next_sent_count = min(prev_sent_count - 1, math.floor(100/avg_tokens_per_sent))

                  summary = smmry(summary, next_sent_count, dialog_turn)
                  
                  uttr_ids = tokenizer.encode(summary)
                  
                  if next_sent_count == 1:  # we summarize the text until one sentence remain, even if its length is > 100.
                    break

                summarized_count += 1
                
                writer.writerow([row[0], row[1], row[2], row[3], row[4], summary, row[6], row[7], row[8]])
                
              else:

                writer.writerow([row[0], row[1], row[2], row[3], row[4], text, row[6], row[7], row[8]])
              

              count += 1
    

    print("Turn count: ", count)
    print("Summaried turn count: ", summarized_count)



