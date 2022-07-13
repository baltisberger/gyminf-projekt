import time
import json
from functions import *

start_time = time.time()

with open('articles_preprocessed.json') as json_file:
    docs = json.load(json_file)

query = 'Ferienanspruch Arbeitnehmer pro Jahr'
query_preprocessed = get_preprocessed_text(query)

tfidf_sorted = tfidf_sort_articles(docs, query_preprocessed)
tfidf_top_10 = top_n_articles(tfidf_sorted, 10)

for tuple in tfidf_top_10:
  print(tuple[0], '\t', tuple[1], '\t', tuple[2])

print('benötigte Zeit (in Sekunden):', time.time() - start_time)