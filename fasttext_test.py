import time
import json
from functions import *

start_time = time.time()

with open('articles_preprocessed.json') as json_file:
    docs = json.load(json_file)

query = 'Ferienanspruch Arbeitnehmer pro Jahr'
query_preprocessed = get_preprocessed_text(query)

ft_sorted = fasttext_sort_articles(docs, query_preprocessed)
ft_top_10 = top_n_articles(ft_sorted, 10)

for tuple in ft_top_10:
  print(tuple[0], '\t', tuple[1], '\t', tuple[2])

print('total time:', time.time() - start_time)