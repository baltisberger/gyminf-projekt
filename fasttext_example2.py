import fasttext
import fasttext.util
from functions import *
from sklearn.metrics.pairwise import cosine_similarity

fasttext.FastText.eprint = lambda x: None

ft = fasttext.load_model('fastText/cc.de.300.bin')

d1 = 'Mann'
d2 = 'Frau'

docs = {
  'd1': 'Mann',
  'd2': 'Frau'
  }

fasttext_docs = []
for doc in docs.values():
  vect_doc = ft.get_word_vector(doc)
  fasttext_docs.append(vect_doc)

cossim = cosine_similarity(fasttext_docs, fasttext_docs)
print(cossim)