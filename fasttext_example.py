import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity

# Warnung unterdrücken
fasttext.FastText.eprint = lambda x: None

# Vortrainiertes Modell laden
ft = fasttext.load_model('fastText/cc.de.300.bin')
print('Dimension des Modells:', ft.get_dimension())

# Wortvektoren und Nearest Neighbors
words = ['Hallo', 'Banane']
for word in words:
  print('Nearest Neighbors:', word)
  ft.get_word_vector(word)
  print(ft.get_nearest_neighbors(word))

# Wortanalogien
print('Wortanalogie:', ft.get_analogies('Berlin', 'Deutschland', 'Frankreich'))

# Beispiel TF-IDF mit fastText
docs = {
  'd1': 'information is the new gold',
  'd2': 'everything is information and information is everything'
}

query = 'what is information retrieval'

# fastText-Vektoren Dokumente
fasttext_docs = []
for doc in docs.values():
  vect_doc = ft.get_sentence_vector(doc)
  fasttext_docs.append(vect_doc)

# fastText-Vektor Abfrage
fasttext_query = ft.get_sentence_vector(query)
fasttext_query = fasttext_query.reshape(1, -1)

# Kosinus-Ähnlichkeit zwischen Abfrage und Dokumenten
cossim = cosine_similarity(fasttext_query, fasttext_docs)
print('cosine similarity query-docs:', cossim)