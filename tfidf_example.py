import pandas as pd
from sklearn.decomposition import PCA
from functions import *
from sklearn.feature_extraction.text import TfidfVectorizer

docs = {
  'd1': 'information is the new gold',
  'd2': 'everything is information and information is everything'
}

query = 'what is information retrieval'

# TF-IDF-Matrix Dokumente
index_docs = [n for n in docs]
vectorizer = TfidfVectorizer(sublinear_tf=True)
tfidf_docs = vectorizer.fit_transform(docs.values())
vocab = vectorizer.vocabulary_
feature_names_docs = vectorizer.get_feature_names_out()

# TF-IDF-Vektor Abfrage
preprocessed_query = get_preprocessed_text(query)
dict_query = {'query': preprocessed_query}
vectorizer = TfidfVectorizer(vocabulary=vocab, sublinear_tf=True)
tfidf_query = vectorizer.fit_transform(dict_query.values())
feature_names_query = vectorizer.get_feature_names_out()

# TF-IDF-Matrix Dokumente ausgeben
df_docs = pd.DataFrame(tfidf_docs.T.todense(), index=feature_names_docs, columns=index_docs)
print(df_docs)

# TF-IDF-Vektor Abfrage ausgeben
df_query = pd.DataFrame(tfidf_query.T.todense(), index=feature_names_query, columns=['query'])
print(df_query)

cos_sim = cosine_similarity(tfidf_query, tfidf_docs)
print('Kosinus-Ähnlichkeit zwischen Abfrage und Dokumenten:')
print(cos_sim)