from functions import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import json

query = 'Widerrufsrecht bei Haustürgeschäften'

with open('articles_preprocessed.json') as json_file:
    docs = json.load(json_file)

# TF-IDF-Matrix Dokumente
vectorizer = TfidfVectorizer(sublinear_tf=True)
tfidf_docs = vectorizer.fit_transform(docs.values())
vocab = vectorizer.vocabulary_
# svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
# svd.fit(tfidf_docs)
# tfidf_docs_svd = svd.transform(tfidf_docs)

# TF-IDF-Vektor Abfrage
preprocessed_query = get_preprocessed_text(query)
dict_query = {'query': preprocessed_query}
vectorizer = TfidfVectorizer(vocabulary=vocab, sublinear_tf=True)
tfidf_query = vectorizer.fit_transform(dict_query.values())
# tfidf_query_svd = svd.transform(tfidf_query)

cos_sim = cosine_similarity(tfidf_query, tfidf_docs)
print('TF-IDF: Kosinus-Ähnlichkeit zwischen Abfrage und Dokumenten')
print(cos_sim)

# cos_sim_pca = cosine_similarity(tfidf_query_svd, tfidf_docs_svd)
# print('TF-IDF: Kosinus-Ähnlichkeit zwischen Abfrage und Dokumenten (SVD)')
# print(cos_sim_pca)