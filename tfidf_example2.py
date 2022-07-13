from functions import *

d1 = 'Mann'
d2 = 'Frau'

docs = {
  'd1': 'Mann',
  'd2': 'Frau'
  }

vectorizer = TfidfVectorizer(sublinear_tf=True)
tfidf_docs = vectorizer.fit_transform(docs.values())
cossim = cosine_similarity(tfidf_docs, tfidf_docs)
print(cossim)