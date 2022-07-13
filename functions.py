import re
import string
import numpy as np
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# fastText-Warnung unterdrücken
fasttext.FastText.eprint = lambda x: None

def get_formatted_article(tag):
    # Alle Links bzw. Fussnoten eliminieren
    for a in tag.find_all('a'):
        a.decompose()
    text = ''
    for child in tag.children:
        # Absätze (z.B. OR 6)
        if child.name == 'p':
            text = text + get_replaced_text(child) + '\n'
        # Aufzählungen (z.B. OR 40a oder OR 271a)
        if child.name == 'dl':
            for c in child.children:
                if c.name == 'dt':
                    text = text + get_replaced_text(c)
                if c.name == 'dd':
                    dl = c.find('dl')
                    if dl == None:
                        text = text + get_replaced_text(c) + '\n'
                    else:
                        text_in_dl = dl.extract()
                        text = text + get_replaced_text(c) + '\n'
                        for d in text_in_dl.children:
                            if d.name == 'dt':
                                text = text + get_replaced_text(c)
                            if d.name == 'dd':
                                text = text + get_replaced_text(c) + '\n'
        # Tabellen (z.B. OR 361)
        if child.name == 'div':
            rows = child.find_all('tr')
            for row in rows:
                td = row.find_all('td')
                for t in td:
                    text = text + get_replaced_text(t) + ' '
                text = text + '\n'
    # Letzten Zeilenumbruch entfernen
    text = text[:-1]
    return text

# Hilfsfunktion für get_formatted_text()
def get_replaced_text(tag):
    text = tag.get_text()
    # Zeilenumbrüche
    text = text.replace('\n', '')
    # Soft-Hyphens
    text = text.replace('\xad', '')
    # Non-Breaking Spaces
    text = text.replace('\xa0', ' ')
    return text

def get_article_text(tag):
    # sup-Tags entfernen (z.B. Absatzbeschriftung)
    for sup in tag.find_all('sup'):
        sup.decompose()
    # dt-Tags entfernen (z.B. Aufzählungszeichen)
    for dt in tag.find_all('dt'):
        dt.decompose()
    # Fussnoten entfernen
    footnotes = tag.find('div', {'class': 'footnotes'})
    if footnotes != None:
        footnotes.decompose()
    # Leerzeichen vor/nach dd- und td-Tags einfügen
    dd_tags = tag.find_all('dd')
    for dd in dd_tags:
        dd.insert_before(' ')
        dd.insert_after(' ')
    td_tags = tag.find_all('td')
    for td in td_tags:
        td.insert_before(' ')
        td.insert_after(' ')
    return tag.get_text()

def get_preprocessed_text(text):
    # Leerzeichen/Tab am Anfang entfernen
    if re.match(r'\s', text):
        text = text[1:]
    # In Lowercase umwandeln
    text = text.lower()
    # Sonderzeichen entfernen
    text = text.replace('\n', '')       # Zeilenumbrüche
    text = text.replace('\xa0', ' ')    # Non-Breaking Spaces
    text = text.replace('\u00ad', '')   # Soft-Hyphens
    text = text.replace('\u00ab', '')   # Left-Point Double Angle Quotation Mark
    text = text.replace('\u00bb', '')   # Right-Point Double Angle Quotation Mark
    text = text.replace('\u2013', ' ')  # En Dash
    text = text.replace('\u2011', '')   # Non-Breaking Hyphen
    # Stoppwörter entfernen
    words = word_tokenize(text)
    german_stop_words = stopwords.words('german')
    filtered_text = [w for w in words if not w in german_stop_words]
    text = " ".join(filtered_text)
    # Daten verbinden ('2. april 1908' -> '2april1908')
    text = re.sub(r'(\d{1,2})\.\s+(\w{3,9})\s+(\d{4})', r'\1\2\3', text)
    # Satzzeichen entfernen
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('…', ' ')
    # Umlaute ersetzen
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    # Mehrfache Leerzeichen  entfernen
    text = re.sub(r'(\s)+', ' ', text)
    return text
    
def doc_cossim_url_tuples(docs_index, docs_vect, query_vect):
    cos_sim = cosine_similarity(query_vect, docs_vect)
    tuples_list = []
    i = 0
    for doc in docs_index:
        art_id = doc
        if art_id.startswith('or'):
            art = art_id[3:]
            url = 'https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#' + art
        if art_id.startswith('zgb'):
            art = art_id[4:]
            url = 'https://www.fedlex.admin.ch/eli/cc/24/233_245_233/de#' + art
        tuple = (art_id, cos_sim[0][i], url)
        tuples_list.append(tuple)
        i += 1
    return sorted(tuples_list, key=lambda x:x[1], reverse=True)

def tfidf_sort_articles(docs, query):
    index_docs = [n for n in docs]
    # TF-IDF-Vektoren Dokumente
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(5,5), sublinear_tf=True)
    tfidf_docs = vectorizer.fit_transform(docs.values())
    vocab = vectorizer.vocabulary_
    # svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    # svd.fit(tfidf_docs)
    # tfidf_docs_svd = svd.transform(tfidf_docs)
    # TF-IDF-Vektor Abfrage
    dict_query = {'query': query}
    vectorizer = TfidfVectorizer(analyzer='char_wb', vocabulary=vocab, ngram_range=(5,5), sublinear_tf=True)
    tfidf_query = vectorizer.fit_transform(dict_query.values())
    # tfidf_query_svd = svd.transform(tfidf_query)
    # Kosinus-Ähnlichkeit zwischen Abfrage und Dokumenten
    return doc_cossim_url_tuples(index_docs, tfidf_docs, tfidf_query)

def fasttext_sort_articles(docs, query):
    index_docs = [n for n in docs]
    ft = fasttext.load_model('fastText/cc.de.300.bin')
    # fastText-Vektoren Dokumente
    fasttext_docs = []
    for doc in docs.values():
        vect_doc = ft.get_sentence_vector(doc)
        fasttext_docs.append(vect_doc)
    # svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    # svd.fit(fasttext_docs)
    # fasttext_docs_svd = svd.transform(fasttext_docs)
    # fastText-Vektoren Abfrage
    fasttext_query = ft.get_sentence_vector(query)
    fasttext_query = fasttext_query.reshape(1, -1)
    # fasttext_query_svd = svd.transform(fasttext_query)
    # Kosinus-Ähnlichkeit zwischen Abfrage und Dokumenten
    return doc_cossim_url_tuples(index_docs, fasttext_docs, fasttext_query)

def top_n_articles(list, n):
    return list[0:n]

def articles_over_threshold(list, treshold):
    n = 0
    for tuple in list:
        if tuple[1] > treshold:
            n += 1
        else:
            break
    return list[0:n]