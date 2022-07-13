from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from functions import *
import json

# Gesetze und Pfad
laws = ['or', 'zgb']
law_path = '/home/oliver/Nextcloud/GymInf-Projekt/Gesetze/'

# Dictionaries bereitstellen
articles_formatted = {}
articles_preprocessed = {}

for law in laws:
    # HTML-Datei einlesen (nur article-Tags relevant)
    article_tags = SoupStrainer('article')
    with open(law_path + law + '.html') as file:
        articles = BeautifulSoup(file, 'html.parser', parse_only=article_tags)
   # Artikel in Dictionary speichern
    last_art_number = 0
    for art in articles:
        # ID und Name des Artikels bestimmen
        art_id = art['id']
        a_tag = art.find('a')
        art_name = a_tag['name']
        # Relevante Artikel bestimmen (z.B. keine Übergangsbestimmungen)
        art_number = ''
        for m in art_name:
            if m.isdigit():
                art_number = art_number + m
        art_number = int(art_number)
        if art_number < last_art_number:
            break
        last_art_number = art_number
        # Arikel als formatierten String auslesen
        art_formatted = get_formatted_article(art.div)
        # Ausgelesener String nicht leer
        if art_formatted != '':
            # Artikel als linguistisch vorverarbeiteten String auslesen
            art_text = get_article_text(art.div)
            art_preprocessed = get_preprocessed_text(art_text)
            # Artikel in Dictionaries einfügen
            articles_formatted[law + '_' + art_id] = art_formatted
            articles_preprocessed[law + '_' + art_id] = art_preprocessed

# Dictionaries als JSON speichern
f = open('articles_formatted.json', 'w')
json.dump(articles_formatted, f)
f.close()
f = open('articles_preprocessed.json', 'w')
json.dump(articles_preprocessed, f)
f.close()

# Beispiel Art. 40a OR
with open(law_path + 'or_art_40_a.txt', 'w') as f:
    f.write(articles_formatted['or_art_40_a'])
    f.write('\n\n')
    f.write(articles_preprocessed['or_art_40_a'])