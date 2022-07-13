import json
from functions import *
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/' , methods=['GET', 'POST'])
def home():
    # GET request
    if request.method == 'GET':
        return render_template('index.html')
    # POST request
    if request.method == 'POST':
        query_text = request.form['queryText']
        display_method = request.form['mySelect']
        with open('articles_preprocessed.json') as json_file:
            docs = json.load(json_file)
        if len(query_text) > 0:
            preprocessed_query = get_preprocessed_text(query_text)
            tfidf_sorted = tfidf_sort_articles(docs, preprocessed_query)
            ft_sorted = fasttext_sort_articles(docs, preprocessed_query)
            if display_method == 'top10':
                tfidf_art = top_n_articles(tfidf_sorted, 10)
                ft_art = top_n_articles(ft_sorted, 10)
            if display_method == 'predictedPositive':
                tfidf_art = articles_over_threshold(tfidf_sorted, 0.3)
                ft_art = articles_over_threshold(ft_sorted, 0.65)
            return render_template('index.html',
                query=query_text,
                display=display_method,
                tfidf=tfidf_art,
                fasttext=ft_art)
        else:
            return render_template('index.html')

app.run(debug=True)