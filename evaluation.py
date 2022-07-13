from functions import *
from statistics import mean
import json
import sys
import pandas as pd

with open('articles_preprocessed.json') as json_file:
    docs = json.load(json_file)

N = len(docs)

test_queries = {
  	'Widerrufsrecht bei Haustürgeschäften': ['or_art_40a', 'or_art_40b', 'or_art_40_c', 'or_art_40_d', 'or_art_40_e', 'or_art_40_f'],
	'Erfüllungsort bei Geldschulden': ['or_art_74'],
	'Transportkosten beim Kaufvertrag': ['or_art_189'],
	'Wohnung untervermieten': ['or_art_262'],
	'Fristlose Kündigung des Arbeitsvertrags': ['or_art_337', 'or_art_337_a', 'or_art_337_b', 'or_art_337_c'],
	'Konkurrenzverbot im Arbeitsvertrag': ['or_art_340', 'or_art_340_a', 'or_art_340_b', 'or_art_340_c'],
	'Mindestkapital AG': ['or_art_621'],
	'Auflösung einer Verlobung': ['zgb_art_91', 'zgb_art_92', 'zgb_art_93'],
	'Scheidung einer Ehe': ['zgb_art_111', 'zgb_art_112', 'zgb_art_114', 'zgb_art_115'],
	'Altersunterschied bei Adoption': ['zgb_art_264_d'],
	'Enterbung eines Kindes': ['zgb_art_477', 'zgb_art_478', 'zgb_art_479', 'zgb_art_480'],

	# Abfragen mit schlechten Resultaten
	# 'Formfreiheit bei Verträgen': ['or_art_11'],
	# 'Haftung des Hauseigentümers': ['or_art_58', 'or_art_59'],
	# 'Verjährungsfrist für Zahnarztrechnung': ['or_art_128'],
	# 'Garantieausschluss beim Kauf': ['or_art_199'],
	# 'Formvorschrift bei Hauskauf': ['or_art_216'],
	# 'Höhe der Mietkaution': ['or_art_257_e'],
	# 'Kündigungsfrist bei Wohnungen': ['or_art_266_c'],
	# 'Kündigung des Arbeitsvertrags während Militärdienst': ['or_art_336_c'],
	# 'Voraussetzungen für Ehe': ['zgb_art_94', 'zgb_art_95', 'zgb_art_96'],
	# 'Name nach Hochzeit': ['zgb_art_160'],
	# 'Gemeinschaftliche Adoption': ['zgb_art_254_a'],
	# 'Pflichtteile im Erbrecht': ['zgb_art_471']
}

method = sys.argv[1] # 'tfidf' oder 'fasttext'

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

average_accuracy = {}
average_recall = {}
average_precision = {}
average_f1_score = {}

for threshold in thresholds:
	accuracies = []
	recalls = []
	precisions = []
	f1_scores = []
	for query in test_queries.keys():
		print('Method:', method)
		print('Threshold:', threshold)
		print('Query:', query)
		preprocessed_query = get_preprocessed_text(query)
		# Predicted Positives
		predicted_positives_tuples = articles_over_threshold(docs, preprocessed_query, threshold, method)
		predicted_positives = len(predicted_positives_tuples)
		predicted_positives_articles = []
		for tuple in predicted_positives_tuples:
			predicted_positives_articles.append(tuple[0])
		# print('predicted positives:', predicted_positives)
		# Predicted Negatives
		predicted_negatives = N - predicted_positives
		# print('predicted negatives:', predicted_negatives)
		# Actual Positives
		actual_positives_articles = test_queries[query]
		actual_positives = len(actual_positives_articles)
		# print('actual positives:', actual_positives)
		# Actual Negatives
		actual_negatives = N - actual_positives
		# print('actual negatives:', actual_negatives)
		# True Positives
		true_positives = 0
		for actual_positive in actual_positives_articles:
			if actual_positive in predicted_positives_articles:
				true_positives += 1
		# print('true positives:', true_positives)
		# False Positive
		false_positives = predicted_positives - true_positives
		# print('false positives:', false_positives)
		# True Negative
		true_negatives = actual_negatives - false_positives
		# print('true negatives:', true_negatives)
		# False Negatives
		false_negatives = actual_positives - true_positives
		# print('false negatives:', false_negatives)
		# Accuracy, Recall, Precision, F1 Score
		accuracy = (true_positives + true_negatives) / N
		if true_positives == 0:
			recall = 0
			precision = 0
		else:
			recall = true_positives / (true_positives + false_negatives)
			precision = true_positives / (true_positives + false_positives)
		if recall == 0 and precision == 0:
			f1_score = 0
		else:
			f1_score = 2 * (precision * recall) / (precision + recall)
		accuracies.append(accuracy)
		recalls.append(recall)
		precisions.append(precision)
		f1_scores.append(f1_score)
		print('accuracy:', accuracy)
		print('recall:', recall)
		print('precision:', precision)
		print('f1 score:', f1_score)
		print('----------------------------------------------------')
	# Average Accuracy, Recall, Precision, F1 Score
	average_accuracy[threshold] = mean(accuracies)
	average_recall[threshold] = mean(recalls)
	average_precision[threshold] = mean(precisions)
	average_f1_score[threshold] = mean(f1_scores)

print('Average Accuracy:', average_accuracy)
print('Average Recall:', average_recall)
print('Average Precision:', average_precision)
print('Average F1 Score:', average_f1_score)