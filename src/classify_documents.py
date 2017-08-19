# classify_documents.py
#=====================================================================
'''
Program to read in Corpus from file and build document classification
machine learning algorythms using Naive Bayes.
'''
#=====================================================================
import sys
import os
import string
import numpy as np
import random
import nltk
import m_io as io
import m_machine_learning as ml
#=====================================================================
if __name__ == '__main__':

	print('\n===================================================')
	print('Running {0}'.format(sys.argv[0]))
	print('===================================================\n')

	#----------------------------------------------------------------
	print('Processing input arguments ...')
	if (len(sys.argv[:])!=4):
		print('\nFAILURE')
		print('Usage: {0} <sensitivity_analysis (true/false)> <num_features (int)> <test_url (string)>\n'.format(sys.argv[0]))
		sys.exit()
	else:
		perform_sensitivity_analysis=False
		if(sys.argv[1]=='true'):
			perform_sensitivity_analysis=True
        	num_features = int(sys.argv[2])
		test_url = sys.argv[3]

	#----------------------------------------------------------------
	print('\nReading and shuffling Corpus data ...')
	categories = ['Rare_diseases', 'Random', 'Infectious_diseases', 'Cancer', 'Congenital_disorders',\
			'Organs', 'Medical_devices', 'Machine_learning_algorithms']
	documents = io.read_vocabulary(categories, min_word_length=3)
	random.shuffle(documents)
	all_words = [word.lower() for page in documents for word in page[0]]

	#----------------------------------------------------------------
	if(perform_sensitivity_analysis):
		results_dir='./results'
		if not os.path.exists(results_dir):
			os.mkdir(results_dir)
		optimal_number_of_features = ml.determine_optimal_number_of_features(documents,all_words)
		classifier, word_features = ml.calculate_learning_curves(documents,all_words,optimal_number_of_features)
	else:
		print('')
		classifier, word_features = ml.build_classification_model(documents,all_words,num_features)

	#----------------------------------------------------------------
	print('\nClassifying test url {0}'.format(test_url))
	test_page = io.gather_tokens_from_website(test_url)
	test_features = ml.document_features(test_page,word_features)
	print('   estimated category = {0}'.format(classifier.classify(test_features)))

	print ('\n===================================================')
	print ('Code complete.')
	print ('===================================================\n')
#=====================================================================
#=====================================================================
