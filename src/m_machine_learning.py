# m_machine_learning.py
#=====================================================================
'''
Module to train machine learning classification of web pages from labelled corpus.
'''

#=====================================================================
import string
import numpy as np
import nltk
import matplotlib
import matplotlib.pyplot as plt


#=====================================================================
def get_most_frequent_words(text,max_number_of_words=1000):
	'''
	Get the specified number of most frequent words from the text.
	'''
#---------------------------------------------------------------------
	word_freq = nltk.FreqDist(text)
	word_freq_sort = sorted(zip(word_freq.values(),word_freq.keys()),reverse=True)	# sort in order of increasing frequency
	if (len(word_freq_sort)>max_number_of_words):
		words = set(zip(*word_freq_sort)[1][0:max_number_of_words])		# assign most frequent words to classify the document
	else:
		words = set(word_freq.keys())
	return words


#=====================================================================
def document_features(page,word_features):
	'''
	Calculate boolean dict stating if a feature word is in the selected page.
	'''
#---------------------------------------------------------------------
	page_words = set(page)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in page_words)
	return features


#=====================================================================
def determine_optimal_number_of_features(documents,all_words,min_number_of_words = 10,num_feature_divisions = 10):
	'''
	Build a series of classification models using an increasing number of total features (increasingly complex).
	'''
#---------------------------------------------------------------------
	print('\nDetermining sensitivity of results to number of word features ...')
	max_number_of_words = min(len(all_words),2000)
	number_of_word_features = np.logspace(np.log10(min_number_of_words),np.log10(max_number_of_words),num_feature_divisions)
	train_accuracy = np.array(np.zeros((num_feature_divisions), dtype=np.float)) 
	validation_accuracy = np.array(np.zeros((num_feature_divisions), dtype=np.float)) 
	for i,nwf in enumerate(number_of_word_features):
		print('   Building classification model using {0} of a maximum of {1} word features.'.format(int(nwf),max_number_of_words))
		word_features = get_most_frequent_words(all_words,max_number_of_words=int(nwf))
		features = [(document_features(page[0],word_features), page[1]) for page in documents]

		print('      Spliting data into training, validation and test sets')
		train_set, validation_set, test_set = split_samples_into_sets(features)

		print('      Building Naive Bayes classifier...')
		classifier = nltk.NaiveBayesClassifier.train(train_set)
		train_accuracy[i] = nltk.classify.accuracy(classifier, train_set)
		validation_accuracy[i] = nltk.classify.accuracy(classifier, validation_set)
		print('         training set accuracy = {0}'.format(train_accuracy[i]))
		print('         validation set accuracy = {0}'.format(validation_accuracy[i]))

	optimal_number_of_features = int(number_of_word_features[np.argmax(validation_accuracy[:])])
	print('   Optimal number of features is {0}'.format(optimal_number_of_features))
	print('      test set accuracy = {0}'.format(nltk.classify.accuracy(classifier, test_set)))

	plot_accuracy(number_of_word_features,train_accuracy,validation_accuracy,\
		'number of features','./results/num_feature_sensitivity.png',log_plot=True)
	write_num_feature_sensitivity_accuracy(number_of_word_features,train_accuracy,validation_accuracy,\
		'./results/num_feature_sensitivity.dat')
	
	return optimal_number_of_features


#=====================================================================
def plot_accuracy(x,train,validation,xlabel,filename,log_plot=False):
	'''
	Plot train and validation accuracy against a general independent variable 'x'.
	'''
#---------------------------------------------------------------------
        matplotlib.rc('font', size=12, family='sans-serif')
        matplotlib.rcParams['xtick.major.pad']=12
        matplotlib.rcParams['ytick.major.pad']=12
	fig = plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel('accuracy')
	ax = fig.add_subplot(111)
	if(log_plot):
		ax.set_xscale('log')
        plt.plot(x, train, 'b-', label='train')
        plt.plot(x, validation, 'r--', label='validation')
	plt.legend()
        plt.subplots_adjust(left=0.22, bottom=0.2, right=0.95, top=0.95)
        plt.savefig(filename)
        plt.close()
	return


#=====================================================================
def write_num_feature_sensitivity_accuracy(number_of_word_features,train_accuracy,validation_accuracy,filename):
	'''
	Write to file the accuracy of the training and validation data set for a range of number of features.
	'''
#---------------------------------------------------------------------
	file = open(filename,'w')
	file.write('#\n')
	file.write('# number of word features, training set error, validation set error\n')
	file.write('#\n')
	for i,nwf in enumerate(number_of_word_features):
		file.write('%d %18.10e %18.10e\n' % (int(nwf), train_accuracy[i], validation_accuracy[i]))
	file.close()
        return


#=====================================================================
def calculate_learning_curves(documents,all_words,optimal_number_of_features,min_proportion = 0.5,num_learning_curve_divisions = 10):
	'''
	Build a series of classification models using an increasing proportion of total samples.
	'''
#---------------------------------------------------------------------
	print('\nCalculating learning curves for optimal model ...')
	sample_proportions = np.linspace(min_proportion,1.0,num_learning_curve_divisions)
	train_accuracy = np.array(np.zeros((num_learning_curve_divisions), dtype=np.float)) 
	validation_accuracy = np.array(np.zeros((num_learning_curve_divisions), dtype=np.float)) 

	word_features = get_most_frequent_words(all_words,max_number_of_words=optimal_number_of_features)
	features = [(document_features(page[0],word_features), page[1]) for page in documents]
	train_set, validation_set, test_set = split_samples_into_sets(features)
	num_train_samples = len(train_set)

	for i,p in enumerate(sample_proportions):
		print('   Building classification model using a proportion of {0} of the total samples.'.format(p))
		train_set	= features[:int(num_train_samples*p)]

		print('      Building Naive Bayes classifier...')
		classifier = nltk.NaiveBayesClassifier.train(train_set)
		train_accuracy[i] = nltk.classify.accuracy(classifier, train_set)
		validation_accuracy[i] = nltk.classify.accuracy(classifier, validation_set)
		print('         training set accuracy = {0}'.format(train_accuracy[i]))
		print('         validation set accuracy = {0}'.format(validation_accuracy[i]))

	print('\nTen most informative features for model with optimal features using all of the training set data ...')
	classifier.show_most_informative_features(10)

	plot_accuracy(sample_proportions,train_accuracy,validation_accuracy,'proportion of total samples','./results/learning_curves.png')
	write_learning_curve_accuracy(sample_proportions,train_accuracy,validation_accuracy,'./results/learning_curves.dat')
	
	return classifier, word_features


#=====================================================================
def write_learning_curve_accuracy(proportion_of_samples,train_accuracy,validation_accuracy,filename):
	'''
	Write to file the accuracy of the training and validation data set for a range of proportions of total samples.
	'''
#---------------------------------------------------------------------
	file = open(filename,'w')
	file.write('#\n')
	file.write('# proportion of total samples, training set error, validation set error\n')
	file.write('#\n')
	for i,p in enumerate(proportion_of_samples):
		file.write('%18.10e %18.10e %18.10e\n' % (p, train_accuracy[i], validation_accuracy[i]))
	file.close()
        return


#=====================================================================
def build_classification_model(documents,all_words,num_features):
	'''
	Build a Naive Bayes classification model using the number of features specified.
	'''
#---------------------------------------------------------------------
	print('   Building classification model using {0} of a maximum of {1} word features.'.format(num_features,len(all_words)))
	word_features = get_most_frequent_words(all_words,max_number_of_words=num_features)
	features = [(document_features(page[0],word_features), page[1]) for page in documents]

	print('   Spliting data into training, validation and test sets')
	train_set, validation_set, test_set = split_samples_into_sets(features)

	print('   Building Naive Bayes classifier...')
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print('      training set accuracy = {0}'.format(nltk.classify.accuracy(classifier, train_set)))
	print('      validation set accuracy = {0}'.format(nltk.classify.accuracy(classifier, validation_set)))
	print('      test set accuracy = {0}'.format(nltk.classify.accuracy(classifier, test_set)))
	return classifier, word_features


#=====================================================================
def split_samples_into_sets(features):
	'''
	Split data into the training, cross validation and test sets.
	'''
#---------------------------------------------------------------------
	num_samples = len(features)
	num_train_samples = int(0.6*num_samples)
	num_validation_samples = int(0.2*num_samples)
	num_test_samples = num_samples - num_train_samples - num_validation_samples
	train_set	= features[:num_train_samples]
	validation_set	= features[num_train_samples:num_train_samples+num_validation_samples]
	test_set	= features[-num_test_samples:]
	return train_set, validation_set, test_set


#=====================================================================
#=====================================================================
