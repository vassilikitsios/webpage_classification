# build_corpus.py
#=====================================================================
'''
Program to strip text from Wikipedia websites in various categories to 
build Corpus for future document classification machine learning studies.
'''
#=====================================================================
import sys
import os
import m_io as io
#=====================================================================
if __name__ == '__main__':

	print('\n===================================================')
	print('Running {0}'.format(sys.argv[0]))
	print('===================================================')

	corpus_dir='./corpus'
	if not os.path.exists(corpus_dir):
		os.mkdir(corpus_dir)

	io.get_text_and_write_corpus('Cancer',corpus_dir)
	io.get_text_and_write_corpus('Infectious_diseases',corpus_dir)
	io.get_text_and_write_corpus('Congenital_disorders',corpus_dir)
	io.get_text_and_write_corpus('Organs',corpus_dir)
	io.get_text_and_write_corpus('Medical_devices',corpus_dir)
	io.get_text_and_write_corpus('Machine_learning_algorithms',corpus_dir)
	io.get_text_and_write_corpus('Rare_diseases',corpus_dir,multiple_pages=True)
	io.get_and_write_random_corpus(corpus_dir)

	print ('\n===================================================')
	print ('Code complete.')
	print ('===================================================\n')
#=====================================================================
#=====================================================================
