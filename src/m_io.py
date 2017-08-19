# m_io.py
#=====================================================================
import sys
import string
import nltk			# natural language toolkit
import urllib			# to read websites
from bs4 import BeautifulSoup	# beautiful soup for cleaning websites


#=====================================================================
def get_text_and_write_corpus(category,corpus_dir,multiple_pages=False):
	'''
	Generate a corpus of text for a given category of wikipedia webpages.
	Starting from the summary category webpage:
		1) get the list of urls on the category webpage
		2) get the text on each of the webpages
		3) write the accumulated text to file
	'''
#---------------------------------------------------------------------
	url_list = get_sub_links_in_category(category,multiple_pages=multiple_pages)
	text = gather_text_in_category(url_list)
	write_text_to_file(text,corpus_dir+"/"+category+".txt")
	return 


#=====================================================================
def get_and_write_random_corpus(corpus_dir):
	'''
	Generate a corpus of text for a random selection of wikipedia webpages.
	A random set of sites have been selected.
	The same random selection is used to ensure repeatability.
	'''
#---------------------------------------------------------------------
	url_list = ['https://en.wikipedia.org/wiki/Minute-Man',\
		'https://en.wikipedia.org/wiki/Kazakhstan_at_the_2014_UCI_Road_World_Championships',\
		'https://en.wikipedia.org/wiki/Lincoln_Chronicle',\
		'https://en.wikipedia.org/wiki/Diwan-khane',\
		'https://en.wikipedia.org/wiki/HighChem',\
		'https://en.wikipedia.org/wiki/Jewish_Community_Watch',\
		'https://en.wikipedia.org/wiki/Jeane_Lassen',\
		'https://en.wikipedia.org/wiki/Tim_Fairbrother',\
		'https://en.wikipedia.org/wiki/Prink_Callison',\
		'https://en.wikipedia.org/wiki/Viterbo_Papacy',\
		'https://en.wikipedia.org/wiki/Heinrich_Donatus,_Hereditary_Prince_of_Schaumburg-Lippe',\
		'https://en.wikipedia.org/wiki/Akuliaruseq_Island',\
		'https://en.wikipedia.org/wiki/Keokuk_Rail_Bridge',\
		'https://en.wikipedia.org/wiki/Ashok_Rudra',\
		'https://en.wikipedia.org/wiki/List_of_people_on_the_postage_stamps_of_Russia',\
		]
	text = gather_text_in_category(url_list)
	write_text_to_file(text,corpus_dir+"/Random.txt")
	return 


#=====================================================================
def get_sub_links_in_category(category,multiple_pages=False):
	'''
	Determine the number of links in each category.
	For category pages over multiple pages, loop through each of the 
	category pages starting with each of the letters of the alphabet.
	'''
#---------------------------------------------------------------------
	links = list()
	if (multiple_pages):
		print('')
		alphabet = list(string.ascii_uppercase)
		url_prefix = 'https://en.wikipedia.org/w/index.php?title=Category:'+category+'&from='
		for letter in alphabet:
			print('Processing web pages in category {0} starting with the letter {1} ...'.format(category,letter))
			url = url_prefix+letter
			get_sub_links_in_webpage(category,url,links)
	else:
		print('\nProcessing web pages in category {0} ...'.format(category))
		url = 'https://en.wikipedia.org/wiki/Category:'+category
		get_sub_links_in_webpage(category,url,links)
	return sorted(list(set(links)))		# use 'set' to remove repeated entries


#=====================================================================
def get_sub_links_in_webpage(category,url,links):
	'''
	For a given category webpage, extract the urls from the text 
	that are not the standard wikipedia links.
	'''
#---------------------------------------------------------------------
	html = urllib.urlopen(url).read()		# open website
	soup = BeautifulSoup(html)			# send raw html text to beautifulSoup for processing
	for link in soup.find_all('a'):
		url_string = link.get('href')
		if (url_string != None):
			if ('Category:' not in url_string) \
				and ('Help:' not in url_string)  and ('Main_Page' not in url_string) and ('Portal:' not in url_string) \
				and ('Special:' not in url_string) and ('Contact_us:' not in url_string) and ('wikidata' not in url_string) \
				and ('shop.wikimedia.org' not in url_string) and ('Wikipedia:' not in url_string) \
				and ('wikipedia.org' not in url_string) and ('creativecommons' not in url_string) \
				and ('wikimediafoundation' not in url_string) and ('mediawiki' not in url_string) \
				and ('wiktionary' not in url_string) \
				and ('%' not in url_string) and ('#' not in url_string) and (category not in url_string):
				links.append('https://en.wikipedia.org'+url_string)
	return


#=====================================================================
def gather_text_in_category(url_list):
	'''
	Get text from all of the urls in a given category.
	'''
#---------------------------------------------------------------------
	all_tokens = list()
	for i,url in enumerate(url_list):
		print('Gathering text from {0}, url number {1} of {2}'.format(url,i+1,len(url_list)))
		valid_tokens = gather_tokens_from_website(url)
		all_tokens.append(valid_tokens)
		print('   number of tokens in this text {0}'.format(len(valid_tokens)))
	return all_tokens


#=====================================================================
def gather_tokens_from_website(url):
	'''
	For a given website extract only the alphabetic text.
	Convert all text to lower case and lemmatize.
	'''
#---------------------------------------------------------------------
	html = urllib.urlopen(url).read()		# open website
	soup = BeautifulSoup(html)			# send raw html text to beautifulSoup for processing
	raw = soup.get_text()				# get website text
	tokens = nltk.word_tokenize(raw)		# split into separate words
	wnl = nltk.WordNetLemmatizer()			
	valid_tokens = [ wnl.lemmatize(w.lower()) for w in tokens if w.isalpha() ]
	return valid_tokens


#=====================================================================
def is_ascii(s):
	'''
	Check if all characters in a given string are ASCII.
	'''
#---------------------------------------------------------------------
	return all(ord(c) < 128 for c in s)


#=====================================================================
def write_text_to_file(text,filename):
	'''
	For each page write all of the words on the same line separated by commas.
	Each new page is written on a new line.
	'''
#---------------------------------------------------------------------
	output_file = open(filename, 'w')
	for page in text:
		for word in page:
			if is_ascii(word):
				output_file.write(word + ", ")
		output_file.write("\n")
	output_file.close()
	return


#=====================================================================
def read_vocabulary(categories,min_word_length=4):
	'''
	Read the corpus of a given category  and return a tuple containing a list of words for a given page, and an assigned category.
	Each webpage within the corpus is written on a new line, with the words on a given line separated by commas.
	'''
#---------------------------------------------------------------------
        all_vocab = list()
        for category in categories:
                filename = './corpus/' + category + '.txt'
                print('   Reading file {0}'.format(filename))
                document = read_text_from_file(filename)
                for page in document:
                        words = [ word.rstrip() for word in string.split(page) if len(word.rstrip())>=min_word_length ]
                        vocab = (words,category)
                        all_vocab.append(vocab)
        return all_vocab


#=====================================================================
def read_text_from_file(filename):
	'''
	Read text from file.
	Each line of the text file contains the words on a given web page.
	'''
#---------------------------------------------------------------------
        input_file = open(filename, 'r')
        text = list()
        text = input_file.readlines()
        input_file.close()
        return text


#=====================================================================
#=====================================================================
