Natural Language Programming Webpage Classification Test
==============================================================

This suite of codes:
  * generates a set of corpus by scraping text from a series of websites; and
  * builds a supervised machine learning classification system from the
    labelled corpus.


List of files:

./run - run script used to run the code to build the corpus, develop the machine learning tool, and test for a given url.

./src/ - Directory containing the source code.

build_corpus.py - Main program used to extract text from various categories of wikipedia websites and write to a corpus document.
* classify_documents.py - Main program used to develop machine learning classification tool from corpus developed above.
* m_io.py - Module used for input/output operations of both main programs.
* m_machine_learning.py - Module used for machine learning tasks.

./corpus/ - Directory containing the pre-generated corpus for each of the categories. The name of each text file in this directory is that of the category

./resuts/ - Directory containing the pre-processed results
* num_feature_sensitivity.dat - Text file of how the training and validation accuracy change with increasing number of word features.
* num_feature_sensitivity.png - Plot of above data.
* learning_curves.dat - Text file of how the training and validation error change with increasing amounts of data used to train the model, for the number of word features that maximises the validation accuracy.
* learning_curves.png - Plot of above data.

