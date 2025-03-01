#/usr/bin/bash

# Install NLTK
pip install nltk
python <<EEE
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_eng')
nltk.download('words_eng')
nltk.download('stopwords_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')  
EEE
