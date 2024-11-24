import string
from bs4 import BeautifulSoup  # For removing HTML
import contractions  # For expanding contractions
from unidecode import unidecode  # For handling accented words
import re
import pkg_resources
from symspellpy import SymSpell, Verbosity  # For spelling correction
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from src.textSummarizer.config.configuration import TextPreProcessingConfig
from typing import List
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from src.textSummarizer.logging import logger

# Define the TextProcessing class
class TextPreProcessing:
    def __init__(self, config: TextPreProcessingConfig):
        self.config = config
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def preprocess_text(self, text):

        def remove_html(text):
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text()

        def remove_urls(text):
            pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)(/\w*)?')
            return re.sub(pattern, "", text)

        def remove_emails(text):
            pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
            return re.sub(pattern, "", text)

        def handle_accents(text):
            return unidecode(text)

        def remove_unicode_chars(text):
            return text.encode("ascii", "ignore").decode()

        def expand_contractions(text):
            """Expands contractions in the text (e.g., don't -> do not)."""
            return contractions.fix(text)
        
        def remove_punctuations(text):
            return re.sub('[%s]' % re.escape(string.punctuation), " ", text)

        def remove_digits(text):
            pattern = re.compile(r'\w*\d+\w*')
            return re.sub(pattern, "", text)

        def remove_extra_spaces(text):
            return re.sub(' +', ' ', text).strip()

        def correct_spelling_symspell(text):
            words = word_tokenize(text)  # Tokenize the text into words
            tagged_words = pos_tag(words)  # Perform POS tagging

            corrected_words = []
            for word, tag in tagged_words:
                if tag in ('NNP', 'NNPS'):  # Skip proper nouns
                    corrected_words.append(word)
                else:
                    corrected_word = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
                    corrected_words.append(corrected_word[0].term if corrected_word else word)

            return " ".join(corrected_words)

        # Step 1: Split the text into sentences for better structure
        sentences = sent_tokenize(text)

        processed_lines = []

        for sentence in sentences:
            if self.config.remove_html:
                sentence = remove_html(sentence)
            if self.config.remove_urls:
                sentence = remove_urls(sentence)
            if self.config.remove_emails:
                sentence = remove_emails(sentence)
            if self.config.handle_accents:
                sentence = handle_accents(sentence)
            if self.config.remove_unicode_chars:
                sentence = remove_unicode_chars(sentence)
            if self.config.expand_contractions:
                sentence = expand_contractions(sentence)
            if self.config.remove_punctuations:
                sentence = remove_punctuations(sentence)
            if self.config.remove_digits:
                sentence = remove_digits(sentence)
            if self.config.remove_extra_spaces:
                sentence = remove_extra_spaces(sentence)
            if self.config.correct_spelling:
                sentence = correct_spelling_symspell(sentence)

            if sentence.strip():  # Skip empty lines
                processed_lines.append(sentence)

        # Step 2: Return cleaned sentences
        return processed_lines

    def process_text_file(self, processed_data: List):
        """ Process a text file using the preprocessing pipeline. """
        # Create delayed tasks for each preprocessing
        tasks = [delayed(self.preprocess_text)(text=text) for text in processed_data]

        # Progress bar using dask's diagnostics
        with ProgressBar():
            results = compute(*tasks, scheduler='processes')
        processed_text = []
        
        for result in results:
            if "error" in result:
                logger.error(f"Error processing text: {result['error']}")
            else:
                logger.info(f"Text processed successfully. Text: {result[:10]}")
                processed_text.append("\n".join(result))
        print("len len\n", len(processed_text))
        import pandas as pd
        df = pd.read_csv('text.csv')
        df = pd.concat([df,pd.DataFrame(processed_text,columns=['processed_text'])],axis=1)
        df.to_csv('text2.csv', index=False)
        return processed_text

