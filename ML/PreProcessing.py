import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import punkt
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from os import path
import glob
import pandas as pd
import numpy as np
np.random.seed(0)

class PreProcessCorpus:
    def __init__(self, dir_path):
        self.satire = []
        self.fake = []
        self.dir_path = dir_path
        # sentence - each sentence in the text
        # parsing type - regular / stemmed
        # id - file name
        # text type - fake news / satire
        self.data_dict = {"sentences": [], "parsing type": [], "file path": [], "text type": []}

    def add_parsed_file(self, file_sentences, parsing_type, file_path, text_type):
        num_sentences = len(file_sentences)
        self.data_dict["sentences"] += file_sentences
        self.data_dict["parsing type"] += [parsing_type]*num_sentences
        self.data_dict["file path"] += [file_path]*num_sentences
        self.data_dict["text type"] += [text_type]*num_sentences

    @staticmethod
    def read_file(file_path, is_stemmed):
        try:
            f = open(file_path, encoding="utf8")
            text = f.read()
            f.close()
        except:
            f = open(file_path)
            text = f.read()
            f.close()
        pre_process = PreProcessText(text)
        return pre_process.process(is_stemmed=is_stemmed)

    def add_all_files(self):
        for text_type in ['satire', 'fake']:
            if text_type is 'satire':
                text_path = r'Satire\finalSatire'
            elif text_type is 'fake':
                text_path = r'Fake\finalFake'
            else:
                raise Exception("Wrong text type!")
            type_path = path.join(self.dir_path, text_path)
            for file_path in glob.glob(f"{type_path}/*.txt"):
                stemmed_file_text = self.read_file(file_path=file_path, is_stemmed=True)
                self.add_parsed_file(stemmed_file_text, "stemmed", file_path, text_type)

    def generate_data_frame(self, test_perc=0.15):
        self.add_all_files()
        data = pd.DataFrame(self.data_dict)
        shuffled_text_names = np.random.permutation(data['file path'].unique())
        text_names = np.array_split(shuffled_text_names, int(1/test_perc))
        validation_text = text_names.pop(0)
        test_text = text_names.pop(0)
        train_text = np.concatenate(text_names)
        cross_mask_train = data["file path"].isin(train_text)
        cross_mask_validation = data["file path"].isin(validation_text)
        cross_mask_test = data["file path"].isin(test_text)
        train_data = data[cross_mask_train].sample(frac=1).reset_index(drop=True)
        validation_data = data[cross_mask_validation].sample(frac=1).reset_index(drop=True)
        test_data = data[cross_mask_test].sample(frac=1).reset_index(drop=True)
        return train_data, validation_data, test_data


class PreProcessText:
    def __init__(self, text: str):
        self.text = BeautifulSoup(text, "lxml").text
        self.text = self.text.replace('\n', ' ')
        self.stem_text = None
        self.text_num = 0
        self.stem_text_num = 0

    def remove_url(self):
        self.text = ' '.join(x for x in self.text.split() if x.startswith('http') == False and x.startswith('www') == False)

    def stemming(self):
        tokenizer = TweetTokenizer()
        new_sentences = []
        for x in self.text:
            x = tokenizer.tokenize(x)
            new_sentences.append(x)
        self.text = new_sentences
        porter_stemmer = PorterStemmer()
        new_sentences_stemmered = []
        for x in new_sentences:
            # new_sentences_stemmered.append(' '.join(porter_stemmer.stem(word) for word in x))
            new_sentences_stemmered.append(tokenizer.tokenize(' '.join(porter_stemmer.stem(word) for word in x)))
        self.stem_text = new_sentences_stemmered

    def sentencing(self):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.text = sent_detector.tokenize(self.text.strip())

    def process(self, is_stemmed):
        self.remove_url()
        self.sentencing()
        self.stemming()
        if is_stemmed:
            return self.get_stemmed()
        else:
            return self.get_text()

    def get_text(self):
        return self.text

    def get_stemmed(self):
        return self.stem_text
