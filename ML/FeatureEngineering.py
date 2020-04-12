import numpy as np
from os import path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
import pickle
import pandas as pd

class FeatureEngineering:
    def __init__(self, data=None):
        if data is not None:
            self.df = data
        self.models_path = r'ML\SavedModels'
        self.bow_model_path = path.join(self.models_path, 'count_vectorizer.pkl')
        self.diff_idf_path = path.join(self.models_path, 'diff_idf.pkl')
        self.indices_path = path.join(self.models_path, 'diff_idf_indices.pkl')

    @staticmethod
    def join_text(text):
        new_text = []
        for x in text:
            if type(x) is str:
                new_text.append(x)
            else:
                new_text.append(" ".join(x))
        return new_text

    def train_bag_of_words(self, ngram=2):
        vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, ngram), stop_words=None)
        new_text = FeatureEngineering.join_text(self.df["sentences"])
        count_vectorizer = vectorizer.fit(new_text)
        with open(self.bow_model_path, 'wb') as f:
            pickle.dump(count_vectorizer, f)
        return new_text

    def generate_bag_of_words(self, data):
        new_text = FeatureEngineering.join_text(data)
        with open(self.bow_model_path, 'rb') as f:
            count_vectorizer = pickle.load(f)
        return count_vectorizer.transform(new_text)

    def find_diff_tfidf(self, bag_of_words, labels, file_path, diff_size=1000, satire_size=100, fake_size=100):
        mask_satire = labels == "satire"
        mask_satire = mask_satire.values
        mask_fake = np.logical_not(mask_satire)

        # Use diff tfidf metric
        N_satire = np.sum(mask_satire)
        N_fake = np.sum(mask_fake)
        satire_idf = np.log(N_satire / (1 + np.sum(bag_of_words[mask_satire, :] != 0, axis=0)))
        fake_idf = np.log(N_fake / (1 + np.sum(bag_of_words[mask_fake, :] != 0, axis=0)))

        diff_idf = 2 * np.abs(satire_idf - fake_idf) / (satire_idf + fake_idf)
        ind_sorted_diff_idf = np.argsort(diff_idf)
        best_diff_indices = ind_sorted_diff_idf[0, -diff_size:]
        best_diff_indices = np.squeeze(np.asarray(best_diff_indices))

        most_common_satire = np.squeeze(np.asarray(np.argsort(satire_idf)[0, 0:satire_size]))
        most_common_fake = np.squeeze(np.asarray(np.argsort(fake_idf)[0, 0:fake_size]))
        best_indices = np.unique(np.concatenate([best_diff_indices, most_common_satire, most_common_fake]))
        tf = bag_of_words[:, best_indices]
        idf = diff_idf[:, best_indices]
        idf_arr = np.squeeze(np.asarray(np.repeat(idf, tf.shape[0], axis=0)))
        tf_arr = np.squeeze(np.asarray(tf.todense()))
        tf_arr = normalize(tf_arr, axis=1, norm='l2')
        tfidf = tf_arr * idf_arr
        tfidf_data = {"tfidf": tfidf, "labels": labels, "file path": file_path}
        with open(self.diff_idf_path, 'wb') as f:
            pickle.dump(idf, f)
        with open(self.indices_path, 'wb') as f:
            pickle.dump(best_indices, f)
        return tfidf_data

    def transfer_diff_tfidf(self, bag_of_words):
        with open(self.diff_idf_path, 'rb') as f:
            diff_idf = pickle.load(f)
        with open(self.indices_path, 'rb') as f:
            indices = pickle.load(f)
        tf_arr = np.squeeze(np.asarray(bag_of_words[:, indices].todense()))
        idf_arr = np.squeeze(np.asarray(np.repeat(diff_idf, tf_arr.shape[0], axis=0)))
        tfidf = tf_arr * idf_arr
        return tfidf
