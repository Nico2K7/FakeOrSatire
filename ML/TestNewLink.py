from ML.PreProcessing import PreProcessText
from Scraper.DataScraper import Scraper
from ML.FeatureEngineering import FeatureEngineering
import catboost as cb
import numpy as np


class LinkTester:
    def __init__(self, link):
        self.link = link
        self.classifier = LinkTester.read_classifier(model_path = r"ML\SavedModels\cataboost_classifier.cbm")

    @staticmethod
    def read_classifier(model_path):
        cb_estimator = cb.CatBoostClassifier()
        cb_estimator.load_model(model_path)
        return cb_estimator

    def get_text_classification(self, text):
        prob = self.classifier.predict_proba(text)[:,1]
        is_fake = np.mean(prob)>0.5
        return is_fake

    def test(self):
        scraper = Scraper(link=self.link)
        parsed_text = scraper.get_parsed_text()
        pre_process = PreProcessText(text=parsed_text)
        processed_text = pre_process.process(is_stemmed=True) # important! must be the same as in training
        fe = FeatureEngineering()
        processed_text = fe.join_text(processed_text)
        bow = fe.generate_bag_of_words(processed_text)
        tfidf_diff = fe.transfer_diff_tfidf(bow)
        is_fake = self.get_text_classification(tfidf_diff)
        return is_fake, parsed_text


if __name__ == "__main__":
    link = r"https://web.archive.org/web/20170228020952/http://nationalreport.net/colorado-pot-shop-accept-food-stamps-taxpayer-funded-marijuana/"
    link_tester = LinkTester(link=link)
    is_fake, parsed_text = link_tester.test()
    print(is_fake)