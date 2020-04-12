from ML.PreProcessing import PreProcessCorpus
from ML.FeatureEngineering import FeatureEngineering
from ML.ModelTraining import Gdb

if __name__ == "__main__":
    corpus = PreProcessCorpus(dir_path=r"C:\ProjectNLP\FakeNewsData\StoryText 2")
    train_data, validation_data, test_data = corpus.generate_data_frame()

    fe = FeatureEngineering(data=train_data)
    fe.train_bag_of_words(2)
    train_bag_of_words = fe.generate_bag_of_words(train_data['sentences'])
    validation_bag_of_words = fe.generate_bag_of_words(validation_data['sentences'])
    test_bag_of_words = fe.generate_bag_of_words(test_data['sentences'])

    tfidf_train = fe.find_diff_tfidf(train_bag_of_words, labels=train_data["text type"],
                                     file_path=train_data["file path"], diff_size=200, satire_size=500, fake_size=500)
    tfidf_validation = {"tfidf":fe.transfer_diff_tfidf(validation_bag_of_words), "file path": validation_data["file path"],
                        "labels": validation_data["text type"]}
    tfidf_test = {"tfidf":fe.transfer_diff_tfidf(test_bag_of_words), "file path": test_data["file path"],
                  "labels": test_data["text type"]}

    tfidf_train["labels"] = 1 * (tfidf_train["labels"] == "fake")
    tfidf_validation["labels"] = 1 * (tfidf_validation["labels"] == "fake")
    tfidf_test["labels"] = 1 * (tfidf_test["labels"] == "fake")

    x_train = tfidf_train["tfidf"]
    y_train = tfidf_train["labels"].values
    x_validation = tfidf_validation["tfidf"]
    y_validation = tfidf_validation["labels"].values
    x_test = tfidf_test["tfidf"]
    y_test = tfidf_test["labels"].values

