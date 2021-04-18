import pickle


def open_scores(dataset):
    f = open("./data/%s/scores.pkl" % dataset, "rb")
    return pickle.load(f)
