import pickle

def first_model():
    model = pickle.load(open('parklane/RF_first.sav', 'rb'))

    return model

def second_model():
    model = pickle.load(open('parklane/RF_second.sav', 'rb'))

    return model