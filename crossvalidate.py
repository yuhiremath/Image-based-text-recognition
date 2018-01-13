import Classifier
import numpy as np
import pandas as pd
import pickle

def load(fileName):
    with open(fileName,'rb') as infile:
        obj = pickle.load(infile)
    infile.close()
    return obj

def dump(data,fileName):
    with open(fileName,'wb') as outfile:
        pickle.dump(data,outfile)
    outfile.close()

if __name__ == "__main__":
    # test_label = [7,4,8,2,6,4,4,3,3,9,5,6,6,5,7,9,0,8,8,2,4,4,6,1,1]
    # test_label = np.float32(test_label)
    # Classifier.classifyAndPredict('0-9')
    df = pd.read_excel('Features/Final_Features.xlsx')
    Accuracy = Classifier.predictAccuracy(df)
    print('Accuracy is ' + str(Accuracy*100) + '%')