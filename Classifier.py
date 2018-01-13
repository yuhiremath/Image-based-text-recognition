import numpy as np
import segment
import pickle as pkl

def generate_validate_data(df,val=None):
    training_features = None
    training_labels = None
    test_features = None
    test_labels = None

    #Grouping the Dataframe by Labels and Validation NO.
    gp = df.groupby(['GLCM_CLASS_LABEL','VALIDATION_NO'])

    # Different groups according to groupby with name[0] and name[1] as classLabel & validationNo of corresponding group
    for name, group in gp:
        cls_label = name[0]
        val_no = name[1]

        #Popping the label and validation NO.
        labels = group.pop('GLCM_CLASS_LABEL')
        group.pop("VALIDATION_NO")
        features = group.as_matrix()

        if val_no == val:
            # Testing Features and Testing Labels
            if test_features is None:
                test_features = features
                test_labels = labels
            else:
                test_features = np.concatenate((test_features,features))
                test_labels = np.concatenate((test_labels,labels))
        else:
            # Training Features and Training Labels
            if training_features is None:
                training_features = features
                training_labels = labels
            else:
                training_features = np.concatenate((training_features,features))
                training_labels = np.concatenate((training_labels,labels))

    return training_features,training_labels,test_features,test_labels

def predictAccuracy(dataFrame):
    final_acc = 0

    print("Prediction Starting.....")
    #Checking accuracy for all five validations
    for i in range(1,6):
        print("\nPrediction of Validation "+str(i)+" starting ....")
        training_features, training_labels, test_features, test_labels = generate_validate_data(dataFrame, i)

        #MinMax normalizer to convert the data into range of 0-1
        from sklearn.preprocessing import MinMaxScaler
        normalizer = MinMaxScaler()

        # training_features shape is fit into normalized and itself is also transformed
        training_features = normalizer.fit_transform(training_features)

        # Model is created and fit to the training features and labels
        from sklearn.svm import SVC
        clf = SVC(C=1000,kernel='rbf')
        clf.fit(training_features,training_labels)
        pred = clf.predict(test_features)

        acc = calAccuracy(test_labels,pred)
        final_acc += acc
        print("Prediction of Validation " + str(i) + " done.....\n")
    # Final accuracy is divided by 5 for average accuracy of all the 5 validations
    return final_acc/5

def calAccuracy(test_labels,pred):
    # Prediction is compared with the actual test labels and the accuracy is calculated and added to final accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(test_labels, pred)
    return acc