from FeatureExtract import extractFeatures
import os

datafolder = "Data/*"
featuresfolder = "Features"
outputfile = "Final_Features.xlsx"

if not os.path.exists(featuresfolder):
    os.makedirs(featuresfolder)


extractFeatures(datafolder, "{}/{}".format(featuresfolder,outputfile))
