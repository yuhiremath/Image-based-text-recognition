# Image-based-text-recognition
Recognition and classification of alphanumeric characters from image or pdf files.

## Prerequisites
1. python 3.5.2
2. matplotlib 1.5.3
3. pandas 0.18.1
4. numpy 1.11.1
5. SimpleITK 0.10.0
6. scipy 0.18.1

## Usage
1. Run "python generatefeatureset.py" (set the variables "datafolder", "featuresfolder" and "outputfile" accordingly before running).
2. Run "python crossvalidate.py" to get the crossvalidation results.

## Project Description

### Data
* Characters 0-9 of Char74k data set is used for training the data.

### Pre-Processing
* The images are first converted into grayscale and binary thresholded to convert them into black and white images.
* The images are cropped with necessary zero padding, so that they are contained in a square.
* All the images are resized to a single size of 100x100. 

### Feature Extraction
* A 100x100 image is sub-divided into 5 parts horizontally.
* For each part of the sub-divided image, GLCM features with different combinations of angles (0, 45, 90, 135) and distances (1, 3, 5) are extracted.
* Extracted features are stored in an excel file.

### Classification
* A five fold cross validation technique is used to classify the images in the dataset.
* SVM classifier with regularization parameter, C = 1000 with 'rbf' kernel is used to classify the images.

### Result
* The cross validation accuracy for digits 0-9 is 84.6%.
