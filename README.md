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

### Training
* Characters 0-9 of Char74k data set is used for training the data.
2. Features for each character are extracted using GLCM matrix with different combinations of angles (0, 45, 90, 135) and distances (1, 3, 5)
3. Extracted features are stored in an excel file.

