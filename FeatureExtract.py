from skimage.feature import greycoprops,greycomatrix
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import glob
import SimpleITK as sitk
import scipy.misc as sci

# Returns all column names with patch NO., distance and angles
def getGLCMColumnNames(patch=[1,2,3,4,5],distances = [1,3,5],angles=[0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0]):
    glcm_columns = []
    for i in range(len(patch)):
        for j in range(len(distances)):
            for k in range(len(angles)):
                glcm_columns.append('GlCM_CONTRAST_{}_{}_{}'.format(patch[i],distances[j],angles[k]))
                glcm_columns.append('GlCM_DISSIMILARITY_{}_{}_{}'.format(patch[i],distances[j],angles[k]))
                glcm_columns.append('GlCM_HOMOGENEITY_{}_{}_{}'.format(patch[i],distances[j],angles[k]))
                glcm_columns.append('GlCM_ASM_{}_{}_{}'.format(patch[i],distances[j],angles[k]))
                glcm_columns.append('GlCM_ENERGY_{}_{}_{}'.format(patch[i],distances[j],angles[k]))
                glcm_columns.append('GlCM_CORRELATION_{}_{}_{}'.format(patch[i],distances[j],angles[k]))

    return glcm_columns

# returns the FeatureVector for all distance and angles
def getGLCMFeatures(image, distances=[1,3,5], angles=[0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0]):
    glcm_feature_vector = []

    x = 0
    for i in range(0,5):
        patch = image[x:x+20][:]
        x+=20
        glcm = greycomatrix(patch, distances, angles, 2, symmetric=True, normed=True)
        for j in range(len(distances)):
            for k in range(len(angles)):
                contrast = float(greycoprops(glcm, 'contrast')[j, k])
                glcm_feature_vector.append(contrast)

                dissimilarity = greycoprops(glcm, 'dissimilarity')[j, k]
                glcm_feature_vector.append(dissimilarity)

                homogeneity = greycoprops(glcm, 'homogeneity')[j, k]
                glcm_feature_vector.append(homogeneity)

                ASM = greycoprops(glcm, 'ASM')[j, k]
                glcm_feature_vector.append(ASM)

                energy = greycoprops(glcm, 'energy')[j, k]
                glcm_feature_vector.append(energy)

                correlation = greycoprops(glcm, 'correlation')[j, k]
                glcm_feature_vector.append(correlation)

    glcm_feature_vector = np.array(glcm_feature_vector)
    return glcm_feature_vector

def preprocess(img):
    s_image = sitk.GetImageFromArray(img)

    #Inversion and threshold
    if(np.max(img) <= 1):
        inverted = sitk.InvertIntensity(s_image, maximum=1)
        thresh = sitk.BinaryThreshold(inverted, 0.5, 1)
    elif(np.max(img) <= 255):
        inverted = sitk.InvertIntensity(s_image, maximum=255)
        thresh = sitk.BinaryThreshold(inverted, 10, 255)

    thresh_arr = sitk.GetArrayFromImage(thresh)

    #Cropping started
    ind = np.where(thresh_arr == 1)
    min_x = np.min(ind[0])
    max_x = np.max(ind[0])
    min_y = np.min(ind[1])
    max_y = np.max(ind[1])

    length = max_x - min_x
    breadth = max_y - min_y

    mid_x = min_x + (int(length / 2) + 1)
    mid_y = min_y + ((int)(breadth / 2) + 1)

    if length > breadth:
        new_min_x = min_x
        new_max_x = max_x
        new_min_y = np.ceil(mid_y - length / 2)
        new_max_y = np.ceil(mid_y + length / 2)
        if new_min_y < 0:
            new_max_y = new_max_y + abs(new_min_y)
            new_min_y = 0
        if new_max_y > 128:
            new_min_y -= (new_max_y - 128)
            new_max_y = 128

    else:
        new_min_y = min_y
        new_max_y = max_y
        new_min_x = np.ceil(mid_x - breadth / 2)
        new_max_x = np.ceil(mid_x + breadth / 2)
        if new_min_x < 0:
            new_max_x = new_max_x + abs(new_min_x)
            new_min_x = 0
        if new_max_x > 128:
            new_min_x -= (new_max_x - 128)
            new_max_x = 128
    # Square Cropped Image
    cropped = thresh_arr[int(new_min_x):int(new_max_x), int(new_min_y):int(new_max_y)]

    # Resizing of the cropped image to 100 by 100
    resized = sci.imresize(cropped, (100, 100), interp='bicubic')
    return resized

def extractFeatures(foldername,outputfile):
    # files assigned to Folder names
    files = glob.glob(foldername)
    i = -1
    cols = getGLCMColumnNames()
    cols.append('GLCM_CLASS_LABEL')
    cols.append('VALIDATION_NO')
    features = None

    for images in files:
        print(images)
        # Retreival of the label number
        i = int(images.split("\\")[-1].split('e')[-1])
        i = i - 1
        spec_images = glob.glob(images + '\*.png')
        for image in spec_images:
            # Retreival of validation no.
            validation = image.split('\\')[-1].split('.')[0].split('-')[-1]
            validation = (int(validation) % 5) + 1
            img = mpimg.imread(image)
            resized = preprocess(img)

            #feature vector being generated
            feature_vector = getGLCMFeatures(resized)
            feature_vector = np.append(feature_vector, i)
            feature_vector = np.append(feature_vector, validation)
            feature_vector = np.column_stack(feature_vector)
            # print(str(image)+' is done...')
            if features is None:
                features = feature_vector
            else:
                features = np.concatenate((features, feature_vector))
        df = pd.DataFrame(features, columns=cols)
        df.to_excel(outputfile)