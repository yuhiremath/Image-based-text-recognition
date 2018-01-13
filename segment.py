import numpy as np
import cv2
import FeatureExtract

def seglist():
    roiList = []
    feature = []

    img = cv2.imread('Pictures\\test1row.png')
    #gray conversion of the image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #image thresholding
    ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    #contours is assigned all the cordinates of the contours
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        # Restricting the contour area
        if cv2.contourArea(cnt) > 20 and cv2.contourArea(cnt) < 700:
            [x, y, w, h] = cv2.boundingRect(cnt)
            # Restricting the height of the contour
            if h > 28:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                roismall = gray[y:y + h, x:x + w]
                roismall = np.float32(roismall)
                resized_roismall = FeatureExtract.preprocess(roismall)
                roiList.append(resized_roismall)

    print("Extracting features for Contours")
    for i in roiList:
        feature.append(FeatureExtract.getGLCMFeatures(i))
    return feature