# Importing the required modules
import cv2
import imutils
import numpy as np
import pytesseract
import re
import difflib
import os

# setting the tesseract cmd
# tesseract is used to detect texts in Images
pytesseract.pytesseract.tesseract_cmd = r'{{ Enter the directory of tesseract here }}'

# Obtaining the list of sample images
directory = r'images'
samepleImages = os.scandir(directory)

# declaring the lists to store outcomes
result = []
accuracy = []
actual = []

# code for detecting license plate numbers
# more information provided in the documentation of the project
for sampleImage in samepleImages:
    image = cv2.imread(os.path.join(sampleImage), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (600, 400))

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 13, 15, 15)

    edged = cv2.Canny(grey, 30, 200)
    contoursDetected = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursDetected = imutils.grab_contours(contoursDetected)
    contoursDetected = sorted(
        contoursDetected, key=cv2.contourArea, reverse=True)[:10]
    screenCount = None

    for contour in contoursDetected:

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        if len(approx) == 4:
            screenCount = approx
            break

    if screenCount is None:
        found = False
        print("Couldn't detect any contours")
    else:
        found = True

    if found:
        cv2.drawContours(image, [screenCount], -1, (0, 0, 255), 3)

    mask = np.zeros(grey.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCount], 0, 255, -1,)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottom_x, bottom_y) = (np.max(x), np.max(y))
    cropped = grey[topx:bottom_x+1, topy:bottom_y+1]

    text = pytesseract.image_to_string(cropped, config='--psm 11')
    result.append(re.sub(r'[^\w -]', '', text))

# opening the data file in which the actual license plate numbers are stored and storing it in a list
file = open('data.txt')
for line in file:
    line = line.rstrip()
    actual.append(line)

# comparing the result we got through openCV and actual data, to find the accuracy using the difflib module
for i in range(10):
    ratio = difflib.SequenceMatcher(None, actual[i], result[i]).ratio()
    accuracy.append(str(round(ratio*100)))

# printing the results
print("=========================================")
print("Result       \tActual       \tAccuracy")
print("=========================================")
for i in range(10):
    print("{:<13}".format(result[i])+'\t' +
          "{:<13}".format(actual[i])+'\t'+accuracy[i])
print("=========================================")

avg_accuracy  = 0
for accurate in accuracy:
    avg_accuracy += int(accurate)

print(f'Average accuracy is = {avg_accuracy/10}%')
print('\n')
