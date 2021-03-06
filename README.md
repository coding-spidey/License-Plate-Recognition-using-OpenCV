# License Plate Recognition using OpenCV

An OpenCV based python program to detect license plates of cars present in the images.

## Execution
- Install all the modules using pip.
- Make sure that the sample images are present in a folder named `images` and it is stored in the root directory.
- Run the python file using python terminal or by simply double clicking the file.

## About Code
- It first resizes the image and then changes it into grayscale. Resizing we help us to avoid any problems with bigger resolution images, make sure the number plate still remains in the frame after resizing. Gray scaling is common in all image processing steps.
- It then uses bilateral filter to remove any background noise and unwanted details from the image.
- Next it performs edge detection using `canny` method from OpenCV. The Threshold Vale 1 and Threshold Value 2 are the minimum and maximum threshold values. Only the edges that have an intensity gradient more than the minimum threshold value and less than the maximum threshold value will be displayed.
- Now it starts looking for contours on our image. Once the counters have been detected it sorts them from big to small and consider only the first 10 results ignoring the others. In our image the counter could be anything that has a closed surface but of all the obtained results the license plate number will also be there since it is also a closed surface.
- Now that it knows where the number plate is, the remaining information is pretty much useless for us. So we can proceed with masking the entire picture except for the place where the number plate is.
- The next step in Number Plate Recognition is to segment the license plate out of the image by cropping it and saving it as a new image.
- The Final step in this Number Plate Recognition is to actually read the number plate information from the segmented image. I have used `pytesseract` here to read text present in the image. If we want to obtain better results, we can use machine learning to train tesseract.

## Output
- We have a file named `data.txt`, in it, we have stored actual number plates of the cars. We read this file and then we use `difflib` module to compare the accuracy of our results and the actual license plates.
- The output contains the resulted license plates as well as actual license plates. It also contains the accuracy of the resulted string.

### Reference :- [License Plate Recognition using OpenCV Python by Praveen](https://medium.com/programming-fever/license-plate-recognition-using-opencv-python-7611f85cdd6c)