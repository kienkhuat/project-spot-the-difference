from skimage.metrics import structural_similarity
import imutils
import cv2
import json
import random
import os

#GENERATE ALTERED GAME IMAGE
inputImage = cv2.imread('./images/pic3a.png')
#Level 1
height, width, dim = inputImage.shape
numberOfCircles = 5
imageWithCircles = inputImage.copy()
for i in range(0, numberOfCircles):
	cv2.circle(imageWithCircles, (random.randint(0, width), random.randint(0, height)), 15, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

cv2.imshow("Image with rectangles", imageWithCircles)

#level 2
grayInputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
cannyImage = cv2.Canny(grayInputImage, 0, 100) #image with edges
# threshImage = cv2.threshold(grayInputImage, 0, 255, cv2.THRESH_BINARY_INV)[1]
genContours = cv2.findContours(cannyImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
genContours = imutils.grab_contours(genContours)

contourToRecolorList = []
for c in genContours:
	area = cv2.contourArea(c)
	if 5 <= area <= 500:
		contourToRecolorList.append(c)		

alteredImage = inputImage.copy()

randomContourToRecolorList = random.sample(contourToRecolorList, 20)
for c in randomContourToRecolorList:
	r = random.randint(0, 100) #random RGB number within 0-100 for darker color
	g = random.randint(0, 100)
	b = random.randint(0, 100)
	cv2.fillPoly(alteredImage, pts=[c], color=[r, g, b])

cv2.imshow("cannyImage", cannyImage)
cv2.imshow("alteredImage", alteredImage)

cv2.imwrite(os.path.join('./images', 'alteredImage.png'), alteredImage)

#GET IMAGE DIFFERENCE DATA
paintingImageFirst = './images/pic1a.png'
paintingImageSecond = './images/pic1b.png'
parkImageFirst = './images/pic2a.png'
parkImageSecond = './images/pic2b.png'
sledgeImageFirst = './images/pic3a.png'
sledgeImageSecond = './images/pic3b.png'

firstImage = inputImage
secondImage = alteredImage

# convert to grayscale
grayFirstImage = cv2.cvtColor(firstImage, cv2.COLOR_BGR2GRAY)
graySecondImage = cv2.cvtColor(secondImage, cv2.COLOR_BGR2GRAY)

# use Structural Similarity Index (SSIM) from Scikit to compare
(score, diff) = structural_similarity(grayFirstImage, graySecondImage, full = True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# subtractedImage = cv2.subtract(firstImage, secondImage)
subtractedImage = cv2.absdiff(grayFirstImage, graySecondImage)
# graySubtractedImage = cv2.cvtColor(subtractedImage, cv2.COLOR_BGR2GRAY)

# threshold the diff image then find contours to get the region of differences
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
threshSubtracted = cv2.threshold(subtractedImage, 245, 255, cv2.THRESH_OTSU)[1]

contours = cv2.findContours(threshSubtracted.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

#remove small contours
for c in contours:
	area = cv2.contourArea(c)
	if area < 1:
		cv2.fillPoly(threshSubtracted, pts=[c], color=0)
		continue

# merge close contours
threshMergedGray = cv2.morphologyEx(threshSubtracted, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)));
contours = cv2.findContours(threshMergedGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = imutils.grab_contours(contours)

jsonData = []
offset = 5
for c in contours:
	area = cv2.contourArea(c)
	(x, y, w, h) = cv2.boundingRect(c)
	jsonData.append({
		"xAxis": x - offset,
		"yAxis": y - offset,
		"width": w + offset,
		"height": h + offset,
	})
	cv2.rectangle(firstImage, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 0, 255), 2)
	cv2.rectangle(secondImage, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 0, 255), 2)

# write to json file
json_object = json.dumps(jsonData, indent=4)
#print(json_object)
with open("gameData.json", "w") as outfile:
    outfile.write(json_object)

# show the output images
cv2.imshow("Subtract", subtractedImage)
# cv2.imshow("First Image", firstImage)
# cv2.imshow("Second Image", secondImage)
# cv2.imshow("Subtracted", subtractedImage)
# cv2.imshow("Thresh Subtracted Merged", threshMergedGray)
# cv2.imshow("Diff", diff)
# cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

