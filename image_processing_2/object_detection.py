import cv2
import numpy as np



img1 = cv2.imread("random_images/square.png")
blur1 = cv2.GaussianBlur(img1,(5,5),0)
gray1=cv2.cvtColor(blur1,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray1,65,255,cv2.THRESH_BINARY_INV)

img2=cv2.imread("random_images/square.png")
blur2 = cv2.GaussianBlur(img2,(5,5),0)
gray2=cv2.cvtColor(blur2,cv2.COLOR_BGR2GRAY)
ret,thresh2 = cv2.threshold(gray2,65,255,cv2.THRESH_BINARY_INV)

diff=cv2.absdiff(thresh2,thresh1)
diff=cv2.bitwise_xor(diff,thresh1)

kernel = np.ones((2,2),np.uint8)
diff=cv2.erode(diff,kernel,iterations = 1)
diff=cv2.dilate(diff,kernel,iterations = 8)

contours, _= cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c=max(contours,key=cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(diff,(x,y),(x+w,y+h),(125,125,125),2)


cv2.imshow("Contours image",diff)
cv2.waitKey(0)

img = cv2.imread('random_images/sample_shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# binarize the image
ret, bw = cv2.threshold(gray, 128, 255, 
cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# find connected components
connectivity = 4
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
sizes = stats[1:, -1]; nb_components = nb_components - 1
min_size = 250 #threshhold value for objects in scene
img2 = np.zeros((img.shape), np.uint8)
for i in range(0, nb_components+1):
    # use if sizes[i] >= min_size: to identify your objects
    color = np.random.randint(255,size=3)
    # draw the bounding rectangele around each object
    cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
    img2[output == i + 1] = color
cv2.imshow('Window', img2)
cv2.waitKey(0)







