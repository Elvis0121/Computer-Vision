import cv2
import numpy as np

image = cv2.imread('random_images/sample_shapes.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# make binary version of image
ret, bw = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# finding components
connectivity = 4
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
sizes = stats[1:, -1]; nb_components -= 1
min_size = 250 # threshold value for objects in view
new_image = np.zeros((image.shape), np.uint8)

for i in range(0, nb_components + 1):
    #identify objects using if sizes >= min_size

    color = (255, 255, 255)
    #color = np.random.randint(255, size = 3)
    

    # drawing a rectangle to bound the objects
    # cv2.rectangle(
    #     new_image,
    #     (stats[i][0], stats[i][1]),
    #     (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]),
    #     (255, 0, 0), 1
    # )
    cv2.rectangle(new_image, (100,100), (200, 150), (100, 0, 0), 1)

    new_image[output == i + 1] = color

cv2.imshow('Custom Detect', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
