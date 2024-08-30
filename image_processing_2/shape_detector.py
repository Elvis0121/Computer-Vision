import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, contour):
        # initialize shape name and approximate contour
        shape = 'unknown'
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # shape detection

        # if there's three vertices, shape is a triangle
        if len(approximation) == 3:
            shape = 'triangle'

        # 4 vertices could be a square or a rectangle
        if len(approximation) == 4:
            # compute the binding box
            # use bounding box to find aspect ratio
            # if the ratio is 1, then it is a square
            # else it is a rectangle

            _, _, w, h = cv2.boundingRect(approximation)
            aspect_ratio = w/h

            shape = 'rectangle'
            if 0.95 <= aspect_ratio <= 1.05: shape = 'square'
        
        # 5 vertices give a pentagon
        if len(approximation) == 5: shape = 'pentagon'
        
        # otherwise, assume shape is a circle
        else: shape = 'circle'

        return shape
    