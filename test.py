
# import required libraries
import cv2

# read input image
img = cv2.imread('/home/blindvision/STEREO_VISION/R_calib/21.jpg')

# convert the input image to a grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
print (ret, corners)
# if chessboard corners are detected
if ret == True:
   
   # Draw and display the corners
   img = cv2.drawChessboardCorners(img, (9,6), corners,ret)
   cv2.imshow('Chessboard',img)
   cv2.waitKey(0)
cv2.destroyAllWindows()