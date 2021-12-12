import cv2 
import numpy as np

def showimg(title, img):
    cv2.imshow(title, img)
    cv2.moveWindow(title, 200, 200)
    cv2.waitKey(0)


# read in image
file_path = "testImages\connect4_img_test.jpg"
board_image = cv2.imread(file_path)

# process image
new_width = 500 # resize 
img_height, img_width, _ = board_image.shape
scale = new_width / img_width
img_width = int(img_width * scale)
img_height = int(img_height * scale)
board_img = cv2.resize(board_image, (img_width, img_height), interpolation=cv2.INTER_AREA)
img_orig = board_img.copy()
showimg('original image resized', img_orig)

# apply bilateral filter
bi_filt_img = cv2.bilateralFilter(board_image, 15, 190, 190)
showimg('bilateralFilter', bi_filt_img)

# outline edges
edge_img = cv2.Canny(bi_filt_img, 75, 150)
showimg('edges', edge_img)

# Find Circles
_,  contours, heirarchies = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Edges to contours


contour_list = []
rect_list = []
position_list = []

for contour in contours:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) # Contour Polygons
    area = cv2.contourArea(contour)
    
    rect = cv2.boundingRect(contour) # Polygon bounding rectangles
    x_rect,y_rect,w_rect,h_rect = rect
    x_rect +=  w_rect/2
    y_rect += h_rect/2
    area_rect = w_rect*h_rect
    
    if ((len(approx) > 8) & (len(approx) < 23) & (area > 250) & (area_rect < (img_width*img_height)/5)) & (w_rect in range(h_rect-10,h_rect+10)): # Circle conditions
        contour_list.append(contour)
        position_list.append((x_rect,y_rect))
        rect_list.append(rect)

img_circle_contours = img_orig.copy()
cv2.drawContours(img_circle_contours, contour_list,  -1, (0,255,0), thickness=1) # Display Circles
for rect in rect_list:
    x,y,w,h = rect
    cv2.rectangle(img_circle_contours,(x,y),(x+w,y+h),(0,0,255),1)

showimg('Circles Detected',img_circle_contours)

