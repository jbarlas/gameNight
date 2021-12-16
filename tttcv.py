# imports
import numpy as np
import cv2
from AdversarialSearch.tttproblem import TTTProblem, TTTUI
from AdversarialSearch import adversarialsearch

# initializes the "board"
board = [["-","-","-"],["-","-","-"],["-","-","-"]]

## INPUT ##
# change to "no" if you do not want to see the processing steps displayed when running the program
see_steps = "yes"
# change to "yes" if you want all of the images to stay up as you proceed through the steps of the program
keep_all_imgs = "no"

# list of images you can run the program on
marker = "testImages/ttt_marker.jpg"
printed = "testImages/ttt_printed.png"
green = "testImages/X_O_2.png"
empty = "testImages/empty.png"
stock = "testImages/stock.png"
# the algoritm is able to correctly identify the squares on this image
# does not correctly identify what X's and O's are where
drawn = "testImages/drawn.jpg"

## INPUT ##
# resizes and displays the image
# change the argument of the next line to run the program on a different image
img = cv2.imread(marker)
img_width, img_height = 600, 600
img = cv2.resize(img,(img_width,img_height))
cv2.imshow('Initial Image (Step 1)',img)
cv2.waitKey(0)
if keep_all_imgs == "no":
    cv2.destroyAllWindows()

# creates and displays a grayscale image
img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
if see_steps == "yes":
    cv2.imshow('Grayscale Image (Step 2)',img_grayscale)
    cv2.waitKey(0)
    if keep_all_imgs == "no":
        cv2.destroyAllWindows()

# converts grayscale image into binary image, pixels with value < 127 become 0, pixels with value > 127 become 255
# displays binary image
threshold, img_threshold = cv2.threshold(img_grayscale,127,255,cv2.THRESH_BINARY)
if see_steps == "yes":
    cv2.imshow('Thresholded Image (Step 3)',img_threshold)
    cv2.waitKey(0)
    if keep_all_imgs == "no":
        cv2.destroyAllWindows()

# gets and displays edges of binary images
img_edge_detection = cv2.Canny(img_threshold, 75, 150) 
if see_steps == "yes":
    cv2.imshow('Edge Detection Image (Step 4)', img_edge_detection)
    cv2.waitKey(0)
    if keep_all_imgs == "no":
        cv2.destroyAllWindows()

# uses the edges from above to find a bounding rectangle for the tic-tac-toe board
# draws and displays a bounding rectangle around the board
white_pixel_indices = np.argwhere(img_edge_detection==255)
min_y = np.min(white_pixel_indices[:,0])
min_x = np.min(white_pixel_indices[:,1])
max_y = np.max(white_pixel_indices[:,0])
max_x = np.max(white_pixel_indices[:,1])
img_threshold = cv2.rectangle(img_threshold,(min_x,min_y),(max_x,max_y),(0,0,0),15)
if see_steps == "yes":
    cv2.imshow('Bounded Image (Step 5)', img_threshold)
    cv2.waitKey(0)
    if keep_all_imgs == "no":
        cv2.destroyAllWindows()

# crops the image around the bounding rectangle so there is no space outside of the board
img_threshold_cropped = img_threshold[min_y:max_y,min_x:max_x]
img = img[min_y:max_y,min_x:max_x]
if see_steps == "yes":
    cv2.imshow('Cropped Threshold Image (Step 6)', img_threshold_cropped)
    cv2.waitKey(0)
    if keep_all_imgs == "no":
        cv2.destroyAllWindows()

# finds the large, exterior contours of the cropped image
# this finds the 9 different boxes of the tic-tac-toe board
# draws the contours
im2, contours, heirarcy = cv2.findContours(img_threshold_cropped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
img_contours = cv2.drawContours(img,contours,-1,(255,0,0),5)
if see_steps == "yes":
    cv2.imshow('Contour Image (Step 7)',img_contours)
    cv2.waitKey(0)
    if keep_all_imgs == "no":
        cv2.destroyAllWindows()

# loops through the contours if they are above a certain size (this excludes small contours)
contour_x = 0
contour_y = 0
for contour in contours:
    if cv2.contourArea(contour) > 10000:
        # displays an image for each square
        x,y,w,h = cv2.boundingRect(contour)
        contour_x = w
        contour_y = h
        square = img_threshold_cropped[y+2:y+h-2,x+2:x+w-2]
        if see_steps == "yes":
            cv2.imshow('Square Image',square)
            cv2.waitKey(0)
            if keep_all_imgs == "no":
                cv2.destroyAllWindows()
        # determines the spot on the board to which the contour refers
        board_y = round(x/w)
        board_x = round(y/h)
        # finds the contours within the smaller box image
        sub_im2, sub_contours, sub_heirarcy = cv2.findContours(square,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # loops through the contours of the smaller box image if their size is within a certain range
        for sub_contour in sub_contours:
            if cv2.contourArea(sub_contour) < 10000 and cv2.contourArea(sub_contour) > 1000:
                # solidity is the ratio of the contour area to the area of the contour's convex hull
                area = cv2.contourArea(sub_contour)
                hull_area = cv2.contourArea(cv2.convexHull(sub_contour))
                solidity = float(area)/hull_area
                print(solidity)
                # O's have a high solidity, and X's have a lower solidity
                if solidity > 0.6:
                    board[board_x][board_y] = "O"
                else:
                    board[board_x][board_y] = "X"

# prints the board state
num_X = 0
num_O = 0
count = 0
print("The board is below!")
for row in board:
    print(row)
    for square in row:
        if square == "X":
            num_X += 1
            count += 1
        elif square == "O":
            num_O += 1
            count += 1

# checks if the game is over
# if the game is not over, the next best move is found based on whose turn it is and this move is shown
ptm = 0
if count == 9:
    print("The game is over!")
    cv2.rectangle(img,(50,250),(525,320),(0,0,0),-1)
    cv2.putText(img,"THE GAME IS OVER",(65,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),5)
    cv2.imshow("THE GAME IS OVER",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif count == 0:
    print("It's X's turn!")
    print("The best move for X is below!")
    cv2.putText(img,"X",(int(0.3*contour_x),int(0.9*contour_y)),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255),10)
    cv2.imshow("THIS IS THE BEST MOVE",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    board[0][0] = "X"
    for row in board:
        print(row)
else:
    if num_X > num_O:
        ptm = 1
        print("It's O's turn!")
        print("The best move for O is below!")
    else:
        print("It's X's turn!")
        print("The best move for X is below!")
    board_to_pass_in = board
    predict_board = TTTProblem(board = board_to_pass_in,player_to_move = ptm)
    move = adversarialsearch.minimax(predict_board)
    if ptm == 0:
        board[move[0]][move[1]] = "X"
        cv2.putText(img,"X",(move[1]*contour_x+int(0.3*contour_x),move[0]*contour_y+int(.9*contour_y)),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255),10)
    elif ptm == 1:
        board[move[0]][move[1]] = "O"
        cv2.putText(img,"O",(move[1]*contour_x+int(0.3*contour_x),move[0]*contour_y+int(.9*contour_y)),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255),10)
    for row in board:
            print(row)
    cv2.imshow("THIS IS THE BEST MOVE",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()