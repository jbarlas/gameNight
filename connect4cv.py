import cv2
import numpy as np
from AdversarialSearch.connect4problem import Connect4Problem, Connect4GUI
from AdversarialSearch import adversarialsearch

def showimg(title, img):
    cv2.imshow(title, img)
    cv2.moveWindow(title, 100, 100)
    cv2.waitKey(0)

# read in image
angle_pieces = "testImages\connect4_angle_w_pieces.jpg"
standard = "testImages\connect4_img_test.jpg"
bright_noise = "testImages\connect4_bright_noise.jpg"
not_great = "testImages\connect4_not_great.jfif"
angle_pieces_unsolve = "testImages\connect4_angle_w_pieces_unsolved.png"
board_image = cv2.imread(angle_pieces)

# process image
new_width = 750
img_height, img_width, _ = board_image.shape
scale = new_width / img_width
img_width = int(img_width * scale)
img_height = int(img_height * scale)
board_image = cv2.resize(board_image, (img_width, img_height), interpolation=cv2.INTER_AREA)
img_orig = board_image.copy()
showimg('original image resized', img_orig)

# apply bilateral filter
bi_filt_img = cv2.bilateralFilter(board_image, 20, 190, 190)
showimg('bilateralFilter', bi_filt_img)

# outline edges
edge_img = cv2.Canny(bi_filt_img, 100, 200)
showimg('edges', edge_img)

# find contours
contour_img,  contours, heirarchies = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

# initialize list of (contours, rectangles, and positions)
cont_list = []

# get all contours
for contour in contours:
    # get length of closed contour
    cont_len = cv2.arcLength(contour,True)
    # approximate polygonal curve
    approx = cv2.approxPolyDP(contour,0.01*cont_len,True)
    # approximate area of the polygon
    area = cv2.contourArea(approx)
    # create bounding rectangle
    rect = cv2.boundingRect(approx)

    # rectangle information 
    x,y,w,h = rect
    rect_area = w*h
    rect_center = (x + w/2, y + h/2)

    cont_list.append((approx, rect, rect_area, rect_center, area))


# filter by circle circle conditions:
#circle_list = list(filter(lambda c: len(c[0]) > 8 \
#    and (int(c[1][2]) in range(int(c[1][3]*.7), int(c[1][3]*1.3)))\
#        and c[4] > (.65 * c[2]), cont_list))

circle_list = cont_list

# only rectangles
rect_list = list(filter(lambda c: len(c[0]) == 4, cont_list))

# guess board rectangle 
board_rect = max(cont_list, key = lambda c: c[2])
bx, by, bw, bh = board_rect[1]
board_area = bw*bh
img_board_circle = img_orig.copy()
cv2.rectangle(img_board_circle,(bx,by),(bx+bw,by+bh),(0,0,255),1)
showimg('identify board', board_image)

# filter circle list 
lower_circle = board_area/(42*4)
upper_circle = board_area/42
# within region 
circle_list = list(filter(lambda c: c[3][0] > bx and c[3][0] < (bx + bw) \
    and c[3][1] > by and c[3][1] < (by + bh), circle_list))
# within size 
circle_list = list(filter(lambda c: c[2] > lower_circle and c[2] < upper_circle \
    and (int(c[1][2]) in range(int(c[1][3]*.7), int(c[1][3]*1.3)))#, cont_list))
        and c[4] > (.5 * c[2]), cont_list))

circle_cont = [c[0] for c in circle_list]
cv2.drawContours(img_board_circle,  circle_cont,  -1, (0,255,0), thickness=1)
for circle in circle_list:
    cx, cy, cw, ch = circle[1]
    cv2.rectangle(img_board_circle, (cx,cy),(cx+cw,cy+ch),(0,0,255),1)
showimg('circles detected', img_board_circle)



# interpolate grid 
grid_overlay = img_orig.copy()
lowest_circle = max(circle_list, key=lambda c: c[3][1])
bottom_row = list(filter(lambda c: c[3][1] > (lowest_circle[3][1] - lowest_circle[1][3]*.75), circle_list))
bottom_cont = [c[0] for c in bottom_row]
rightmost_circle = max(circle_list, key=lambda c: c[3][0])
rightmost_row = list(filter(lambda c: c[3][0] > (rightmost_circle[3][0] - rightmost_circle[1][2]*.75), circle_list))
rightmost_cont = [c[0] for c in rightmost_row]
if len(bottom_cont) < 2 or len(rightmost_cont) < 2:
    print('NEED TWO CIRCLES FROM THE BOTTOM AND RIGHT ROWS')

# find two adjacent circles in the bottom row
bottom_centers = [c[3] for c in bottom_row]
bottom_centers = sorted(bottom_centers, key=lambda c: c[0], reverse=True)
min_width = float('inf')
prev_x = bottom_centers[0][0]
bottom_pair = None
for i in range(1, len(bottom_centers)):
    w = -(bottom_centers[i][0] - prev_x)
    if w < min_width and w > bottom_row[i][1][2]*.5:
        min_width = w
        bottom_pair = (bottom_row[i-1], bottom_row[i])
    prev_x = bottom_centers[i][0]

# find two adjacent circles in rightmost row
right_centers = [c[3] for c in rightmost_row]
right_centers = sorted(right_centers, key=lambda c: c[1], reverse=True)
min_height = float('inf')
prev_y = right_centers[0][1]
right_pair = None
for i in range(1, len(right_centers)):
    h = -(right_centers[i][1] - prev_y)
    if h < min_height and h > rightmost_row[i][1][3]*.5:
        min_height = h
        right_pair = (rightmost_row[i-1], rightmost_row[i])
    prev_y = right_centers[i][1]

# get slope between bottom pieces
bottom_center_pair = [c[3] for c in bottom_pair]
b_dy = int(bottom_center_pair[0][1] - bottom_center_pair[1][1])
b_dx = int(bottom_center_pair[0][0] - bottom_center_pair[1][0])
b_slope = b_dy/b_dx

# get slope between rightmost pieces
right_center_pair = [c[3] for c in right_pair]
r_dy = int(right_center_pair[0][1] - right_center_pair[1][1])
r_dx = int(right_center_pair[0][0] - right_center_pair[1][0])
if r_dx == 0: # if the pieces on the right are perfectly straight
    r_slope = 0
else:
    r_slope = -r_dy/r_dx

# determine intercept between lines (bottom right piece)
bx = bottom_center_pair[0][0]
by = bottom_center_pair[0][1]
rx = right_center_pair[0][0]
ry = right_center_pair[0][1]
x_int = int(((r_slope * rx) - (b_slope * bx) + (ry - by)) / (r_slope - b_slope))
y_int = int(b_slope * (x_int - bx) + by)

cv2.line(grid_overlay, (int(bottom_center_pair[0][0]), int(bottom_center_pair[0][1])), \
    (int(bottom_center_pair[1][0]), int(bottom_center_pair[1][1])), (255, 0, 0), thickness=1)
cv2.line(grid_overlay, (int(right_center_pair[0][0]), int(right_center_pair[0][1])), \
    (int(right_center_pair[1][0]), int(right_center_pair[1][1])), (255, 0, 0), thickness=1)

cv2.circle(grid_overlay, (x_int, y_int), 26, (0, 0, 0), thickness=6)

# populate the rest of the grid
grid_centers = np.zeros((6,  7, 2))
row_start_y = y_int
row_start_x = x_int
for r in range(6):
    for c in range(7):
        grid_centers[r][c][0] = row_start_x + (b_dx * c*1.01) - np.exp(np.sqrt(c+r))
        grid_centers[r][c][1] = row_start_y + (b_dy * c*1.01)
        cv2.circle(grid_overlay, (int(grid_centers[r][c][0]), int(grid_centers[r][c][1])), 20, (0, 0, 255), thickness=2)
    row_start_y -= r_dy * np.exp(.05 * r)
    row_start_x -= r_dx * np.exp(.05 * r)


bottom_cont += rightmost_cont
for_line = [c[0] for c in bottom_pair] + [c[0] for c in right_pair] 
cv2.drawContours(grid_overlay,  for_line,  -1, (0,255,0), thickness=1)

showimg('grid overlay', grid_overlay)



# filter by color
img_hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV) 

lower_red = np.array([150, 150, 100])  
upper_red = np.array([255, 255, 255])  
mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
img_red = cv2.bitwise_and(img_orig, img_orig, mask=mask_red)
showimg("Red Mask",img_red)

lower_yellow = np.array([10, 150, 100])
upper_yellow = np.array([60, 255, 255])
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
img_yellow = cv2.bitwise_and(img_orig, img_orig, mask=mask_yellow)
showimg("Yellow Mask",img_yellow)

rows = 6
cols = 7

# identify piece color
grid = np.zeros((rows,cols))
id_red = 1
id_yellow = 2
img_grid_overlay = img_orig.copy()
img_grid = np.zeros([img_height,img_width,3], dtype=np.uint8)

for c in range(cols):
    for r in range(rows):
        x = int(grid_centers[r][c][0])
        y = int(grid_centers[r][c][1])
        rad = 20
        img_grid_circle = np.zeros((img_height, img_width))
        cv2.circle(img_grid_circle, (x,y), rad, (255,255,255),thickness=-1)
        img_res_red = cv2.bitwise_and(img_grid_circle, img_grid_circle, mask=mask_red)
        img_grid_circle = np.zeros((img_height, img_width))
        cv2.circle(img_grid_circle, (x,y), rad, (255,255,255),thickness=-1)
        img_res_yellow = cv2.bitwise_and(img_grid_circle, img_grid_circle,mask=mask_yellow)
        cv2.circle(img_grid_overlay, (x,y), rad, (0,255,0),thickness=1)
        if img_res_red.any() != 0:
            grid[r][c] = id_red
            cv2.circle(img_grid, (x,y), rad, (0,0,255),thickness=-1)
        elif img_res_yellow.any() != 0 :
            grid[r][c] = id_yellow
            cv2.circle(img_grid, (x,y), rad, (0,255,255),thickness=-1)
grid_to_print = np.flipud(grid)
for i in range(len(grid)):
    grid_to_print[i] = np.flipud(grid_to_print[i])

print('Grid Detected:\n', grid_to_print)
showimg('Img Grid Overlay',img_grid_overlay)
showimg('Img Grid',img_grid)

# determine best move
num_red = sum([np.count_nonzero(row == 1) for row in grid])
num_yellow = sum([np.count_nonzero(row == 2) for row in grid])

if not any([0 in row for row in grid]):
    grid_full = True
    print("Grid Full")

elif num_red < num_yellow:
    board = Connect4Problem(board=grid)
    move = (adversarialsearch.alpha_beta_cutoff(board, 5, board.heuristic_func), id_red)
    print(move)
    #move = (ConnectFourAI.bestMove(grid*(-1), 1, -1), id_yellow)
    print("Red To Move: Column {}".format(move[0]+1))
    
else:
    board = Connect4Problem(board=grid)
    move = (adversarialsearch.alpha_beta_cutoff(board, 5, board.heuristic_func), id_yellow)
    print(move)
    #move = (ConnectFourAI.bestMove(grid, 1, -1), id_red)
    print("Yellow To Move: Column {}".format(move[0]+1))
    

# display image with output
if any([0 in row for row in grid]):
    img_output = img_orig.copy()
    empty_slots = sum([ row[move[0]] == 0 for row in grid ])
    col = move[0]
    row = np.where(grid[:, col]==0)[0][0]
    (x, y) = (int(grid_centers[row][col][0]), int(grid_centers[row][col][1]))
    if move[1] == id_red:
        cv2.circle(img_output, (x+6,y+3), 20, (0,0,255),thickness=-1)
    if move[1] == id_yellow:
        cv2.circle(img_output, (x+6,y+3), r+4, (0,255,255),thickness=-1)
    cv2.circle(img_output, (x+6,y+3), 22, (0, 0, 0),thickness=2)

    showimg("output",img_output)


cv2.destroyAllWindows()






