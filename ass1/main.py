import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

CELL_LENGTH = 21.34
ROWS = 9
COLS = 6

manual_corners = []

def click_event(event, x, y, flags, params):
    global manual_corners
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        manual_corners.append([x, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        cv.imshow('image', img)

    
def get_corners_manually(img):
    print("Click on the four corners of the chessboard in the following order: top left, top right, bottom right, bottom left. When done, press any key to continue.")
    global manual_corners
    cv.imshow('image', img)
    cv.setMouseCallback('image', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('image', img)
    if len(manual_corners) != 4:
        print("The number of corners is not 4. Please try again.")
        manual_corners = []
        return get_corners_manually(img)
    rows = int(input("Enter the number of rows of the chessboard seen: "))
    cols = int(input("Enter the number of columns of the chessboard seen: "))
    cv.waitKey(0)
    cv.destroyAllWindows()
    corners = interpolate_chessboard_corners(manual_corners, rows, cols, img)
    #print(corners)
    manual_corners = []
    # draw the corners on the image
    #for corner in corners:
    #   cv.circle(img, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)

    #cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return corners


def interpolate_chessboard_corners(corners, rows, cols, img):
    # interpolate the corners of the chessboard
    # corners: list of 4 corners of the chessboard
    # rows: number of rows of the chessboard
    # cols: number of columns of the chessboard
    # returns: list of corners of the chessboard

    tl, tr, br, bl = corners
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
    cv.imshow('original', img)
    cv.imshow('warped', warped)
    cv.waitKey(0)
    # get the coordinates of the four corners in the format [x, y] and dtype float32
    #p1 = np.array(corners[0], dtype=np.float32)
    #p2 = np.array(corners[1], dtype=np.float32)
    #p3 = np.array(corners[2], dtype=np.float32)
    #p4 = np.array(corners[3], dtype=np.float32)

    # Create a matrix of points representing the chessboard
    #points = np.zeros((rows * cols, 2), dtype=np.float32)
    #for r in range(rows):
    #    for c in range(cols):
    #        points[r * cols + c] = [c * 100, r * 100]

    # Calculate the perspective transformation matrix
    #transform_matrix = cv.getPerspectiveTransform(np.array([p1, p2, p3, p4]), points)

    # Apply the transformation matrix to the matrix of points
    #transformed_points = cv.perspectiveTransform(points.reshape(1, -1, 2), transform_matrix)

    # Reshape the matrix of points to the correct format
    #transformed_points = transformed_points.reshape(-1, 2)

    return warped

    # linearly interpolate the corners of the chessboard
    # interpolate the top row
    # top_row = []
    # for i in range(cols):
    #     top_row.append([top_left[0] + i * (top_right[0] - top_left[0]) / (cols - 1), top_left[1] + i * (top_right[1] - top_left[1]) / (cols - 1)])
    # # interpolate the bottom row
    # bottom_row = []
    # for i in range(cols):
    #     bottom_row.append([bottom_left[0] + i * (bottom_right[0] - bottom_left[0]) / (cols - 1), bottom_left[1] + i * (bottom_right[1] - bottom_left[1]) / (cols - 1)])
    # # interpolate the left column
    # left_column = []
    # for i in range(rows):
    #     left_column.append([top_left[0] + i * (bottom_left[0] - top_left[0]) / (rows - 1), top_left[1] + i * (bottom_left[1] - top_left[1]) / (rows - 1)])
    # # interpolate the right column
    # right_column = []
    # for i in range(rows):
    #     right_column.append([top_right[0] + i * (bottom_right[0] - top_right[0]) / (rows - 1), top_right[1] + i * (bottom_right[1] - top_right[1]) / (rows - 1)])

    # # interpolate the corners of the chessboard and take the right and left column and top and bottom row into account
    # corners = []
    # for i in range(rows):
    #     for j in range(cols):
    #         corners.append([top_row[j][0] + i * (bottom_row[j][0] - top_row[j][0]) / (rows - 1), top_row[j][1] + i * (bottom_row[j][1] - top_row[j][1]) / (rows - 1)])

    # return corners



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((COLS * ROWS, 3), np.float32)
objp[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1,2) * CELL_LENGTH
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# read images from folder relative to the main.py file
images = glob.glob('C:/Users/nikol/OneDrive/Documents/School/Computer Vision/2023/Assignments/Assignment 1/infocv/ass1/images/webcam/*.jpg')
print(len(images))
img_counter = 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (ROWS, COLS), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (ROWS, COLS), corners2, ret)
        plt.figure(figsize = (20,15))
        # plt.imshow(img)
        # plt.show()
        # cv.imshow('img', img)
        # cv.waitKey(0)
    else:
        img_warped = get_corners_manually(img)
        gray = cv.cvtColor(img_warped, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (ROWS, COLS), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img_warped, (ROWS, COLS), corners2, ret)
            plt.figure(figsize = (20,15))
            cv.imshow('img', img_warped)
    img_counter += 1
    print(img_counter)
    



# cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv.imread('ass1/images/webcam/00.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)



