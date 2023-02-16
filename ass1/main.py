import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import os

CELL_LENGTH = 21.34
ROWS = 9
COLS = 6
CAMERA_CONFIG_PATH = "camera_config_all.npz"
TRAINING_IMAGES_AMNT = 0 # Set to 0 to use all images that are found in IMAGE_PATH, set to a number to use that many images
IMAGE_PATH = "ass1/images/"
SHOW_IMAGES = False # Set True to show the images of the found corners during training, False to not show images
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FRAME_RATE = 24 # set frame rate of the webcam

VIDEO = False # Set True to use webcam, False to use images


OBJP = np.zeros((COLS * ROWS, 3), np.float32)
OBJP[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1,2) * CELL_LENGTH

class Color:
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)

    
def get_corners_manually(img):
    print("Click on the four corners of the chessboard. When done, press any key to continue.")
    manual_corners = []
    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            manual_corners.append([x, y])
            font = cv.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ', ' + str(y)
            cv.putText(visual_image, strXY, (x, y), font, .5, (255, 255, 0), 2)
            cv.imshow('Manual corners', visual_image)
    #In case the text  and text-color messes with the corner interpolation
    visual_image = img.copy()
    cv.imshow('Manual corners', visual_image)
    cv.setMouseCallback('Manual corners', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    if len(manual_corners) != 4:
        print("The number of corners is not 4. Please try again.")
        return get_corners_manually(img)
    corners = interpolate_chessboard_corners(manual_corners, ROWS, COLS)
    cv.destroyAllWindows()
    return corners

#OLD - NOT FOR USE
# #Warp image dimensions to fit the screen in correct dimension
# def warp_image_interpolate_chessboard_corners(corners, rows, cols, img):
#     #Four points map to calculate width and heigth of the image
#     tl, tr, br, bl = corners
#     rect = np.zeros((4, 2), dtype = "float32")
#     rect[0] = tl
#     rect[1] = tr
#     rect[2] = br
#     rect[3] = bl
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))

#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))

#     #mapping
#     dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")
#     #Compute perspective transform
#     M = cv.getPerspectiveTransform(rect, dst)
#     #apply pespective transform to the entire image to warp it
#     warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
#     cv.imshow('original', img)
#     cv.imshow('warped', warped)
#     cv.waitKey(0)

#     return warped

def save_camera_config(mtx, dist, rvecs, tvecs, optimal_camera_matrix, file=CAMERA_CONFIG_PATH):
    np.savez(file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, optimal_camera_matrix=optimal_camera_matrix)
    

def load_camera_config(file=CAMERA_CONFIG_PATH):
    with np.load(file) as X:
        mtx, dist, rvecs, tvecs, optimal_camera_matrix= [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs', 'optimal_camera_matrix')]
    return mtx, dist, rvecs, tvecs, optimal_camera_matrix

def draw_world_axis(img, rVecs, tVecs, cameraMatrix, d, size=1):
    #Create array that will hold four 3D points
    points = np.float32([[0,0,0],[size * CELL_LENGTH,0,0],[0,size * CELL_LENGTH,0],[0,0,-size * CELL_LENGTH]])
                                                                                                                                        #^might have to be negative
    #Project 3D points to 2D image
    imgpts, jac = cv.projectPoints(points, rVecs, tVecs, cameraMatrix, d)
    point_one = tuple(map(int, imgpts[0].ravel()))
    point_two = tuple(map(int, imgpts[1].ravel()))
    point_three = tuple(map(int, imgpts[2].ravel()))
    point_four = tuple(map(int, imgpts[3].ravel()))
    
	#Draws XYZ lines in different colors and thickness of lines
    cv.line(img, point_one, point_two, Color.RED, 3)
    cv.line(img, point_one, point_three, Color.GREEN, 3)
    cv.line(img, point_one, point_four, Color.BLUE, 3)
    return img

def draw_cube(img, rVecs, tVecs, cameraMatrix, d, size=1):
    # todo: image 29, the cube doesn't follow the z-axis and is slightly off
    points = np.float32([[0,0,0], [0,size*CELL_LENGTH,0], [size*CELL_LENGTH,size*CELL_LENGTH,0], [size*CELL_LENGTH,0,0], #Bottom 4 points 
    [0,0,-size*CELL_LENGTH],[0,size*CELL_LENGTH,-size*CELL_LENGTH],[size*CELL_LENGTH,size*CELL_LENGTH,-size*CELL_LENGTH],[size*CELL_LENGTH,0,-size*CELL_LENGTH] ]) #Top 4 points -> Z might  have to be negative
    #3D points projected to 2D coordinates
    imgpts, jac = cv.projectPoints(points, rVecs, tVecs, cameraMatrix, d)

    points = []
    for i in range(0, 8):
        points.append(tuple(map(int, imgpts[i].ravel())))


    colors = {
        "bottom": Color.CYAN,
        "top": Color.CYAN,
        "sides": Color.CYAN
    }

    thiccness = 3

    groups = [
        {"points": points[:4], "color": colors["bottom"]},
        {"points": [points[0], points[4]], "color": colors["sides"]},
        {"points": [points[1], points[5]], "color": colors["sides"]},
        {"points": [points[2], points[6]], "color": colors["sides"]},
        {"points": [points[3], points[7]], "color": colors["sides"]},
        {"points": points[4:8], "color": colors["top"]},
    ]

    for group in groups:
        for j in range(len(group["points"])):
            cv.line(img, group["points"][j], group["points"][(j+1)%len(group["points"])], group["color"], thiccness)

    return img


#Manually generate chessboard points using user input corners
def interpolate_chessboard_corners(corners, rows, cols):
    #Defining corner order
    tl, tr, br, bl = corners
    #Transforming the corner array into the type the function getPerspective has
    points = np.zeros((4, 2), dtype = "float32")
    points[0] = tl
    points[1] = tr
    points[2] = br
    points[3] = bl

    #Calculating width and heigth of the image inside the boundary of the 4 corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    #mapping 4 new corners from the new heigth and width
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    
    #linearly expanding all the points
    x_coord = np.linspace(dst[0][0], dst[1][0], rows)
    y_coord = np.linspace(dst[0][1], dst[3][1], cols)

    #making a holder array that holds all points
    rect = np.zeros((rows*cols, 2), dtype = "float32")
    counter = 0
    for j in range(cols):
        for k in range(rows):
            rect[counter] = [x_coord[k],y_coord[j]]
            counter += 1
    #Compute perspective transform from unwarped points to warped points
    M = cv.getPerspectiveTransform(dst, points)

    # Apply the transformation matrix to the array of all points
    transformed_points = cv.perspectiveTransform(rect.reshape(-1,1,2), M)
    return transformed_points


def preprocess_image(img):
    #todo: play around with these parameters
    # not sure if this is even necessary...
    # although I just found an image where this function makes a difference
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    # sharpen the edges of the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv.filter2D(img, -1, kernel)
    # cv.imshow("preprocessed", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def train_camera():
    # Arrays to store object points and image points from all the images.
    object_points = [] # 3d point in real world space
    image_points = [] # 2d points in image plane.
    # read images from folder relative to the main.py file
    images = glob.glob(IMAGE_PATH + '*.jpg')
    if len(images) == 0:
        print("No images found in " + IMAGE_PATH)
        exit()
    for i, fname in enumerate(images):
        if TRAINING_IMAGES_AMNT != 0 and i >= TRAINING_IMAGES_AMNT:
            break
        print(f"Processing image {i}...")
        img = cv.imread(fname)
        ret, corners, gray = find_chessboard_corners(img, CRITERIA)
        object_points.append(OBJP)
        image_points.append(corners)
    return object_points, image_points, gray


def find_chessboard_corners(img, camera=False):
    gray = preprocess_image(img)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (ROWS, COLS), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), CRITERIA)
        if SHOW_IMAGES:
            cv.drawChessboardCorners(img, (ROWS, COLS), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return True, corners2, gray
    elif not camera:
        manual_corners = get_corners_manually(img)
        corners2 = cv.cornerSubPix(gray, manual_corners, (11,11), (-1,-1), CRITERIA)
        if SHOW_IMAGES:
            # Draw and display the corners
            cv.drawChessboardCorners(img, (ROWS, COLS), corners2, True)
            cv.imshow('img', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return False, corners2, gray


def calibrate_camera():
    # check if camera calibration file exists
    if not os.path.isfile(CAMERA_CONFIG_PATH):
        print('Calibrating camera...')
        objpoints, imgpoints, gray = train_camera()
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = gray.shape[:2]
        optimal_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        save_camera_config(mtx, dist, rvecs, tvecs, optimal_camera_matrix)
        return mtx, dist, rvecs, tvecs, optimal_camera_matrix
    else:
        return load_camera_config()

def video(mtx, dist, rvecs, tvecs, optimal_camera_matrix):
    # Inspired from here https://github.com/pavithran-s/Camera_Calibaration/blob/master/draw_cube.ipynb
    print("Opening webcam...")
    webcam = cv.VideoCapture(1)
    if webcam.isOpened(): 
        print("Webcam opened")
    else:
        print("Unable to read camera feed")
    width = int(webcam.get(3))
    height = int(webcam.get(4))
    out = cv.VideoWriter('output.mp4',cv.VideoWriter_fourcc('m','p','4','v'), FRAME_RATE, (width,height))
    while True:
        has_frame, frame = webcam.read()
        if has_frame == False:
            break
        # grayscale image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray,(ROWS, COLS), None)
        # ret, corners, gray = find_chessboard_corners(frame)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), CRITERIA)
            ret, rvecs, tvecs = cv.solvePnP(OBJP, corners2, mtx, dist)
            frame = draw_world_axis(frame, rvecs, tvecs, mtx, dist, size=5)
            frame = draw_cube(frame, rvecs, tvecs, mtx, dist, size=3)
        cv.imshow('images',frame)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break
    # write the output video to file
    out.release()
    cv.destroyAllWindows()
    webcam.release()


def image(mtx, dist, rvecs, tvecs, optimal_camera_matrix):
    images = glob.glob(IMAGE_PATH + '*.jpg')
    for fname in images:
        img = cv.imread(fname)
        ret, corners, gray = find_chessboard_corners(img)
        _, rvecs, tvecs = cv.solvePnP(OBJP, corners, mtx, dist)
        img = draw_world_axis(img, rvecs, tvecs, mtx, dist, size=5)
        img = draw_cube(img, rvecs, tvecs, mtx, dist, size=3)
        cv.imshow(f'cube {fname}', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    # test_image_number = 22
    # image_name = IMAGE_PATH + str(test_image_number).zfill(2) + '.jpg'
    # img = cv.imread(image_name)
    # _, corners, _ = find_chessboard_corners(img)
    # _, rvecs, tvecs = cv.solvePnP(OBJP, corners, mtx, dist)
    # img = draw_world_axis(img, rvecs, tvecs, mtx, dist, size=5)
    # img = draw_cube(img, rvecs, tvecs, mtx, dist, size=3)
    # cv.imshow('cube', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def main():
    mtx, dist, rvecs, tvecs, optimal_camera_matrix = calibrate_camera()
    if VIDEO:
        video(mtx, dist, rvecs, tvecs, optimal_camera_matrix)
    else:
        image(mtx, dist, rvecs, tvecs, optimal_camera_matrix)

if __name__ == '__main__':
    main()