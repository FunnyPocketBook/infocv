import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import os

CELL_LENGTH = 21.34
ROWS = 9
COLS = 6
CAMERA_CONFIG_PATH = "camera_config_all.npz"
TRAINING_IMAGES_AMNT = 0
IMAGE_PATH = "ass1/images/"


objp = np.zeros((COLS * ROWS, 3), np.float32)
objp[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1,2) * CELL_LENGTH

manual_corners = []

    
def get_corners_manually(img):
    print("Click on the four corners of the chessboard in the following order: top left, top right, bottom right, bottom left. When done, press any key to continue.")
    global manual_corners, ROWS, COLS
    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            manual_corners.append([x, y])
            font = cv.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ', ' + str(y)
            cv.putText(visual_image, strXY, (x, y), font, .5, (255, 255, 0), 2)
            cv.imshow('image', visual_image)
    #In case the text  and text-color messes with the corner interpolation
    visual_image = img.copy()
    cv.imshow('image', visual_image)
    cv.setMouseCallback('image', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('image', visual_image)
    if len(manual_corners) != 4:
        print("The number of corners is not 4. Please try again.")
        manual_corners = []
        return get_corners_manually(img)
    cv.destroyAllWindows()
    corners = interpolate_chessboard_corners(manual_corners, ROWS, COLS)
    #print(corners)
    manual_corners = []
    # draw the corners on the image
    #for corner in corners:
    #   cv.circle(img, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)

    #cv.imshow("Image", img)
    cv.waitKey(0)
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

def save_camera_config(mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints, file=CAMERA_CONFIG_PATH):
    np.savez(file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, optimal_camera_matrix=optimal_camera_matrix, objpoints=objpoints, imgpoints=imgpoints)
    

def load_camera_config(file=CAMERA_CONFIG_PATH):
    with np.load(file) as X:
        mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs', 'optimal_camera_matrix', 'objpoints', 'imgpoints')]
    return mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints

def draw_world_axis(img, rVecs, tVecs, cameraMatrix, d, size=1):
    img = img.copy()
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
    cv.line(img, point_one, point_two, (255, 0, 0), 3)
    cv.line(img, point_one, point_three, (0, 255, 0), 3)
    cv.line(img, point_one, point_four, (0, 0, 255), 3)
    return img

def draw_cube(img, rVecs, tVecs, cameraMatrix, d, size=1):
    img = img.copy()
    points = np.float32([[0,0,0], [0,size*CELL_LENGTH,0], [size*CELL_LENGTH,size*CELL_LENGTH,0], [size*CELL_LENGTH,0,0], #Bottom 4 points 
    [0,0,-size*CELL_LENGTH],[0,size*CELL_LENGTH,-size*CELL_LENGTH],[size*CELL_LENGTH,size*CELL_LENGTH,-size*CELL_LENGTH],[size*CELL_LENGTH,0,-size*CELL_LENGTH] ]) #Top 4 points -> Z might  have to be negative
    #3D points projected to 2D coordinates
    imgpts, jac = cv.projectPoints(points, rVecs, tVecs, cameraMatrix, d)

    points = []
    for i in range(0, 8):
        points.append(tuple(map(int, imgpts[i].ravel())))

    colors = {
        "bottom": (255, 0, 0),
        "top": (0, 255, 0),
        "sides": (0, 0, 255)
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


def train_camera(show_images=False):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Arrays to store object points and image points from all the images.
    object_points = [] # 3d point in real world space
    image_points = [] # 2d points in image plane.
    # read images from folder relative to the main.py file
    images = glob.glob(IMAGE_PATH + '*.jpg')
    print(len(images))
    for i, fname in enumerate(images):
        if TRAINING_IMAGES_AMNT != 0 and i >= TRAINING_IMAGES_AMNT:
            break
        print(i)
        img = cv.imread(fname)
        objp, corners, gray = find_chessboard_corners(img, criteria, show_images)
        object_points.append(objp)
        image_points.append(corners)
    return object_points, image_points, gray


def find_chessboard_corners(img, criteria, show_images=False):
    gray = preprocess_image(img)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (ROWS, COLS), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        if show_images:
            cv.drawChessboardCorners(img, (ROWS, COLS), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
    else:
        manual_corners = get_corners_manually(img)
        corners2 = cv.cornerSubPix(gray, manual_corners, (11,11), (-1,-1), criteria)
        if show_images:
            # Draw and display the corners
            cv.drawChessboardCorners(img, (ROWS, COLS), corners2, True)
            cv.imshow('img', img)
            cv.waitKey(0)
    cv.destroyAllWindows()
    return objp, corners2, gray


def calibrate_camera(show_images=False):
    # check if camera calibration file exists
    if not os.path.isfile(CAMERA_CONFIG_PATH):
        print('Calibrating camera...')
        objpoints, imgpoints, gray = train_camera(show_images)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = gray.shape[:2]
        optimal_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # save mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints and imgpoints to file
        save_camera_config(mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints)
        return mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints
    else:
        # load mtx, dist, rvecs, tvecs, objpoints and imgpoints from file
        return load_camera_config()

def video():
    mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints = calibrate_camera(show_images=False)
    print("Opening webcam...")
    webcam = cv.VideoCapture(0)
    if webcam.isOpened(): 
        print("Webcam opened")
    else:
        print("Unable to read camera feed")
    out = cv.VideoWriter('output.mp4',cv.VideoWriter_fourcc('M','J','P','G'), 10, (int(webcam.get(3)),int(webcam.get(4))))
    while True:
        has_frame, frame = webcam.read()
        if has_frame == False:
            break
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray,(ROWS, COLS), None)
        if ret == True:
            ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)
            frame = draw_cube(frame, rvecs, tvecs, mtx, dist, 2)
        cv.imshow('images',frame)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break
    # write the output video to file
    out.release()
    cv.destroyAllWindows()
    webcam.release()


VIDEO = False


def main():
    if VIDEO:
        video()
        return
    else:
        mtx, dist, rvecs, tvecs, optimal_camera_matrix, objpoints, imgpoints = calibrate_camera(show_images=True)
        test_image_number = 2
        image_name = IMAGE_PATH + str(test_image_number).zfill(2) + '.jpg'
        img = cv.imread(image_name)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp, imgp, gray = find_chessboard_corners(img, criteria, show_images=False)
        # h,  w = img.shape[:2]
        # undistort
        # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, optimal_camera_matrix, (w,h), 5)
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        ret, rvecs, tvecs = cv.solvePnP(objp, imgp, mtx, dist)
        size = 3
        # cv.imshow('world_axis', draw_world_axis(img, rvecs, tvecs, mtx, dist, size))
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # Draw a cube on the image
        cv.imshow('cube', draw_cube(img, rvecs, tvecs, mtx, dist, size))
        cv.waitKey(0)
        cv.destroyAllWindows()
    



if __name__ == '__main__':
    main()