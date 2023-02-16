import numpy as np
import cv2 as cv
import glob
import os
from config import Config

config = Config()

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
COLS = config.get("chessboard", "cols")
ROWS = config.get("chessboard", "rows")
CELL_LENGTH = config.get("chessboard", "cell_length")
CAMERA_CONFIG_PATH = config.get("calibration", "calibration_file")
IMAGE_PATH = config.get("calibration", "image_path")
TRAINING_IMAGES_AMNT = config.get("calibration", "training_images_amnt")
USE_WEBCAM = config.get("webcam")


OBJP = np.zeros((COLS * ROWS, 3), np.float32)
OBJP[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1,2) * CELL_LENGTH

class Color:
    """Color constants in BGR format"""
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)


def get_corners_manually(img: np.ndarray) -> np.ndarray:
    """Get the corners of the chessboard manually by clicking on the corners of the chessboard.
    
    Parameters
    ----------
    img : np.ndarray
        The image of the chessboard.

    Returns
    -------
    corners : np.ndarray
        The corners of the chessboard.
    """
    print("Click on the four corners of the chessboard. When done, press any key to continue.")
    manual_corners = []
    def click_event(event, x, y, flags, params):
        """Callback function for the mouse click event. Adds the clicked point to the list of corners and displays the point on the image."""
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            manual_corners.append([x, y])
            font = cv.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ', ' + str(y)
            cv.putText(visual_image, strXY, (x, y), font, .5, (255, 255, 0), 2)
            cv.imshow('Manual corners', visual_image)
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


def save_camera_config(mtx, dist, rvecs, tvecs, optimal_camera_matrix, file=CAMERA_CONFIG_PATH):
    """Save the camera configuration to a file.
    
    Parameters
    ----------
    mtx : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    rvecs : np.ndarray
        The rotation vectors.
    tvecs : np.ndarray
        The translation vectors.
    optimal_camera_matrix : np.ndarray
        The optimal camera matrix.
    file : str, optional
        The path to the file, by default CAMERA_CONFIG_PATH (set in config.yml)
    """
    np.savez(file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, optimal_camera_matrix=optimal_camera_matrix)
    

def load_camera_config(file=CAMERA_CONFIG_PATH):
    """Load the camera configuration from a file.

    Parameters
    ----------
    file : str, optional
        The path to the file, by default CAMERA_CONFIG_PATH (set in config.yml)

    Returns
    -------
    mtx : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    rvecs : np.ndarray
        The rotation vectors.
    tvecs : np.ndarray
        The translation vectors.
    opt_cam_mtx : np.ndarray
        The optimal camera matrix.
    """
    with np.load(file) as X:
        mtx, dist, rvecs, tvecs, optimal_camera_matrix= [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs', 'optimal_camera_matrix')]
    return mtx, dist, rvecs, tvecs, optimal_camera_matrix


def draw_world_axis(img, rVecs, tVecs, cameraMatrix, dist, size=1):
    """Draws the world axis on the image.
    
    Parameters
    ----------
    img : np.ndarray
        The image to draw the axis on.
    rVecs : np.ndarray
        The rotation vectors.
    tVecs : np.ndarray
        The translation vectors.
    cameraMatrix : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    size : int, optional
        The size of the axis, by default 1
        
    Returns
    -------
    img : np.ndarray
        The image with the axis drawn on it.
    """
    points = np.float32([[0,0,0],[size * CELL_LENGTH,0,0],[0,size * CELL_LENGTH,0],[0,0,-size * CELL_LENGTH]])
    imgpts, jac = cv.projectPoints(points, rVecs, tVecs, cameraMatrix, dist)
    point_one = tuple(map(int, imgpts[0].ravel()))
    point_two = tuple(map(int, imgpts[1].ravel()))
    point_three = tuple(map(int, imgpts[2].ravel()))
    point_four = tuple(map(int, imgpts[3].ravel()))
    
    cv.line(img, point_one, point_two, Color.RED, 3)
    cv.line(img, point_one, point_three, Color.GREEN, 3)
    cv.line(img, point_one, point_four, Color.BLUE, 3)
    return img


def draw_cube(img, rvecs, tvecs, cam_mtx, dist, size=1):
    """Draws a cube on the image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw the cube on.
    rvecs : np.ndarray
        The rotation vectors.
    tvecs : np.ndarray
        The translation vectors.
    cam_mtx : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    size : int, optional
        The size of the cube, by default 1

    Returns
    -------
    img : np.ndarray
        The image with the cube drawn on it.
    """
    points = np.float32([[0,0,0], [0,size*CELL_LENGTH,0], [size*CELL_LENGTH,size*CELL_LENGTH,0], [size*CELL_LENGTH,0,0], #Bottom 4 points 
    [0,0,-size*CELL_LENGTH],[0,size*CELL_LENGTH,-size*CELL_LENGTH],[size*CELL_LENGTH,size*CELL_LENGTH,-size*CELL_LENGTH],[size*CELL_LENGTH,0,-size*CELL_LENGTH] ]) #Top 4 points -> Z might  have to be negative
    #3D points projected to 2D coordinates
    points, jac = cv.projectPoints(points, rvecs, tvecs, cam_mtx, dist)

    points = [tuple(map(int, points[i].ravel())) for i in range(0, 8)]
    colors = {
        "bottom": Color.CYAN,
        "top": Color.CYAN,
        "sides": Color.CYAN
    }
    thiccness = 1
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


def interpolate_chessboard_corners(corners, rows, cols):
    """Interpolates the chessboard corners.

    Parameters
    ----------
    corners : np.ndarray
        The corners of the chessboard.
    rows : int
        The number of rows in the chessboard.
    cols : int
        The number of columns in the chessboard.

    Returns
    -------
    np.ndarray
        The interpolated corners.
    """
    tl, tr, br, bl = [np.array(corner) for corner in corners] # top left, top right, bottom right, bottom left
    points = np.array([tl, tr, br, bl], dtype = "float32")
    
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    #mapping 4 new corners from the new heigth and width
    dst = np.array([
		[0, 0],
		[max_width - 1, 0],
		[max_width - 1, max_height - 1],
		[0, max_height - 1]
        ], dtype = "float32")
    
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
    """Preprocesses the image. This includes converting it to grayscale, thresholding it, dilating it and sharpening the edges.

    Parameters
    ----------
    img : np.ndarray
        The image to preprocess.

    Returns
    -------
    img : np.ndarray
        The preprocessed image.
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    # sharpen the edges of the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv.filter2D(img, -1, kernel)
    if config.get("verbose"):
        cv.imshow("preprocessed", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return img


def train_camera():
    """Trains the camera using the images in the IMAGE_PATH folder.
    
    Returns
    -------
    object_points : list
        The object points in real world space.
    image_points : list
        The image points in image space.
    shape : tuple
        The shape of the images.
    """
    object_points = []
    image_points = []
    images = glob.glob(IMAGE_PATH + '*.jpg')
    if len(images) == 0:
        print(f"No images found in '{os.path.abspath(IMAGE_PATH)}'! Exiting...")
        exit()
    for i, fname in enumerate(images):
        if TRAINING_IMAGES_AMNT != 0 and i >= TRAINING_IMAGES_AMNT:
            break
        print(f"Processing image {fname}...")
        img = cv.imread(fname)
        ret, corners, gray = find_chessboard_corners(img)
        object_points.append(OBJP)
        image_points.append(corners)
    return object_points, image_points, gray.shape


def find_chessboard_corners(img, camera=False, preprocess=False):
    """Finds the chessboard corners in the image.
    First, it tries to find the corners automatically. If that fails, an image preprocessing step is applied and attempts to find the corners automatically again. If that fails, it asks the user to manually select the corners.


    Parameters
    ----------
    img : np.ndarray
        The image to find the corners in.
    camera : bool, optional
        Whether the image is from the camera or not, by default False. If it is from the camera, it will not ask the user to manually select the corners.

    Returns
    -------
    ret : bool
        Whether the corners were found or not.
    corners : np.ndarray
        The corners of the chessboard.
    gray : np.ndarray
        The grayscale image.
    """
    if preprocess:
        gray = preprocess_image(img)
    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (ROWS, COLS), None)
    if ret == True:
        corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), CRITERIA)
        if config.get("verbose"):
            cv.drawChessboardCorners(img, (ROWS, COLS), corners, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return True, corners, gray
    elif not camera and not preprocess:
        return find_chessboard_corners(img, camera, preprocess=True)
    elif not camera:
        manual_corners = get_corners_manually(img)
        corners = cv.cornerSubPix(gray, manual_corners, (11,11), (-1,-1), CRITERIA)
        if config.get("verbose"):
            cv.drawChessboardCorners(img, (ROWS, COLS), corners, True)
            cv.imshow('img', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return True, corners, gray
    return False, corners, gray


def calibrate_camera():
    """Calibrates the camera and saves the calibration data to a file. If the file already exists, it will load the calibration data from the file.
    
    Returns
    -------
    mtx : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    rvecs : np.ndarray
        The rotation vectors.
    tvecs : np.ndarray
        The translation vectors.
    optimal_camera_matrix : np.ndarray
        The optimal camera matrix.
    """
    if not os.path.isfile(CAMERA_CONFIG_PATH):
        print('Calibrating camera...')
        objpoints, imgpoints, img_shape = train_camera()
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)
        h,  w = img_shape[:2]
        optimal_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        save_camera_config(mtx, dist, rvecs, tvecs, optimal_camera_matrix)
        return mtx, dist, rvecs, tvecs, optimal_camera_matrix
    else:
        return load_camera_config()


def video(mtx, dist, rvecs, tvecs, optimal_camera_matrix):
    """Opens the webcam and draws the cube on the chessboard in the video feed. It also saves the video to a file.
    
    Parameters
    ----------
    mtx : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    rvecs : np.ndarray
        The rotation vectors.
    tvecs : np.ndarray
        The translation vectors.
    optimal_camera_matrix : np.ndarray
        The optimal camera matrix.
    """
    # Inspired from here https://github.com/pavithran-s/Camera_Calibaration/blob/master/draw_cube.ipynb
    print("Opening webcam...")
    webcam = cv.VideoCapture(0, cv.CAP_DSHOW)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    if webcam.isOpened(): 
        print("Webcam opened")
    else:
        print("Unable to read camera feed. Exiting...")
        exit()
    width = int(webcam.get(3))
    height = int(webcam.get(4))
    # write video file as mp4
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, 5, (width, height))
    while True:
        has_frame, frame = webcam.read()
        if has_frame == False:
            break
        ret, corners, gray = find_chessboard_corners(frame, camera=True)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), CRITERIA)
            ret, rvecs, tvecs = cv.solvePnP(OBJP, corners2, mtx, dist)
            frame = draw_world_axis(frame, rvecs, tvecs, mtx, dist, size=5)
            frame = draw_cube(frame, rvecs, tvecs, mtx, dist, size=3)
        cv.imshow('images',frame)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break
    out.release()
    cv.destroyAllWindows()
    webcam.release()


def image(mtx, dist, rvecs, tvecs, optimal_camera_matrix):
    """Draws the cube on the chessboard in the images from the test folder.

    Parameters
    ----------
    mtx : np.ndarray
        The camera matrix.
    dist : np.ndarray
        The distortion coefficients.
    rvecs : np.ndarray
        The rotation vectors.
    tvecs : np.ndarray
        The translation vectors.
    optimal_camera_matrix : np.ndarray
        The optimal camera matrix.
    """
    image_path = config.get('test_image_path')
    images = glob.glob(os.path.join(image_path, '*.jpg'))
    if len(images) == 0:
        print(f"No images found in the folder {image_path}. Exiting...")
        exit()
    for fname in images:
        img = cv.imread(fname)
        _, corners, _ = find_chessboard_corners(img)
        _, rvecs, tvecs = cv.solvePnP(OBJP, corners, mtx, dist)
        img = draw_world_axis(img, rvecs, tvecs, mtx, dist, size=5)
        img = draw_cube(img, rvecs, tvecs, mtx, dist, size=3)
        cv.imshow(f'cube {fname}', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


def main():
    """Main function. It will calibrate the camera (or load the camera parameters from file) and then either open the webcam or draw the cube on the chessboard in the images from the test folder."""
    mtx, dist, rvecs, tvecs, optimal_camera_matrix = calibrate_camera()
    if USE_WEBCAM:
        video(mtx, dist, rvecs, tvecs, optimal_camera_matrix)
    else:
        image(mtx, dist, rvecs, tvecs, optimal_camera_matrix)


if __name__ == '__main__':
    main()