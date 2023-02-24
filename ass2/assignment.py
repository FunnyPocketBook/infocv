import cv2
import glm
import random
import numpy as np

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    subtract_background()
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations

#Create background image from all frames of videoclip. Each subsequent frame is slightly more transparent
def get_background_image(path = 'ass2/data/cam1/'):
    cap = cv2.VideoCapture(path + 'background.avi')
    ret, next_frame = cap.read()
    first_frame = next_frame
    alpha = 0.0 
    beta = 0.0
    counter = 0

    if not ret:
        return first_frame

    average_image = cv2.addWeighted(next_frame,alpha,first_frame,beta,0.0)
    while(ret):
        alpha = 1.0 / (counter+1)
        beta = 1.0 - alpha
        counter += 1
        average_image = cv2.addWeighted(next_frame,alpha,first_frame,beta,0.0)
        ret, next_frame = cap.read()
    cv2.imwrite(path + 'background.jpg',average_image)
    return average_image

#Thresholds for channels
#These values are the best OTSU method found and i adjusted a bit afterwards
threshhold_h = 13
threshhold_s = 13
threshhold_v = 75

#UI Names
H_name = 'H'
S_name = 'S'
V_name = 'V'
window_bar_name = 'Bars'

#Load files from directory and subtract background from video
#TODO: File loading should probablty be in a separate function so files are not loaded on each update
#TODO: rewrite the function so it works like this:
#On a single update in the scene 
#Each camera calls this
#Get the new frame from the video per camera
#Perform background subtraction on the new frame
#Return true foreground
def subtract_background(path = 'ass2/data/cam1/'):
    cap = cv2.VideoCapture(path + 'video.avi')
    background_image = cv2.imread(path + 'background.jpg')
    background_imageHSV = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
    background_channels = cv2.split(background_imageHSV)
    kernelErode = np.ones((3, 3), np.uint8)
    kernelDialate = np.ones((3, 3), np.uint8)

    #Initialize UI elements
    cv2.namedWindow(window_bar_name)
    cv2.createTrackbar(H_name, window_bar_name , threshhold_h, 255, on_low_H_thresh_trackbar)
    cv2.createTrackbar(S_name, window_bar_name , threshhold_s, 255, on_low_S_thresh_trackbar)
    cv2.createTrackbar(V_name, window_bar_name , threshhold_v, 255, on_low_V_thresh_trackbar)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame ', frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(frameHSV, (7, 7), 0)
        frame_channels = cv2.split(blurred)

        #H channel
        temp_frame = cv2.absdiff(background_channels[0], frame_channels[0])
        t1, im1 = cv2.threshold(temp_frame, threshhold_h, 255,  cv2.THRESH_BINARY)
        cv2.erode(im1,kernelErode,im1)
        cv2.erode(im1,kernelErode,im1)
        cv2.erode(im1,kernelErode,im1)
        cv2.dilate(im1,kernelDialate,im1)
        cv2.dilate(im1,kernelDialate,im1)
        cv2.imshow('H ', im1)

        #S channel
        temp_frame = cv2.absdiff(background_channels[1], frame_channels[1])
        t2, im2 = cv2.threshold(temp_frame, threshhold_s, 255,  cv2.THRESH_BINARY)
        cv2.erode(im2,kernelErode,im2)
        cv2.erode(im2,kernelErode,im2)
        cv2.erode(im2,kernelErode,im2)
        cv2.imshow('S ', im2)
        
        #V channel
        temp_frame = cv2.absdiff(background_channels[2], frame_channels[2])
        t3, im3 = cv2.threshold(temp_frame, threshhold_v, 255,  cv2.THRESH_BINARY)
        cv2.erode(im3,kernelErode,im3)
        cv2.imshow('V ', im3)
        
        true_foreground = cv2.bitwise_or(im1, im2)
        true_foreground = cv2.bitwise_or(true_foreground, im3)
        cv2.dilate(true_foreground,kernelDialate,true_foreground)
        cv2.dilate(true_foreground,kernelDialate,true_foreground)
        
        cv2.imshow('True foreground ', true_foreground)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #code commet incase i need it
        #contours, hierarchy = cv2.findContours(foreground[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,)
        #cv2.erode(foreground[1],kernelErode,foreground[1] )
        #cv2.dilate(foreground[1],kernelDia,foreground[1] )
        #cv2.drawContours(foreground[1], contours, -1, (0,255,0), 3)

    
    return True

#Slider events
def on_low_H_thresh_trackbar(val):
    global threshhold_h
    threshhold_h = val
    threshhold_h = min(255, threshhold_h)
    cv2.setTrackbarPos(H_name, window_bar_name, threshhold_h)
def on_low_S_thresh_trackbar(val):
    global threshhold_s
    threshhold_s = val
    threshhold_s = min(255, threshhold_s)
    cv2.setTrackbarPos(S_name, window_bar_name, threshhold_s)
def on_low_V_thresh_trackbar(val):
    global threshhold_v
    threshhold_v = val
    threshhold_v = min(255, threshhold_v)
    cv2.setTrackbarPos(V_name, window_bar_name, threshhold_v)