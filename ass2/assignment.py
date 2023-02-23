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

#Load files from directory and subtract background from video
#TODO: File loading should probablty be in a separate function so files are not loaded on each update
def subtract_background(path = 'ass2/data/cam1/'):
    cap = cv2.VideoCapture(path + 'video.avi')
    background_image = cv2.imread(path + 'background.jpg')
    background_imageHSV = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
    background_channels = cv2.split(background_imageHSV)
    threshhold_h = 10
    threshhold_s = 15
    threshhold_v = 20
    kernelErode = np.ones((2, 2), np.uint8)
    kernelDialate = np.ones((3, 3), np.uint8)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame ', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_channels = cv2.split(frameHSV)

        #H channel
        temp_frame = cv2.absdiff(background_channels[0], frame_channels[0])
        im1 = cv2.threshold(temp_frame, threshhold_h, 255, cv2.THRESH_BINARY)
        #cv2.erode(im1[1],kernelErode,im1[1] )
        cv2.imshow('H ', im1[1])
        #S channel
        temp_frame = cv2.absdiff(background_channels[1], frame_channels[1])
        im2 = cv2.threshold(temp_frame, threshhold_s, 255, cv2.THRESH_BINARY)
        #cv2.erode(im2[1],kernelErode,im2[1] )
        cv2.imshow('S ', im2[1])
        
        #V channel
        temp_frame = cv2.absdiff(background_channels[2], frame_channels[2])
        im3 = cv2.threshold(temp_frame, threshhold_v, 255, cv2.THRESH_BINARY)
        cv2.dilate(im3[1],kernelDialate,im3[1] )
        cv2.imshow('V ', im3[1])
        
        true_foreground = cv2.bitwise_and(im3[1], im2[1])
        #cv2.erode(true_foreground,kernelErode,true_foreground )
        #cv2.dilate(true_foreground,kernelDialate,true_foreground )
        
        cv2.imshow('True foreground ', true_foreground)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #code commet incase i need it
        #contours, hierarchy = cv2.findContours(foreground[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,)
        #cv2.erode(foreground[1],kernelErode,foreground[1] )
        #cv2.dilate(foreground[1],kernelDia,foreground[1] )
        #cv2.drawContours(foreground[1], contours, -1, (0,255,0), 3)

    
    return True