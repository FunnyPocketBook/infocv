import cv2
import glm
import numpy as np
import copy
block_size = 1
scalling_factor = 50


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data

#Globals
list_voxels = []
list_list_points = []
listout_of_bounds = []
frame_counter = 0

def construct_voxel_space(step = 32, voxel_space_half_size = 1000):
    print("Generating voxel space...")
    camera_props = {}
    for cam in range(1,5):
        camM, camd, camrvecs, camtvecs = load_camera_properties('cam'+str(cam))
        camera_props['cam'+str(cam)] = [camM, camd, camrvecs, camtvecs]

    for x in range(-voxel_space_half_size,voxel_space_half_size,step):
        for y in range(0,voxel_space_half_size * 2,step):
            for z in range(-voxel_space_half_size,voxel_space_half_size,step):
                points = np.float32([[x,z,-y]])
                projected_points = []
                out_of_bounds_a = []
                for cam in range(1,5):
                    out_of_bounds = False
                    camM, camd, camrvecs, camtvecs = camera_props['cam'+str(cam)]
                    imgpts, jac = cv2.projectPoints(points, camrvecs, camtvecs, camM, camd)
                    point = tuple(map(int, imgpts[0].ravel()))
                    if point[1] > 485 or point[0] > 643 or point[0] < 0 or point[1] < 0:
                        out_of_bounds = True
                    projected_points.append(point)
                    out_of_bounds_a.append(out_of_bounds)
                list_list_points.append(projected_points)
                listout_of_bounds.append(out_of_bounds_a)
                list_voxels.append([x / scalling_factor, y /scalling_factor, z/scalling_factor])
    print("Done generating voxel space...")
    return list_voxels

def check_voxel_visibility():
    global frame_counter
    data = []
    true_foregrounds = {}
    for cam in range(1,5):
        true_foregrounds['cam'+str(cam)] = subtract_background('cam'+str(cam))

    for i in range(len(list_voxels)):

        camera_counter = 0
        for j in range(1,5):
            if listout_of_bounds[i][j-1] == True:
                break
            true_foreground = true_foregrounds['cam'+str(j)]
            color = true_foreground[list_list_points[i][j-1][1],list_list_points[i][j-1][0]]
            if color == 255:
                camera_counter += 1
            else:
                break
        if camera_counter == 4:
            data.append(list_voxels[i])
    frame_counter  += 1
    print(frame_counter)
    return data
            

def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    print("Generating voxel positions...")
    data = []
    xL = int(-width/2)
    xR = int(width/2)
    yL = 0
    yR = height
    zL = int(-depth/2)
    zR = int(depth/2)
    #Bottom 4
    data.append([xL,yL,zL])
    data.append([xL,yL,zR])
    data.append([xR,yL,zR])
    data.append([xR,yL,zL])
    #Top 4
    data.append([xL,yR,zL])
    data.append([xL,yR,zR])
    data.append([xR,yR,zR])
    data.append([xR,yR,zL])

    data = data + check_voxel_visibility()
    print("Done generating voxel positions...")
    return data

def get_camera_pos(rvecs, tvecs):
    rotM, j = cv2.Rodrigues(rvecs)
    cameraPosition = -np.matrix(rotM).transpose() * np.matrix(tvecs)
    #OpenCV Y down, Z forward meanwhile OpenGL uses Y for up so swap it
    #Coordinates converted to meters
    #Swap sign for up since opencv uses -Z
    return [cameraPosition[0]/scalling_factor,-cameraPosition[2]/scalling_factor,cameraPosition[1]/scalling_factor]


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.

    #Cam1
    cam1M, cam1d, cam1rvecs, cam1tvecs = load_camera_properties('cam1')
    cam1pos = get_camera_pos(cam1rvecs, cam1tvecs)

    #Cam2
    cam2M, cam2d, cam2rvecs, cam2tvecs = load_camera_properties('cam2')
    cam2pos = get_camera_pos(cam2rvecs, cam2tvecs)
    #Cam3
    cam3M, cam3d, cam3rvecs, cam3tvecs = load_camera_properties('cam3')
    cam3pos = get_camera_pos(cam3rvecs, cam3tvecs)
    #Cam4
    cam4M, cam4d, cam4rvecs, cam4tvecs = load_camera_properties('cam4')
    cam4pos = get_camera_pos(cam4rvecs, cam4tvecs)
    
    return [cam1pos,
            cam2pos,
            cam3pos,
            cam4pos]

def get_extrensic_matrix(rvecs, tvecs):
    rv = rvecs.ravel()
    m44eye = np.identity(4)
    rotM, j = cv2.Rodrigues(rv)
    m44eye[:3, :3] = rotM
    return m44eye

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    #Cam1
    cam1M, cam1d, cam1rvecs, cam1tvecs = load_camera_properties('cam1')
    cam1M44 = get_extrensic_matrix(cam1rvecs, cam1tvecs)

    #Cam2
    cam2M, cam2d, cam2rvecs, cam2tvecs = load_camera_properties('cam2')
    cam2M44 = get_extrensic_matrix(cam2rvecs, cam2tvecs)
    #Cam3
    cam3M, cam3d, cam3rvecs, cam3tvecs = load_camera_properties('cam3')
    cam3M44 = get_extrensic_matrix(cam3rvecs, cam3tvecs)
    #Cam4
    cam4M, cam4d, cam4rvecs, cam4tvecs = load_camera_properties('cam4')
    cam4M44 = get_extrensic_matrix(cam4rvecs, cam4tvecs)
    #OpenCV to OpenGL conversion?
    cam_angles = [[0,0,90], [0,0,90], [0,0,90], [0,0,90]]
    cam_rotations = [glm.mat4(cam1M44), glm.mat4(cam2M44), glm.mat4(cam3M44), glm.mat4(cam4M44)]
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
def subtract_background(cameraID = 'cam1'):
    global frame_counter
    path = 'ass2/data/' + cameraID + '/'
    cap = cv2.VideoCapture(path + 'video.avi')
    #fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #duration = frame_count/fps
    background_image = cv2.imread(path + 'background.jpg')
    background_imageHSV = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
    background_channels = cv2.split(background_imageHSV)
    kernelErode = np.ones((3, 3), np.uint8)
    kernelDialate = np.ones((3, 3), np.uint8)
    #Initialize UI elements
    #cv2.namedWindow(window_bar_name)
    #cv2.createTrackbar(H_name, window_bar_name , threshhold_h, 255, on_low_H_thresh_trackbar)
    #cv2.createTrackbar(S_name, window_bar_name , threshhold_s, 255, on_low_S_thresh_trackbar)
    #cv2.createTrackbar(V_name, window_bar_name , threshhold_v, 255, on_low_V_thresh_trackbar)
    if  frame_counter > frame_count:
        frame_counter = 0
    cap.set(1,frame_counter)
    ret, frame = cap.read()
    if not ret:
        print("Video error")
        return None
    #cv2.imshow('Frame ', frame)
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
    #cv2.imshow('H ', im1)

    #S channel
    temp_frame = cv2.absdiff(background_channels[1], frame_channels[1])
    t2, im2 = cv2.threshold(temp_frame, threshhold_s, 255,  cv2.THRESH_BINARY)
    cv2.erode(im2,kernelErode,im2)
    cv2.erode(im2,kernelErode,im2)
    cv2.erode(im2,kernelErode,im2)
    #cv2.imshow('S ', im2)
    
    #V channel
    temp_frame = cv2.absdiff(background_channels[2], frame_channels[2])
    t3, im3 = cv2.threshold(temp_frame, threshhold_v, 255,  cv2.THRESH_BINARY)
    cv2.erode(im3,kernelErode,im3)
    #cv2.imshow('V ', im3)
    
    true_foreground = cv2.bitwise_or(im1, im2)
    true_foreground = cv2.bitwise_or(true_foreground, im3)
    cv2.dilate(true_foreground,kernelDialate,true_foreground)
    cv2.dilate(true_foreground,kernelDialate,true_foreground)
    
    #cv2.imshow('True foreground ', true_foreground)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #code commet incase i need it
    #contours, hierarchy = cv2.findContours(foreground[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,)
    #cv2.erode(foreground[1],kernelErode,foreground[1] )
    #cv2.dilate(foreground[1],kernelDia,foreground[1] )
    #cv2.drawContours(foreground[1], contours, -1, (0,255,0), 3)
    return true_foreground

#Load camera properties from folder directory
def load_camera_properties(cameraID = 'cam1'):
    path = 'ass2/data/' + cameraID + '/'
    s = cv2.FileStorage(path + "config.xml", cv2.FILE_STORAGE_READ)
    camMatrix = s.getNode('CameraMatrix').mat()
    d = s.getNode('DistortionCoeffs').mat()
    rvecs = s.getNode('RotationValues').mat()
    tvecs = s.getNode('TranslationValues').mat()
    return camMatrix, d, rvecs, tvecs

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