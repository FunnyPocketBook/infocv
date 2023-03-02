import cv2
import glm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(module)s: %(message)s", level=logging.INFO,
    handlers=[
        logging.FileHandler("log.log"),
        logging.StreamHandler()
    ])

log = logging.getLogger(__name__)

block_size = 1
scaling_factor = 50


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
set_of_displayed_voxels = set()
list_of_displayed_voxels = []
list_projected_points = []
list_out_of_bounds = []
lookup_projected_to_voxel = [{}] * 4
# [
# {}, cam1
# {}, cam2
# {}, cam3
# {}, cam4
# ]
previous_foregrounds = {}
frame_counter = 0
total_time = 0

def construct_voxel_space(step = 32, voxel_space_half_size = 1000):
    log.info("Generating voxel space...")
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
                    else:
                        lookup_projected_to_voxel[cam-1][point] = [x / scaling_factor, y /scaling_factor, z/scaling_factor]
                    projected_points.append(point)
                    out_of_bounds_a.append(out_of_bounds)
                list_projected_points.append(projected_points)
                list_out_of_bounds.append(out_of_bounds_a)
                list_voxels.append([x / scaling_factor, y /scaling_factor, z/scaling_factor])
    log.info("Done generating voxel space...")
    return list_voxels

def check_voxel_visibility():
    global frame_counter, previous_foregrounds, set_of_displayed_voxels, list_of_displayed_voxels, total_time
    data = []
    true_foregrounds = {}
    f_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subtract_background, 'cam'+str(cam)) for cam in range(1,5)]
        for future in as_completed(futures):
            true_foregrounds[future.result()[0]] = future.result()[1]
    f_end = time.time()
    # log.info("Time to get foregrounds: " + str(f_end - f_start))
    new_white = {
        "cam1": [],
        "cam2": [],
        "cam3": [],
        "cam4": []
    }
    new_black = {
        "cam1": [],
        "cam2": [],
        "cam3": [],
        "cam4": []
    }
    # If previous_foregrounds is an empty dict, then we are on the first frame
    if not previous_foregrounds:
        for i, voxel in enumerate(list_voxels):
            camera_counter = 0
            for j in range(1,5):
                if list_out_of_bounds[i][j-1] == True:
                    break
                true_foreground = true_foregrounds['cam'+str(j)]
                color = true_foreground[list_projected_points[i][j-1][1],list_projected_points[i][j-1][0]]
                if color == 255:
                    camera_counter += 1
                else:
                    break
            if camera_counter == 4:
                list_of_displayed_voxels.append(voxel)
                set_of_displayed_voxels.add(tuple(voxel))
                data.append(voxel)
        frame_counter += 1
        previous_foregrounds = true_foregrounds.copy()
        return data
    
    # Get the difference between the current foreground and the previous foreground
    foreground_start = time.time()
    for cam in range(1,5):
        previous_foreground = previous_foregrounds['cam'+str(cam)]
        current_foreground = true_foregrounds['cam'+str(cam)]
        diff = cv2.absdiff(current_foreground, previous_foreground)
        # Create a mask that is white where the pixel values in the second frame are newly 255
        new_black_mask = cv2.bitwise_and(previous_foreground, cv2.bitwise_not(current_foreground))
        # Create a mask that is white where the pixel values in the second frame are newly 0
        new_white_mask = cv2.bitwise_and(cv2.bitwise_not(previous_foreground), current_foreground)
        new_white['cam'+str(cam)] = new_white_mask
        new_black['cam'+str(cam)] = new_black_mask
        # cv2.imshow('cam4 new white', new_white_mask)
        # cv2.imshow('cam4 new black', new_black_mask)	
        # cv2.imshow('cam4 diff', diff)
        # cv2.imshow('cam4 current foreground', current_foreground)
        # cv2.imshow('cam4 previous foreground', previous_foreground)
        # cv2.waitKey(1)
    foreground_end = time.time()
    # log.info("Foreground difference time: " + str(foreground_end - foreground_start))


    voxel_all_cam_start = time.time()
    for cam in range(1, 5):
        new_white_mask = new_white['cam'+str(cam)]
        new_black_mask = new_black['cam'+str(cam)]
        
        # time in ms
        transpose_start = time.time()
        # Find the indices of white and black points in the masks
        white_indices = np.transpose(np.where(new_white_mask == 255))
        black_indices = np.transpose(np.where(new_black_mask == 255))
        transpose_end = time.time()
        # log.info("Transpose time: " + str(transpose_end - transpose_start))

        voxel_start = time.time()
        white_voxels = [tuple(lookup_projected_to_voxel[cam-1][tuple(p)]) for p in white_indices if tuple(p) in lookup_projected_to_voxel[cam-1]]
        black_voxels = [tuple(lookup_projected_to_voxel[cam-1][tuple(p)]) for p in black_indices if tuple(p) in lookup_projected_to_voxel[cam-1]]
        voxel_end = time.time()
        # log.info("Voxel time: " + str(voxel_end - voxel_start))
        set_of_displayed_voxels.update(white_voxels)
        set_of_displayed_voxels.difference_update(black_voxels)
    voxel_all_cam_end = time.time()
    # log.info("Voxel all cam time: " + str(voxel_all_cam_end - voxel_all_cam_start))

    frame_counter  += 1
    previous_foregrounds = true_foregrounds.copy()
    total_time += time.time() - f_start
    log.info(f"{frame_counter}, Average time: " + str(total_time/frame_counter))
    return list(set_of_displayed_voxels)
            

def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    log.info("Generating voxel positions...")
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
    log.info("Done generating voxel positions...")
    return data

def get_camera_pos(rvecs, tvecs):
    rotM, j = cv2.Rodrigues(rvecs)
    cameraPosition = -np.matrix(rotM).transpose() * np.matrix(tvecs)
    #OpenCV Y down, Z forward meanwhile OpenGL uses Y for up so swap it
    #Coordinates converted to meters
    #Swap sign for up since opencv uses -Z
    return [cameraPosition[0]/scaling_factor,-cameraPosition[2]/scaling_factor,cameraPosition[1]/scaling_factor]


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
    #Thresholds for channels
    #These values are the best OTSU method found and i adjusted a bit afterwards
    threshhold_h = 13
    threshhold_s = 13
    threshhold_v = 75
    start_time = time.time()
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
    if frame_counter > frame_count:
        frame_counter = 0
    cap.set(1, frame_counter)
    ret, frame = cap.read()
    if not ret:
        log.error("Video error")
        return None
    if frame_counter != 0:
        cap.set(1, frame_counter-1)
        ret, previous_frame = cap.read()
        if not ret:
            log.error("Video error")
            return None
    else:
        previous_frame = frame

    
    

    #cv2.imshow('Frame ', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    morph_time = time.time()
    foreground, is_previous_frame = get_mask(frame, background_image, False)
    # frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # blurred = cv2.GaussianBlur(frameHSV, (7, 7), 0)
    # frame_channels = cv2.split(blurred)

    # # Subtract background from frame
    # h = cv2.absdiff(background_channels[0], frame_channels[0])
    # s = cv2.absdiff(background_channels[1], frame_channels[1])
    # v = cv2.absdiff(background_channels[2], frame_channels[2])

    # #Threshold channels
    # ret, h = cv2.threshold(h, threshhold_h, 255, cv2.THRESH_BINARY)
    # ret, s = cv2.threshold(s, threshhold_s, 255, cv2.THRESH_BINARY)
    # ret, v = cv2.threshold(v, threshhold_v, 255, cv2.THRESH_BINARY)

    # # Erode and dilate
    # h = cv2.erode(h, kernelErode, iterations=3)
    # h = cv2.dilate(h, kernelDialate, iterations=2)

    # s = cv2.erode(s, kernelErode, iterations=3)

    # v = cv2.erode(v, kernelErode, iterations=1)

    # #Combine channels
    # foreground = cv2.bitwise_and(h, s)
    # foreground = cv2.bitwise_and(foreground, v)
    # foreground = cv2.dilate(foreground, kernelDialate, iterations=2)

    #H channel
    # temp_frame = cv2.absdiff(background_channels[0], frame_channels[0])
    # t1, im1 = cv2.threshold(temp_frame, threshhold_h, 255,  cv2.THRESH_BINARY)
    # cv2.erode(im1,kernelErode,im1)
    # cv2.erode(im1,kernelErode,im1)
    # cv2.erode(im1,kernelErode,im1)
    # cv2.dilate(im1,kernelDialate,im1)
    # cv2.dilate(im1,kernelDialate,im1)
    # #cv2.imshow('H ', im1)

    # #S channel
    # temp_frame = cv2.absdiff(background_channels[1], frame_channels[1])
    # t2, im2 = cv2.threshold(temp_frame, threshhold_s, 255,  cv2.THRESH_BINARY)
    # cv2.erode(im2,kernelErode,im2)
    # cv2.erode(im2,kernelErode,im2)
    # cv2.erode(im2,kernelErode,im2)
    # #cv2.imshow('S ', im2)
    
    # #V channel
    # temp_frame = cv2.absdiff(background_channels[2], frame_channels[2])
    # t3, im3 = cv2.threshold(temp_frame, threshhold_v, 255,  cv2.THRESH_BINARY)
    # cv2.erode(im3,kernelErode,im3)
    # #cv2.imshow('V ', im3)
    
    # true_foreground = cv2.bitwise_or(im1, im2)
    # true_foreground = cv2.bitwise_or(true_foreground, im3)
    # cv2.dilate(true_foreground,kernelDialate,true_foreground)
    # cv2.dilate(true_foreground,kernelDialate,true_foreground)
    
    #cv2.imshow('True foreground ', true_foreground)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #code commet incase i need it
    #contours, hierarchy = cv2.findContours(foreground[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,)
    #cv2.erode(foreground[1],kernelErode,foreground[1] )
    #cv2.dilate(foreground[1],kernelDia,foreground[1] )
    #cv2.drawContours(foreground[1], contours, -1, (0,255,0), 3)
    # log.info(f"Morph: {time.time() - morph_time}")
    # log.info(f"Background: {time.time() - start_time}")
    return cameraID, foreground
def get_mask(frame, background_image, is_previous_frame):
    background_imageHSV = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
    background_channels = cv2.split(background_imageHSV)
    kernelErode = np.ones((3, 3), np.uint8)
    kernelDilate = np.ones((3, 3), np.uint8)

    threshhold_h = 13
    threshhold_s = 13
    threshhold_v = 75
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(frameHSV, (7, 7), 0)
    frame_channels = cv2.split(blurred)

    # Subtract background from frame
    h = cv2.absdiff(background_channels[0], frame_channels[0])
    s = cv2.absdiff(background_channels[1], frame_channels[1])
    v = cv2.absdiff(background_channels[2], frame_channels[2])

    #Threshold channels
    _, h = cv2.threshold(h, threshhold_h, 255, cv2.THRESH_BINARY)
    _, s = cv2.threshold(s, threshhold_s, 255, cv2.THRESH_BINARY)
    _, v = cv2.threshold(v, threshhold_v, 255, cv2.THRESH_BINARY)

    # Erode and dilate
    h = cv2.erode(h, kernelErode, iterations=3)
    h = cv2.dilate(h, kernelDilate, iterations=2)

    s = cv2.erode(s, kernelErode, iterations=3)

    v = cv2.erode(v, kernelErode, iterations=1)

    #Combine channels
    foreground = cv2.bitwise_or(h, s)
    foreground = cv2.bitwise_or(foreground, v)
    foreground = cv2.dilate(foreground, kernelDilate, iterations=2)

    return foreground, is_previous_frame


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