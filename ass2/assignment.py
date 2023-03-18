import cv2
import glm
import numpy as np
import copy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from engine.config import config

from skimage import measure
from skimage.draw import ellipsoid

block_size = 1
scalling_factor = 50


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    colors = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            #colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
            colors.append([1.0, 1.0, 1.0])
    return data, colors

#Globals
list_voxels = []
list_list_points = []
listout_of_bounds = []
list_offline_histograms = []
cluster_colors = []
voxel_to_cluster = []
frame_counter = 0

# tracking
path_history = []
path_history_colors = []

def construct_voxel_space(step = 32, voxel_space_half_size = 1000):
    print("Generating voxel space...")
    camera_props = {}
    for cam in range(1,5):
        camM, camd, camrvecs, camtvecs = load_camera_properties('cam'+str(cam))
        camera_props['cam'+str(cam)] = [camM, camd, camrvecs, camtvecs]

    for x in range(-voxel_space_half_size,voxel_space_half_size,step):
        for y in range(0,2300,step):
            for z in range(-voxel_space_half_size - 1000,voxel_space_half_size + 1000,step):
                points = np.float32([[x,z,-y]])
                projected_points = []
                out_of_bounds_a = []
                for cam in range(1,5):
                    out_of_bounds = False
                    camM, camd, camrvecs, camtvecs = camera_props['cam'+str(cam)]
                    imgpts, jac = cv2.projectPoints(points, camrvecs, camtvecs, camM, camd)
                    if imgpts[0][0][1] > 485 or imgpts[0][0][0] > 643 or imgpts[0][0][0] < 0 or imgpts[0][0][1] < 0:
                        out_of_bounds = True
                        out_of_bounds_a.append(out_of_bounds)
                        continue
                    point = tuple(map(int, imgpts[0].ravel()))
                    projected_points.append(point)
                    out_of_bounds_a.append(out_of_bounds)
                list_list_points.append(projected_points)
                listout_of_bounds.append(out_of_bounds_a)
                list_voxels.append([x / scalling_factor, y /scalling_factor, z/scalling_factor])
    print("Done generating voxel space...")
    return list_voxels

def check_voxel_visibility():
    global frame_counter, path_history, path_history_colors
    data = []
    pixels_cam = []
    colors = []
    true_foregrounds = {}
    frames = {}
    for cam in range(1,5):
        true_foregrounds['cam'+str(cam)], frames['cam'+str(cam)] = subtract_background('cam'+str(cam))
        # res = cv2.bitwise_and(frames['cam'+str(cam)],frames['cam'+str(cam)],mask = true_foregrounds['cam'+str(cam)])
        # cv2.imshow('Masked frame', res)
        # cv2.waitKey(0)
        
    for i, voxel in enumerate(list_voxels):
        camera_counter = 0
        #rgb_col = []
        for j in range(1,5):
            if listout_of_bounds[i][j-1] == True:
                break
            true_foreground = true_foregrounds['cam'+str(j)]
            frame = frames['cam'+str(j)]
            color = true_foreground[list_list_points[i][j-1][1],list_list_points[i][j-1][0]]
            #rgb_col.append(frame[list_list_points[i][j-1][1],list_list_points[i][j-1][0]])
            if color == 255:
                camera_counter += 1
            else:
                break
        if camera_counter == 4:
            data.append(list_voxels[i])
            pixels_cam.append(list_list_points[i])

    print(frame_counter)
    
    if frame_counter == 0:
        cluster_colors.append([[0],[0,1,1]])
        cluster_colors.append([[1],[0.1,0.5,0.1]])
        cluster_colors.append([[2],[0.5,0.1,0.1]])
        cluster_colors.append([[3],[0.1,0.1,0.5]])
    labels, centers = construct_kmeans_clusters(data)
    colors, center_colors = construct_models(labels, pixels_cam, data, centers, frames)
    for i, center in enumerate(centers):
        path_history.append([center[0],0,center[1]])
        path_history_colors.append(center_colors[i][1])
    data.extend(path_history)
    colors.extend(path_history_colors)
    frame_counter  += 5
    return data, colors

def construct_kmeans_clusters(points):
    #Remove height and convert to float
    pointsXZ = np.zeros((len(points),2), dtype=np.float32)
    for v in range(len(points)):
        pointsXZ[v] = points[v][0],points[v][2]
    #Cluster count
    k = 4
    #kmeans stuff
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    compactness, labels, centers = cv2.kmeans(pointsXZ,k,None,criteria,attempts,flags)
    return labels, centers

def find_angle(p1, p2, p3):
    #Calculate the vectors between the center point and the other two points
    v1 = (p1[0]-p3[0], p1[1]-p3[1])
    v2 = (p2[0]-p3[0], p2[1]-p3[1])

    #Calculate the dot product between the two vectors
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]

    #Calculate the lengths of the two vectors
    v1_length = math.sqrt(v1[0]**2 + v1[1]**2)
    v2_length = math.sqrt(v2[0]**2 + v2[1]**2)

    #Calculate the angle between the two vectors
    angle_radians = math.acos(dot_product / (v1_length * v2_length))

    #Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def construct_models(labels , pixeldata, voxels, centers, online_frames):

    #Globals
    global list_offline_histograms
    global frame_counter
    global cluster_colors
    global voxel_to_cluster


    frameCam1 =  online_frames['cam'+str(1)]
    frameCam2 =  online_frames['cam'+str(2)]
    frameCam3 =  online_frames['cam'+str(3)]
    frameCam4 =  online_frames['cam'+str(4)]

    if frame_counter == 0:
        path = 'ass2/data/'
        #Read specific frames 
        #For cam1,2,4 first frame is good enough
        capCam1 = cv2.VideoCapture(path + 'cam1/video.avi')
        capCam2 = cv2.VideoCapture(path + 'cam2/video.avi')
        capCam3 = cv2.VideoCapture(path + 'cam3/video.avi')
        capCam4 = cv2.VideoCapture(path + 'cam4/video.avi')

        capCam1.set(1,0)
        _, frameCam1 = capCam1.read()
        capCam2.set(1,0)
        _, frameCam2 = capCam2.read()
        #Frame 1200 seems the best for cam3 only
        capCam3.set(1,1200)
        _, frameCam3 = capCam3.read()
        capCam4.set(1,0)
        _, frameCam4 = capCam4.read()

    #Color array used for the comparison step
    colors = []

    #Calculate clusters distance to camera
    distancesPView = []
    cam_positions, _ = get_cam_positions()
    for i in range(4):
        distances = []
        for j in range(4):
            cam2Dx = cam_positions[i][0]
            cam2Dz = cam_positions[i][2]
            cam2D = [cam2Dx,cam2Dz]
            cluster2D = centers[j]
            dist = math.dist(cam2D,cluster2D)
            distances.append(dist)
        distancesPView.append(distances)

    #Check if cluster is occluded
    #if two clusters have a low angle to the camera
    #   THEN: if one cluster is closer to the camera than the other.
    cam_good_view = []
    for k in range(4):
        view_good = []
        for i in range(4):
            is_visible = True
            for j in range(4):
                if i == j:
                    continue
                cam2D = [cam_positions[k][0],cam_positions[k][2]]
                cluster2DF = centers[i]
                cluster2DS = centers[j]
                diff = find_angle(cluster2DF, cluster2DS, cam2D)
                #print(diff)
                if diff < 5:
                    if distancesPView[k][i] > distancesPView[k][j]:
                        is_visible = False
            view_good.append(is_visible)
        cam_good_view.append(view_good)

    if len(cam_good_view) == 1:
        print('yes')

    #Transform frames from BGR to HSV
    frameCam1 = cv2.cvtColor(frameCam1, cv2.COLOR_BGR2HSV)
    frameCam2 = cv2.cvtColor(frameCam2, cv2.COLOR_BGR2HSV)
    frameCam3 = cv2.cvtColor(frameCam3, cv2.COLOR_BGR2HSV)
    frameCam4 = cv2.cvtColor(frameCam4, cv2.COLOR_BGR2HSV)
    
    #Array holder for all views
    frame_views = []
    frame_views.append(frameCam1)
    frame_views.append(frameCam2)
    frame_views.append(frameCam3)
    frame_views.append(frameCam4)

    #Final return array of colors
    data = []

    #Views
    view_array = []

    #Masks
    for j in range(4):
        #we just need the shape, it doesn't matter which frame we use
        person1 = np.zeros((frameCam2.shape[0],frameCam2.shape[1],1), dtype=np.uint8)
        person2 = np.zeros((frameCam2.shape[0],frameCam2.shape[1],1), dtype=np.uint8)
        person3 = np.zeros((frameCam2.shape[0],frameCam2.shape[1],1), dtype=np.uint8)
        person4 = np.zeros((frameCam2.shape[0],frameCam2.shape[1],1), dtype=np.uint8)
        for i in range(len(labels)):
            if voxels[i][1] < 7 or voxels[i][1] > 30: #Roughly above pants and bellow head (only tshirt)
                continue

            if labels[i] == 0: 
                person1[pixeldata[i][j][1],pixeldata[i][j][0]] = 255
            elif labels[i] == 1:
                person2[pixeldata[i][j][1],pixeldata[i][j][0]] = 255
            elif labels[i] == 2:
                person3[pixeldata[i][j][1],pixeldata[i][j][0]] = 255
            elif labels[i] == 3:
                person4[pixeldata[i][j][1],pixeldata[i][j][0]] = 255
        view_array.append([person1,person2,person3,person4])
        
    #Histogram parameters
    histSize = 16
    histRange = (1, 256)
    accumulate = False
    hist_h = frameCam2.shape[1]

    #Histograms array
    list_view_histograms = []
    #Subtract frame pixels from mask, split into channels, calculate histograms per channel and normalize values
    for j in range(len(view_array)):
        list_histograms = []
        for i in range(4):
            res = cv2.bitwise_and(frame_views[j],frame_views[j],mask = view_array[j][i])
            bgr_planes = cv2.split(res)
            h_hist = cv2.calcHist(bgr_planes, [0], view_array[j][i], [histSize], histRange, accumulate=accumulate)
            s_hist = cv2.calcHist(bgr_planes, [1], view_array[j][i], [histSize], histRange, accumulate=accumulate)
            v_hist = cv2.calcHist(bgr_planes, [2], view_array[j][i], [histSize], histRange, accumulate=accumulate)
            cv2.normalize(h_hist, h_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(s_hist, s_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(v_hist, v_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

            list_histograms.append([h_hist,s_hist,v_hist])
        list_view_histograms.append(list_histograms)
    
    #if this is the first frame, push the histograms to storage (Offline histograms) and skip comparison
    if frame_counter == 0:
        list_offline_histograms = copy.copy(list_view_histograms)
        for i in range(len(labels)):
            for j in range(len(cluster_colors)):
                if labels[i] == cluster_colors[j][0]:
                    data.append(cluster_colors[j][1])
                    voxel_to_cluster.append(labels[i])
        #Early return
        return data, cluster_colors

    #Calculate online to offline comparison values
    #PER: view
    #     PER: online-histogram
    #                        PER: offline-histograms 
    #                                 DO: assign [ id of online cluster, value for each comparison, id of offline cluster]
    #     DO: assign per each view all data of previous array
    view_match_array = []
    for k in range(4):
        match_array = []
        for i in range(len(list_view_histograms)):
            oldVal = 0
            newVal = 0
            hVal = 0
            sVal = 0
            vVal = 0
            for j in range(len(list_offline_histograms)):
                #If the color model is occluded don't calculate it and return only a 0.
                if cam_good_view[k][i] == True:
                #----------------------------------------------
                    hVal = cv2.compareHist(list_view_histograms[k][i][0],list_offline_histograms[k][j][0],0)
                    sVal = cv2.compareHist(list_view_histograms[k][i][1],list_offline_histograms[k][j][1],0)
                    vVal = cv2.compareHist(list_view_histograms[k][i][2],list_offline_histograms[k][j][2],0)
                #-----------------------------------------------
                newVal = hVal+sVal+vVal
                match_array.append([i,newVal,j])
        view_match_array.append([k,match_array])       
    
    #Find best match from the data in the previous loop
    #PER: online cluster index [ j ]
    #       PER: view
    #               DO: get all possible comparisons for the j cluster and assign them to array
    #       PER: view
    #               PER: comparison
    #                       DO: if its higher than previous comparion, then the index matches better -> assign new comparison value and the index that it matched on. Do this for all comparisons.
    #       DO: Assign from offline colors the correct online color and append that we have used this label so we don't label 2 cluster with the same color.
    match_index = 0
    used_indexes = []
    for j in range(4):
        firstId = 0
        secondId = 0
        thirdId = 0
        fourthId = 0
        comparisonsPView = []
        for view in range(4):
                firstId = view_match_array[view][1][match_index][1]
                secondId = view_match_array[view][1][match_index+1][1]
                thirdId = view_match_array[view][1][match_index+2][1]
                fourthId = view_match_array[view][1][match_index+3][1]
                comparisonsPView.append([firstId,secondId,thirdId,fourthId])

        Indexid = 0
        oldVal = 0
        for k in range(4):
            for i in range(4):
                if comparisonsPView[k][i] > oldVal:
                    if i not in used_indexes:
                        oldVal = comparisonsPView[k][i]
                        Indexid = i
        match_index += 4
        used_indexes.append(Indexid)
        colors.append([[j],cluster_colors[Indexid][1]])
    #How the view_match_array looks like
    #view_match_array[0-4] views
    #view_match_array[0-4][0] = k
    #view_match_array[0-4][1] = match_array
    #view_match_array[0-4][1][0-15] = all comparisons 
    #view_match_array[0-4][1][0-15][0] = id of guy
    #view_match_array[0-4][1][0-15][1] = value of comapirons
    #view_match_array[0-4][1][0-15][2] = to which cluster it was compared to

    for i in range(len(labels)):
        for j in range(len(colors)):
            if labels[i] == colors[j][0]:
                data.append(colors[j][1])
                voxel_to_cluster.append(labels[i])
    return data, colors

def set_voxel_positions(width, height, depth):
    global frame_counter
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    colors = []
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
    for i in range(8):
        colors.append([1.0, 0, 0])
    
    vx, cl = check_voxel_visibility()
    data = data + vx
    colors = colors + cl
    print("Done generating voxel positions...")
    return data, colors

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
            cam4pos],[[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,1.0,0]]

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

def generate_mesh():
    print('Generating mesh')
    # Generate a level set about zero of two identical ellipsoids in 3D
    #voxels = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'])
    #size = len(voxels)
    ellip_base = ellipsoid(3, 3, 3, levelset=True)
    #ellip_double = np.concatenate((ellip_base[:-1, ...],ellip_base[2:, ...]), axis=0)
    voxels = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'])
    #data = np.zeros((len(voxels),len(voxels),len(voxels)), np.float32)
    # input: z_coords, y_coords, x_coords
    zint = []
    yint = []
    xint = []
    for x in range(len(voxels)):
        zint.append(np.floor(voxels[x][2]).astype(int))
        yint.append(np.floor(voxels[x][1]).astype(int))
        xint.append(np.floor(voxels[x][0]).astype(int))
    shape = tuple([np.max(intcoords) +1 for intcoords in [zint, yint, xint]])
    image = np.zeros(shape)
    image[zint, yint, xint] += 1

    #data = np.reshape(voxels,(3,1))
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(image,0, spacing=(1,1,1))

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 32")
    ax.set_ylabel("y-axis: b = 32")
    ax.set_zlabel("z-axis: c = 32")

    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_zlim(-32, 32)

    plt.tight_layout()
    plt.show()
    print('Done generating mesh')
    return True

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
    # cv2.imshow('Frame ', frame)
    # cv2.imshow('True foreground ', true_foreground)
    # cv2.waitKey(0)
    return true_foreground, frame

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