import cv2
import glm
import numpy as np
import copy
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
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

#Globals
list_voxels = []
list_list_points = []
listout_of_bounds = []
list_offline_histograms = []
cluster_colors = []
frame_counter = 0

# timers and counters
set_voxel_positions_total_time = 0
background_check_total_time = 0
voxel_check_total_time = 0
voxel_vis_total_time = 0

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
    global frame_counter, background_check_total_time, voxel_check_total_time, voxel_vis_total_time
    data = []
    pixels_cam2 = []
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
            pixels_cam2.append(list_list_points[i][1])

    print(frame_counter)
    labels = construct_kmeans_clusters(data)
    if frame_counter == 0:
        cluster_colors.append([[0],[0.1,0.1,0.1]])
        cluster_colors.append([[1],[0.1,0.5,0.1]])
        cluster_colors.append([[2],[0.5,0.1,0.1]])
        cluster_colors.append([[3],[0.1,0.1,0.5]])
        colors = construct_models('ass2/data/cam2/',labels, pixels_cam2, data)
    else:
        colors = construct_models('ass2/data/cam2/',labels, pixels_cam2, data)
    frame_counter  += 1
    return data, colors

def construct_kmeans_clusters(points):
    #Remove height and convert to float
    pointsXY = np.zeros((len(points),2), dtype=np.float32)
    for v in range(len(points)):
        pointsXY[v] = points[v][0],points[v][2]
    #Cluster count
    k = 4
    #kmeansstuff
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    compactness, labels, centers = cv2.kmeans(pointsXY,k,None,criteria,attempts,flags)
    #TODO: assign each voxel a refference to the same center
    return labels

#default cam 2 first frame since its pretty good
def construct_models(path, labels , pixeldata, voxels ):
    cap = cv2.VideoCapture(path + 'video.avi')
    ret, frame = cap.read()
    colors = []
    data = []
    if not ret: 
        return colors
    person1 = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
    person2 = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
    person3 = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
    person4 = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
    for i in range(len(labels)):
        if labels[i] == 0 and voxels[i][1] > 9 and voxels[i][1] < 30: #Roughly above pants and bellow head (only tshirt)
            #colors.append([0.1,0.1,0.1])
            #Get color pixel for 2nd camera only
            #TODO: If multiple cameras are used, adjust accoredinly
            person1[pixeldata[i][1],pixeldata[i][0]] = 255
        elif labels[i] == 1 and voxels[i][1] > 9 and voxels[i][1] < 30:
            #colors.append([0.1,0.5,0.1])
            person2[pixeldata[i][1],pixeldata[i][0]] = 255
        elif labels[i] == 2 and voxels[i][1] > 9 and voxels[i][1] < 30:
            #colors.append([0.5,0.1,0.1])
            person3[pixeldata[i][1],pixeldata[i][0]] = 255
        elif labels[i] == 3 and voxels[i][1] > 9 and voxels[i][1] < 30:
            #colors.append([0.1,0.1,0.5])
            person4[pixeldata[i][1],pixeldata[i][0]] = 255
    
    histSize = 256
    histRange = (0, 256)
    accumulate = False
    hist_h = frame.shape[1]

    #Person 1 histogram
    res1 = cv2.bitwise_and(frame,frame,mask = person1)
    bgr_planes1 = cv2.split(res1)
    b_hist1 = cv2.calcHist(bgr_planes1, [0], person1, [histSize], histRange, accumulate=accumulate)
    g_hist1 = cv2.calcHist(bgr_planes1, [1], person1, [histSize], histRange, accumulate=accumulate)
    r_hist1 = cv2.calcHist(bgr_planes1, [2], person1, [histSize], histRange, accumulate=accumulate)
    cv2.normalize(b_hist1, b_hist1, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist1, g_hist1, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist1, r_hist1, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    #Person 2 histogram
    res2 = cv2.bitwise_and(frame,frame,mask = person2)
    bgr_planes2 = cv2.split(res2)
    b_hist2 = cv2.calcHist(bgr_planes2, [0], person2, [histSize], histRange, accumulate=accumulate)
    g_hist2 = cv2.calcHist(bgr_planes2, [1], person2, [histSize], histRange, accumulate=accumulate)
    r_hist2 = cv2.calcHist(bgr_planes2, [2], person2, [histSize], histRange, accumulate=accumulate)
    cv2.normalize(b_hist2, b_hist2, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist2, g_hist2, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist2, r_hist2, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    #Person 3 histogram
    res3 = cv2.bitwise_and(frame,frame,mask = person3)
    bgr_planes3 = cv2.split(res3)
    b_hist3 = cv2.calcHist(bgr_planes3, [0], person3, [histSize], histRange, accumulate=accumulate)
    g_hist3 = cv2.calcHist(bgr_planes3, [1], person3, [histSize], histRange, accumulate=accumulate)
    r_hist3 = cv2.calcHist(bgr_planes3, [2], person3, [histSize], histRange, accumulate=accumulate)
    cv2.normalize(b_hist3, b_hist3, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist3, g_hist3, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist3, r_hist3, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    #Person 4 histogram
    res4 = cv2.bitwise_and(frame,frame,mask = person4)
    bgr_planes4 = cv2.split(res4)
    b_hist4 = cv2.calcHist(bgr_planes4, [0], person4, [histSize], histRange, accumulate=accumulate)
    g_hist4 = cv2.calcHist(bgr_planes4, [1], person4, [histSize], histRange, accumulate=accumulate)
    r_hist4 = cv2.calcHist(bgr_planes4, [2], person4, [histSize], histRange, accumulate=accumulate)
    cv2.normalize(b_hist4, b_hist4, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist4, g_hist4, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist4, r_hist4, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    person_ids = []
    person_ids.append([0])
    person_ids.append([1])
    person_ids.append([2])
    person_ids.append([3])
    if frame_counter == 0:
        #first frame create offline ones
        list_offline_histograms.append([b_hist1,g_hist1,r_hist1])
        list_offline_histograms.append([b_hist2,g_hist2,r_hist2])
        list_offline_histograms.append([b_hist3,g_hist3,r_hist3])
        list_offline_histograms.append([b_hist4,g_hist4,r_hist4])

        for i in range(len(labels)):
            for j in range(len(cluster_colors)):
                if labels[i] == cluster_colors[j][0]:
                    data.append(cluster_colors[j][1])
    else:
        #compare online with offline
        oldVal = 0
        newVal = 0
        newId = 0
        #online person 1
        for i in range(4):
            bVal = cv2.compareHist(b_hist1,list_offline_histograms[i][0],0)
            gVal = cv2.compareHist(g_hist1,list_offline_histograms[i][1],0)
            rVal = cv2.compareHist(r_hist1,list_offline_histograms[i][2],0)
            newVal = bVal+gVal+rVal
            if newVal > oldVal:
                oldVal = newVal
                newId = i
                #first hgistogram matches 3rd one offline
        #labels[0] == colors[newId]
        colors.append([[0],cluster_colors[newId][1]])
         #compare online with offline
        oldVal = 0
        newVal = 0
        newId = 0
        #online person 2
        for i in range(4):
            bVal = cv2.compareHist(b_hist2,list_offline_histograms[i][0],0)
            gVal = cv2.compareHist(g_hist2,list_offline_histograms[i][1],0)
            rVal = cv2.compareHist(r_hist2,list_offline_histograms[i][2],0)
            newVal = bVal+gVal+rVal
            if newVal > oldVal:
                oldVal = newVal
                newId = i
                #first hgistogram matches 3rd one offline
        #labels[1] == colors[newId]
        colors.append([[1],cluster_colors[newId][1]])
         #compare online with offline
        oldVal = 0
        newVal = 0
        newId = 0
        #online person 3
        for i in range(4):
            bVal = cv2.compareHist(b_hist3,list_offline_histograms[i][0],0)
            gVal = cv2.compareHist(g_hist3,list_offline_histograms[i][1],0)
            rVal = cv2.compareHist(r_hist3,list_offline_histograms[i][2],0)
            newVal = bVal+gVal+rVal
            if newVal > oldVal:
                oldVal = newVal
                newId = i
                #first hgistogram matches 3rd one offline
        #labels[2] == colors[newId]
        colors.append([[2],cluster_colors[newId][1]])
         #compare online with offline
        oldVal = 0
        newVal = 0
        newId = 0
        #online person 4
        for i in range(4):
            bVal = cv2.compareHist(b_hist4,list_offline_histograms[i][0],0)
            gVal = cv2.compareHist(g_hist4,list_offline_histograms[i][1],0)
            rVal = cv2.compareHist(r_hist4,list_offline_histograms[i][2],0)
            newVal = bVal+gVal+rVal
            if newVal > oldVal:
                oldVal = newVal
                newId = i
                #first hgistogram matches 3rd one offline
        #labels[3] == colors[newId]
        colors.append([[3],cluster_colors[newId][1]])
        cv2.imshow('p1', res1)
        cv2.imshow('p2', res2)
        cv2.imshow('p3', res3)
        cv2.imshow('p4', res4)


    
        for i in range(len(labels)):
            for j in range(len(colors)):
                if labels[i] == colors[j][0]:
                    data.append(colors[j][1])
    return data

def set_voxel_positions(width, height, depth):
    global background_check_total_time, voxel_check_total_time, voxel_vis_total_time, frame_counter, set_voxel_positions_total_time
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