import cv2
import glm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from engine.config import config

from skimage import measure
from skimage.draw import ellipsoid
from scipy.ndimage.interpolation import zoom


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
frame_counter = 0
captures = {
    "cam1": {
        "cap": cv2.VideoCapture('ass2/data/cam1/video.avi'),
        "background": cv2.imread('ass2/data/cam1/background.jpg'),
    },
    "cam2": {
        "cap": cv2.VideoCapture('ass2/data/cam2/video.avi'),
        "background": cv2.imread('ass2/data/cam2/background.jpg'),
    },
    "cam3": {
        "cap": cv2.VideoCapture('ass2/data/cam3/video.avi'),
        "background": cv2.imread('ass2/data/cam3/background.jpg'),
    },
    "cam4": {
        "cap": cv2.VideoCapture('ass2/data/cam4/video.avi'),
        "background": cv2.imread('ass2/data/cam4/background.jpg'),
    },
}



def construct_voxel_space(step = 32, voxel_space_half_size = 1000):
    """Generates the voxel space and stores it in a global variable.
    The voxel space is a list of 3D points that represent the center of each voxel.
    The voxel space is generated in the range [-voxel_space_half_size, voxel_space_half_size] in each dimension.
    The step parameter controls the distance between voxels in the voxel space.
    
    Args:
        step (int, optional): Distance between voxels in the voxel space. Defaults to 32.
        voxel_space_half_size (int, optional): Half the size of the voxel space in each dimension. Defaults to 1000.
        
    Returns:
        list: A list of 3D points that represent the center of each voxel.
    """
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
    """Checks the visibility of each voxel in the voxel space.
    
    Returns:
        data {list}: A list of 3D points that represent the center of each visible voxel.
        colors {list}: A list of RGB colors that represent the color of each visible voxel.
    """
    global frame_counter
    data = []
    colors = []
    true_foregrounds = {}
    frames = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subtract_background, 'cam'+str(cam)) for cam in range(1,5)]
        for future in as_completed(futures):
            foreground = future.result()[0]
            frame = future.result()[1]
            cam_id = future.result()[2]
            true_foregrounds[cam_id] = foreground
            frames[cam_id] = frame
        
    for i, voxel in enumerate(list_voxels):
        camera_counter = 0
        rgb_col = []
        for j in range(1,5):
            if listout_of_bounds[i][j-1] == True:
                break
            true_foreground = true_foregrounds['cam'+str(j)]
            frame = frames['cam'+str(j)]
            color = true_foreground[list_list_points[i][j-1][1],list_list_points[i][j-1][0]]
            rgb_col.append(frame[list_list_points[i][j-1][1],list_list_points[i][j-1][0]])
            if color == 255:
                camera_counter += 1
            else:
                break
        if camera_counter == 4:
            data.append(voxel)
            temp_rgb = np.float32([0.0,0.0,0.0])
            for i in range(4):
                temp_rgb[0] += rgb_col[i][0]
                temp_rgb[1] += rgb_col[i][1]
                temp_rgb[2] += rgb_col[i][2]
            temp_rgb[0] = (temp_rgb[0] /4)/255
            temp_rgb[1] = (temp_rgb[1] /4)/255
            temp_rgb[2] = (temp_rgb[2] /4)/255
            colors.append(temp_rgb)
    frame_counter  += 1
    print(f"{frame_counter}")
    return data, colors
            

def set_voxel_positions(width, height, depth):
    """Sets the position of the voxels in the voxel space.

    Args:
        width (int): The width of the voxel space.
        height (int): The height of the voxel space.
        depth (int): The depth of the voxel space.
    
    Returns:
        data {list}: A list of 3D points that represent the center of each voxel.
        colors {list}: A list of RGB colors that represent the color of each voxel.
    """
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
    """Gets the position of the camera in the world coordinates.
    
    Args:
        rvecs {list}: A list of rotation vectors.
        tvecs {list}: A list of translation vectors.
        
    Returns:
        cameraPosition {list}: A list of 3D points that represent the position of the camera in the world coordinates.
    """
    rotM, j = cv2.Rodrigues(rvecs)
    cameraPosition = -np.matrix(rotM).transpose() * np.matrix(tvecs)
    #OpenCV Y down, Z forward meanwhile OpenGL uses Y for up so swap it
    #Coordinates converted to meters
    #Swap sign for up since opencv uses -Z
    return [cameraPosition[0]/scalling_factor,-cameraPosition[2]/scalling_factor,cameraPosition[1]/scalling_factor]


def get_cam_positions():
    """Gets the position of all cameras in the world coordinates.

    Returns:
        [list]: A list of 3D points that represent the position of all cameras in the world coordinates.
    """

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
    """Gets the extrinsic matrix of the camera.
    
    Args:
        rvecs {list}: A list of rotation vectors.
        tvecs {list}: A list of translation vectors.
    
    Returns:
        m44eye {list}: A 4x4 matrix that represents the extrinsic matrix of the camera.
    """
    rv = rvecs.ravel()
    m44eye = np.identity(4)
    rotM, j = cv2.Rodrigues(rv)
    m44eye[:3, :3] = rotM
    return m44eye

def get_cam_rotation_matrices():
    """Gets the rotation matrices of all cameras.

    Returns:
        [list]: A list of 4x4 matrices that represent the rotation matrices of all cameras.
    """
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
    """Gets the background image of the video clip.
    
    Keyword Arguments:
        path {str} -- The path to the video clip (default: {'ass2/data/cam1/'}).
        
    Returns:
        average_image {ndarray} -- The background image.
    """
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
    """Generates a surface mesh of the currently displayed image."""
    print('Generating mesh')

    voxels, _ = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'])

    min_x = min(voxel[0] for voxel in voxels)
    min_y = min(voxel[1] for voxel in voxels)
    min_z = min(voxel[2] for voxel in voxels)

    shifted_voxels = [(voxel[0]-min_x, voxel[1]-min_y, voxel[2]-min_z) for voxel in voxels]

    x_size = max(voxel[0] for voxel in shifted_voxels) + 1
    y_size = max(voxel[1] for voxel in shifted_voxels) + 1
    z_size = max(voxel[2] for voxel in shifted_voxels) + 1
    data = np.zeros((x_size, z_size, y_size), dtype=np.uint8)

    for voxel in shifted_voxels:
        data[int(voxel[0]), int(voxel[2]), int(voxel[1])] = 1

    data = zoom(data, 1.5)

    #data = np.reshape(voxels,(3,1))
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(data,0, spacing=(1,1,1))

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, config['world_width'])
    ax.set_zlim(0, config['world_height'])
    ax.set_ylim(0, config['world_width'])

    plt.tight_layout()
    plt.show()
    print('Done generating mesh')
    return True


def get_mask(frame, background_image):
    """Returns a mask of the frame with the background subtracted
    
    Arguments:
        frame {np.array} -- The frame to be processed
        background_image {np.array} -- The background image

    Returns:
        [np.array] -- The mask of the frame
    """
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

    return foreground


def subtract_background(camera_id = 'cam1'):
    """Subtracts the background from the video and saves it to a file
    
    Keyword Arguments:
        camera_id {str} -- The camera id (default: {'cam1'})
    
    Returns:
        true_foreground {np.array} -- The mask of the frame
        frame {np.array} -- The frame to be processed
        camera_id {str} -- The camera id
    """

    global frame_counter, captures

    cap = captures[camera_id]["cap"]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    background_image = captures[camera_id]["background"]
    if  frame_counter > frame_count:
        frame_counter = 0
    cap.set(1,frame_counter)
    ret, frame = cap.read()
    if not ret:
        print("Video error")
        return None
    true_foreground = get_mask(frame, background_image)
    return true_foreground, frame, camera_id

#Load camera properties from folder directory
def load_camera_properties(camera_id = 'cam1'):
    """Loads the camera properties from the config.xml file
    
    Keyword Arguments:
        camera_id {str} -- The camera id (default: {'cam1'})
        
    Returns:
        camMatrix {np.array} -- The camera matrix
        d {np.array} -- The distortion coefficients
        rvecs {np.array} -- The rotation vectors
        tvecs {np.array} -- The translation vectors
    """
    path = 'ass2/data/' + camera_id + '/'
    s = cv2.FileStorage(path + "config.xml", cv2.FILE_STORAGE_READ)
    camMatrix = s.getNode('CameraMatrix').mat()
    d = s.getNode('DistortionCoeffs').mat()
    rvecs = s.getNode('RotationValues').mat()
    tvecs = s.getNode('TranslationValues').mat()
    return camMatrix, d, rvecs, tvecs

#Slider events
def on_low_H_thresh_trackbar(val):
    """Sets the low H threshhold
    
    Arguments:
        val {int} -- The value of the slider
    """
    global threshhold_h
    threshhold_h = val
    threshhold_h = min(255, threshhold_h)
    cv2.setTrackbarPos(H_name, window_bar_name, threshhold_h)
def on_low_S_thresh_trackbar(val):
    """Sets the low S threshhold
    
    Arguments:
        val {int} -- The value of the slider
    """
    global threshhold_s
    threshhold_s = val
    threshhold_s = min(255, threshhold_s)
    cv2.setTrackbarPos(S_name, window_bar_name, threshhold_s)
def on_low_V_thresh_trackbar(val):
    """Sets the low V threshhold
    
    Arguments:
        val {int} -- The value of the slider
    """
    global threshhold_v
    threshhold_v = val
    threshhold_v = min(255, threshhold_v)
    cv2.setTrackbarPos(V_name, window_bar_name, threshhold_v)