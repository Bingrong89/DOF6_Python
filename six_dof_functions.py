import math
import numpy as np
import cv2

def read_ply(plyfilepath):
    # Open the ply file and extract relevant data.
    vertex_list = []
    triangle_list = []
    ply_obj = open(plyfilepath,'r')
    end_header = False    
    vertex_count = 0
    for line in ply_obj:
        if end_header == False:
            if 'element vertex' in line:
                vertex_total = line.strip().split(' ')[-1]
                assert vertex_total.isdigit(),\
                'vertex_count value not positive integer!'             
            if 'element face' in line:
                triangle_total = line.strip().split(' ')[-1]
                assert triangle_total.isdigit(),\
                'triangle_count value not integer!'             
            if 'end_header' in line:
                end_header = True
        
        elif end_header == True:
            if vertex_count != int(vertex_total):
                vertex_list.append(line.strip().split())
                vertex_count +=1
            else:
                triangle_list.append(line.strip().split()[1:])  
    ply_obj.close()    
    return vertex_list, triangle_list
    
    
def ratiotest(knnmatch_output,threshold=0.8):
    newlist = []
    for pair in knnmatch_output:
        if len(pair) !=2:
            continue
        dist1 = pair[0].distance
        dist2 = pair[1].distance        
        ratio = dist1/dist2        
        if ratio < threshold:
            newlist.append(pair) 
    return newlist


def symmetry_test(array1,array2):
    symmatch = []
    for pair1 in array1:
        for pair2 in array2:
            if (pair1[0].queryIdx == pair2[0].trainIdx) and\
               (pair2[0].queryIdx == pair1[0].trainIdx):
                symmatch.append([pair1[0].queryIdx, 
                                 pair1[0].trainIdx, pair1[0].distance])
                break
    return symmatch               


def draw_2d_points(img,points_list,color):
    for point in points_list:
        cv2.circle(img,(int(point[0]),int(point[1])),4,color,-1,8)


def get_projection_matrix(rotation_matrix,translation_matrix):
    projection_matrix = np.zeros((3,4)) 
    projection_matrix[:3,:3] = rotation_matrix
    projection_matrix[:,3] = translation_matrix[:,0] 
    return projection_matrix


def estimatePoseRANSAC(list_points3d,list_points2d,calibration_matrix,flags,
                       inliers,iterationscount,reprojectionerror,confidence):
    distCoeffs = np.zeros((4,1))
    output_rot_vec = np.zeros((3,1))
    output_trans_vec = np.zeros((3,1))
    useExtrinsicguess = False
    
    # SolvePnPRansac documentation link below
    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
    retval, output_rot_vec, output_trans_vec, inliers =\
    cv2.solvePnPRansac(list_points3d,list_points2d,calibration_matrix,None,
                       flags=flags,iterationsCount=iterationscount)
                       
    rotation_matrix = np.zeros((3,3))
    rotation_matrix,_ = cv2.Rodrigues(output_rot_vec,rotation_matrix)
    translation_matrix = output_trans_vec  
    projection_matrix = get_projection_matrix(rotation_matrix,
                                              translation_matrix)
    if inliers is None:
        inliers = np.empty(0)
    
    return rotation_matrix, translation_matrix, \
           projection_matrix, inliers, retval


def backproject3dPoint(point3d,calibration_matrix,projection_matrix): 
    """
    Explictly use np.matmul to ensure conventional matrix multiplication
    even if python lists are used to represent the arrays.
    """
    step_one = np.matmul(calibration_matrix,projection_matrix) # Shape(3,4)
    step_two = np.matmul(step_one,point3d) # Shape(3,1)
    # Normalization
    x_coord = step_two[0] / step_two[2]
    y_coord = step_two[1] / step_two[2] 
    return (x_coord,y_coord)


def initKalmanFilter(n_states,n_measurements,n_inputs,dt):
    KF = cv2.KalmanFilter(n_states,n_measurements,n_inputs,cv2.CV_64F)
    KF.processNoiseCov = cv2.setIdentity(KF.processNoiseCov,1e-5)
    KF.measurementNoiseCov = cv2.setIdentity(KF.measurementNoiseCov,1e-2)
    KF.errorCovPost = cv2.setIdentity(KF.errorCovPost,1)
    dt_2 = 0.5*(dt**2)
    """ 
    Problem: Assigning directly to elements in KF.transitionMatrix 
    and KF.measurementMatrix through numpy indexing does not actually
    change the underlying data when calling them.
    
    Quick solution: Reassign the variable directly to a new array object.       
    """
    tempTM = KF.transitionMatrix
    tempMM = KF.measurementMatrix
    
    # Position
    tempTM[0,3] = dt
    tempTM[1,4] = dt
    tempTM[2,5] = dt
    tempTM[3,6] = dt
    tempTM[4,7] = dt
    tempTM[5,8] = dt
    tempTM[0,6] = dt_2
    tempTM[1,7] = dt_2
    tempTM[2,8] = dt_2
    
    # Orientation
    tempTM[9,12] = dt
    tempTM[10,13] = dt
    tempTM[11,14] = dt
    tempTM[12,15] = dt
    tempTM[13,16] = dt
    tempTM[14,17] = dt
    tempTM[9,15] = dt_2
    tempTM[10,16] = dt_2
    tempTM[11,17] = dt_2
    
    tempMM[0,0] = 1 # X
    tempMM[1,1] = 1 # Y
    tempMM[2,2] = 1 # Z
    tempMM[3,9] = 1 # Roll
    tempMM[4,10] = 1 # Pitch
    tempMM[5,11] = 1 # Yaw
    
    KF.transitionMatrix = tempTM
    KF.measurementMatrix = tempMM   
    return KF


def update_KF(KF,measurement):
    # Update Kalman filter with good measurements.
    prediction = KF.predict()
    estimated = KF.correct(measurement)
    
    translation_estimated = np.empty((3,1),dtype=np.float64)
    translation_estimated[0] = estimated[0]
    translation_estimated[1] = estimated[1]
    translation_estimated[2] = estimated[2]
    
    eulers_estimated = np.empty((3,1),dtype=np.float64)
    eulers_estimated[0] = estimated[9]
    eulers_estimated[1] = estimated[10]
    eulers_estimated[2] = estimated[11]
    
    rotation_estimated = euler2rot(eulers_estimated)   
    return translation_estimated,rotation_estimated


def rot2euler(rot_mat):
    # From Utils.cpp, cv::Mat rot2euler(const cv::Mat & rotationMatrix).
    euler = np.empty((3,1),dtype=np.float64)
    
    m00 = rot_mat[0,0]
    m02 = rot_mat[0,2]
    m10 = rot_mat[1,0]
    m11 = rot_mat[1,1]
    m12 = rot_mat[1,2]
    m20 = rot_mat[2,0]
    m22 = rot_mat[2,2]  
    bank = None
    attitude=  None
    heading = None
    
    if m10 > 0.998:
        bank = 0
        attitude = math.pi/2
        heading = math.atan2(m02,m22)
    
    elif m10 <-0.998:
        bank = 0
        attitude = -math.pi/2
        heading = math.atan2(m02,m22)
    
    else:
        bank = math.atan2(-m12,m11)
        attitude = math.asin(m10)
        heading = math.atan2(-m20,m00)
        
    euler[0] = bank
    euler[1] = attitude
    euler[2] = heading     
    return euler    


def euler2rot(euler):
    # From Utils.cpp, cv::Mat euler2rot(const cv::Mat & euler).
    rot_mat = np.empty((3,3),dtype=np.float64)
    bank = euler[0]
    attitude = euler[1]
    heading = euler[2]
    
    ch = math.cos(heading)
    sh = math.sin(heading)
    ca = math.cos(attitude)
    sa = math.sin(attitude)
    cb = math.cos(bank)
    sb = math.sin(bank)
    
    m00 = ch * ca
    m01 = (sh*sb) - (ch*sa*cb)
    m02 = (ch*sa*sb) + (sh*cb)
    m10 = sa
    m11 = ca * cb
    m12 = -ca * sb
    m20 = -sh * ca
    m21 = (sh*sa*cb) + (ch*sb)
    m22 = -(sh*sa*sb) + (ch*cb)
    
    rot_mat[0,0] = m00
    rot_mat[0,1] = m01
    rot_mat[0,2] = m02
    rot_mat[1,0] = m10
    rot_mat[1,1] = m11
    rot_mat[1,2] = m12
    rot_mat[2,0] = m20
    rot_mat[2,1] = m21
    rot_mat[2,2] = m22    
    return rot_mat   


def fillMeasurements(translation_matrix,rotation_matrix,n_measurements):
    """ Accurately speaking, it is creating here, not filling.
    Leaving the name as is to make it easier to trace back to the
    original C++ implementation.  
    """
    measurements = np.zeros((n_measurements,1),dtype=np.float64)
    measured_eulers = rot2euler(rotation_matrix)
    
    measurements[0] = translation_matrix[0] # X
    measurements[1] = translation_matrix[1] # Y
    measurements[2] = translation_matrix[2] # Z
    measurements[3] = measured_eulers[0] # Roll
    measurements[4] = measured_eulers[1] # Pitch
    measurements[5] = measured_eulers[2] # Yaw
    
    return measurements 


def drawObjectMesh(img, vertices,triangles,calib_mat,
                   proj_mat,col=(255,100,100)):
    for triangle in triangles:   
        point_3d_0 = vertices[int(triangle[0])][:]
        point_3d_1 = vertices[int(triangle[1])][:]
        point_3d_2 = vertices[int(triangle[2])][:]
        
        # PnpProblem.cpp line 171, appending string here since triangles
        # are loaded as arrays of strings representing integers.
        point_3d_0.append('1')
        point_3d_1.append('1')
        point_3d_2.append('1')
        
        point_3d_0 = np.array(point_3d_0,dtype='float').reshape(4,1)
        point_3d_1 = np.array(point_3d_1,dtype='float').reshape(4,1)
        point_3d_2 = np.array(point_3d_2,dtype='float').reshape(4,1)
        
        point_2d_0 = backproject3dPoint(point_3d_0,calib_mat,proj_mat)
        point_2d_1 = backproject3dPoint(point_3d_1,calib_mat,proj_mat)
        point_2d_2 = backproject3dPoint(point_3d_2,calib_mat,proj_mat)

        # cv2.line changes the image object by reference.     
        _ = cv2.line(img,point_2d_0,point_2d_1,col,2) 
        _ = cv2.line(img,point_2d_1,point_2d_2,col,2) 
        _ = cv2.line(img,point_2d_2,point_2d_0,col,2)
 

def drawArrow(img, tail, head, color, arrow_size,thickness=20):
    cv2.line(img,tail,head,color,thickness)
    # Indices 0 1 2 represent Axes X Y Z respectively.
    angle = math.atan2(tail[1] - head[1] ,tail[0] - head[0])
    # First segment
    value_one = int(head[0] + (arrow_size*math.cos(angle + (math.pi/4))))
    value_two = int(head[1] + (arrow_size*math.sin(angle + (math.pi/4))))
    cv2.line(img,(value_one,value_two),head,color,thickness)
    
    # Second segment, same as first but sign flip inside the trigo argument.
    value_one = int(head[0] + (arrow_size*math.cos(angle - (math.pi/4))))
    value_two = int(head[1] + (arrow_size*math.sin(angle - (math.pi/4))))
    cv2.line(img,(value_one,value_two),head,color,thickness)


def draw_3d_axes(img,projection_to_use,calib_mat):
    X = 5
    origin = backproject3dPoint(np.array([[0],[0],[0],[1]]),
             calib_mat,projection_to_use)
    x_axis = backproject3dPoint(np.array([[X],[0],[0],[1]]),
             calib_mat,projection_to_use)
    y_axis = backproject3dPoint(np.array([[0],[X],[0],[1]]),
             calib_mat,projection_to_use)
    z_axis = backproject3dPoint(np.array([[0],[0],[X],[1]]),
             calib_mat,projection_to_use)
    
    red = (0,0,255)
    yellow = (0,255,255)
    blue = (255,0,0)
    black = (0,0,0)
    
    drawArrow(img,origin,x_axis,red,9,2)
    drawArrow(img,origin,y_axis,yellow,9,2)
    drawArrow(img,origin,z_axis,blue,9,2)
    cv2.circle(img,origin,2,black,-1,8)    
