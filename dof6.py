"""Python 3.7, OpenCV 4.1.1"""
import os
import sys
import math
import argparse

import numpy as np
import cv2
from cv2 import FileStorage

from six_dof_functions import *

parser = argparse.ArgumentParser('Input video, .yml and .ply file names in'
                                 ' same directory or absolute paths '
                                 '\nWhile video is playing: '
                                 '\n Press ESC to exit'
                                 '\n Press s to pause or resume\n'
                                 )
parser.add_argument('--video',type=str,help='Input video file',
                    default='box.mp4')
parser.add_argument('--yml',type=str,help='Input yml file',
                    default='cookies_ORB.yml')
parser.add_argument('--ply',type=str,help='Input ply file',
                     default='box.ply')

# Camera calibration matrix
FOCALLENGTH = 55.
SENSOR_X = 22.3
SENSOR_Y = 14.9
WIDTH = 640
HEIGHT = 480
calibration_matrix = np.zeros((3,3),dtype=np.float64)
calibration_matrix[0,0] = WIDTH*FOCALLENGTH / SENSOR_X #FX
calibration_matrix[1,1] = HEIGHT*FOCALLENGTH / SENSOR_Y #FY
calibration_matrix[0,2] = WIDTH/2 #CX
calibration_matrix[1,2] = HEIGHT/2 #CY
calibration_matrix[2,2] = 1 

# Ratio test
RATIO_TEST_THRESHOLD = 0.8

# SolvePnpRansac parameters
iterationsCount = 500 
reprojectionError = 6.0 #2.0
confidence = 0.99 #0.95

# Kalman Filter initialization parameters
KF_N_STATES = 18
KF_N_MEASUREMENTS = 6
KF_N_INPUTS = 0
KF_DT = 0.125

#Inlier condition for triggering usage of measured and update KF with measured.
MIN_INLIERS_KALMAN = 30
KF = initKalmanFilter(KF_N_STATES,KF_N_MEASUREMENTS,KF_N_INPUTS,KF_DT)

# Create matcher
"""
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
"""
matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING)
# Feature detector
orb = cv2.ORB_create(nfeatures=2000)

def image_2_6dof(color_img,model_points3d,model_descriptors,
                 mesh_vertices,mesh_triangles):
    """ Input: Raw color image
        Output: Color image with mesh and axes drawn
    """
    # Get ORB features and descriptors from input img.
    img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    img_keypoints = orb.detect(img)
    img_keypoints, img_descriptors = orb.compute(img,img_keypoints)
    
    # Match input img ORB descriptors with model descriptors.
    match12 = matcher.knnMatch(img_descriptors,model_descriptors,2)
    match21 = matcher.knnMatch(model_descriptors,img_descriptors,2)
    
    # This is the block from robustMatch, fast version is not implemented.
    removed1 = ratiotest(match12,RATIO_TEST_THRESHOLD)
    removed2 = ratiotest(match21,RATIO_TEST_THRESHOLD)
    good_matches = symmetry_test(removed1,removed2)
    for row in removed1:
        good_matches.append([row[0].queryIdx,row[0].trainIdx,row[0].distance])    
    
    points3d_model_match = []
    points2d_scene_match = []

    for x in range(len(good_matches)):
        point3d_model =  model_points3d[good_matches[x][1]] #1 for trainidx
        points3d_model_match.append(point3d_model)
        
        points2d_scene = img_keypoints[good_matches[x][0]].pt #0 for queryidx
        points2d_scene_match.append(points2d_scene)
    
    draw_2d_points(color_img,points2d_scene_match,(0,0,200))
    
    # Pose estimation with PnP and Ransac, main function: cv2.solvePnPRansac.
    measurements = np.zeros((KF_N_MEASUREMENTS,1))
    good_measurement = False
    projection_to_use = None
    color = None
    list_2dpoints_inliers = []

    if len(good_matches) >=4:
        rotation_matrix,translation_matrix,projection_matrix,inliers, retval\
        = estimatePoseRANSAC(np.array(points3d_model_match), 
                             np.array(points2d_scene_match),
                             calibration_matrix,
                             cv2.SOLVEPNP_EPNP, 
                             np.empty(0), iterationsCount, 
                             reprojectionError,confidence)
                                                                             
        if inliers.shape[0]>0 and retval:
            for inlier in inliers:
                list_2dpoints_inliers.append(points2d_scene_match[inlier.item()])
            draw_2d_points(color_img,list_2dpoints_inliers,(200,0,0))
            
            if len(inliers) >=MIN_INLIERS_KALMAN:
                measurements = fillMeasurements(translation_matrix,
                                                rotation_matrix,
                                                KF_N_MEASUREMENTS)
                good_measurement = True
        translation_estimated, rotation_estimated = update_KF(KF,measurements)
        
        if not good_measurement:
            """ If this block executes before KF is ever updated with nonzero 
                measurements,calling drawobjectMesh might result in script 
                crashing due to division by zero inside backproject3dPoint 
                at the normalization step.
            """
            projection_to_use = get_projection_matrix(rotation_estimated,
                                                      translation_estimated)
            color = (255,0,0)
        else:
            projection_to_use = projection_matrix
            color = (0,255,0)
        # Draw object mesh based on selected projection matrix.
        drawObjectMesh(color_img,mesh_vertices,mesh_triangles,
                       calibration_matrix,projection_to_use,color)        
        # Draw the cartesian axes arrows on the predicted pose.
        draw_3d_axes(color_img,projection_to_use,calibration_matrix)

    # Cannot use cv2.solvePnPRansac if less than 4 in good_matches, just skip.
    else:
        pass
    return color_img


def factory(video_file,model_points3d,model_descriptors,
            mesh_vertices,mesh_triangles):
    """ Main function, handles the non image processing stuff.
    """
    video = cv2.VideoCapture(video_file)
    while (video.isOpened()):
        ret,frame = video.read()
        if not ret:
            break
        dof_img = image_2_6dof(frame,model_points3d,model_descriptors,
                                mesh_vertices,mesh_triangles)
        cv2.imshow('6 DOF',dof_img)
        key = cv2.waitKey(100)
        if key == 27: #Esc key
            cv2.destroyAllWindows()
            print("Terminating video")
            sys.exit()
        elif key == ord('s'):
            print("Pausing video")
            while True:
                key2 = cv2.waitKey(1)
                if key2 == ord('s'): 
                    print("Resuming video")
                    break
                if key2 == 27:
                    cv2.destroyAllWindows()
                    print("Terminating video")
                    sys.exit()
                else:
                    continue
    print("End of video")
    

if __name__ == '__main__':
    args = parser.parse_args()
    video_file = args.video
    yml_file = args.yml
    ply_file = args.ply
    
    # Read yml file, FileStorage class documentation link below
    # https://docs.opencv.org/master/da/d56/classcv_1_1FileStorage.html
    fs_storage = FileStorage(yml_file,0)  
    model_points3d = fs_storage.getNode('points_3d').mat()
    model_descriptors = fs_storage.getNode('descriptors').mat() 
    # Read from ply file
    mesh_vertices,mesh_triangles = read_ply(ply_file)
    
    factory(video_file,model_points3d,model_descriptors,
            mesh_vertices,mesh_triangles)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
