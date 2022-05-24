import numpy as np
import cv2
import pyrealsense2 as rs
import sys
import os
import json


def get_arm_index(subset, left=False):
    p_num = len(subset)
    ind = [[0]*3 for _ in range(p_num)]
    if not left:
        for i in range(2, 5):
            for j in range(p_num):
                ind[j][i-2] = int(subset[j][i])
    else:
        for i in range(5, 8):
            for j in range(p_num):
                ind[j][i-2] = int(subset[j][i])
    for i in range(p_num):
        if i == 0:
            correction_i = 0
        if -1 in ind[i-correction_i]:
            del ind[i-correction_i]
            correction_i += 1
    return ind


def get_xy(index, candidate):
    p_num = len(index)
    xy_list = [[[-1]*2 for _ in range(3)] for _ in range(p_num)]
    for i in range(p_num):
        for j in range(3):
            xy_list[i][j][0:2] = candidate[index[i][j]][0:2]
    return xy_list


def pixel_to_world_XYZ(intrinsics, xy_list, depth_image):
    p_num = len(xy_list)
    world_XYZ_list = [[[-1]*3 for _ in range(3)] for _ in range(p_num)]

    for i in range(p_num):
        for j in range(3):
            xy = xy_list[i][j][0:2]
            depth = depth_image[int(xy[1])][int(xy[0])]
            world_XYZ_list[i][j][0:3] = rs.rs2_deproject_pixel_to_point(
                intrinsics, [int(xy[0]), int(xy[1])], depth)
    return world_XYZ_list


def calculate_angle(world_XYZ):
    p_shoulder = np.array(world_XYZ[0][0])
    p_elbow = np.array(world_XYZ[0][1])
    p_wrist = np.array(world_XYZ[0][2])
    upperarm = p_shoulder - p_elbow
    forearm = p_wrist - p_elbow
    length_vec_upperarm = np.linalg.norm(upperarm)
    length_vec_forearm = np.linalg.norm(forearm)
    inner_product = np.inner(upperarm, forearm)
    angle_rad = np.arccos(
        inner_product / (length_vec_upperarm * length_vec_forearm))
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def draw_armpose(canvas, xy):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [
                  0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    p_num = len(xy)
    # 0番目の人(推定してる人のみの点を表示)
    for j in range(3):
        x, y = xy[0][j][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[j+2], thickness=-1)
    return canvas


    # 複数人数の点も描画
""" for i in range(p_num):
        for j in range(3):
            x, y = xy[i][j][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[j+2], thickness=-1) """


def load_data():
    # json fileの読み込み部分
    args = sys.argv
    json_path = args[1]
    dir = os.path.split(os.path.abspath(json_path))[0]
    with open(json_path) as f:
        config = json.load(f)
        print(config)
    color_path = os.path.join(dir, config["color_file"])
    depth_path = os.path.join(dir, config["depth_file"])
    frequency = config["frequency"]

    # データの読み込み
    video = cv2.VideoCapture(color_path)
    depth_frames = np.load(depth_path)
    depth_frames = depth_frames[depth_frames.files[0]]
    return video, depth_frames, frequency
