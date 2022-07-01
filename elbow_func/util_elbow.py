import numpy as np
import pandas as pd
import pyrealsense2 as rs
import cv2
import sys
import os
import json
import copy

import matplotlib.pyplot as plt


def get_arm_index(subset, left=False):
    """
    openposeのsubsetを基に腕の座標indexを取得する
    """
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
    """
    openposeのcandidateから腕の3点のxy座標を取得する
    """
    p_num = len(index)
    xy_list = [[[-1]*2 for _ in range(3)] for _ in range(p_num)]
    for i in range(p_num):
        for j in range(3):
            xy_list[i][j][0:2] = candidate[index[i][j]][0:2]
    return xy_list


def pixel_to_world_XYZ(intrinsics, xy_list, depth_image):
    """
    カメラの座標系から現実世界のXYZ座標に変換する
    """
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
    """
    3次元座標から角度を算出する
    """
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


def calculate_angle_360(world_XYZ):
    """
    3次元座標から360度で角度を算出する
    """
    p_shoulder = np.array(world_XYZ[0][0])
    p_elbow = np.array(world_XYZ[0][1])
    p_wrist = np.array(world_XYZ[0][2])
    upperarm = p_shoulder - p_elbow
    forearm = p_wrist - p_elbow
    cross = np.cross(upperarm, forearm)

    length_vec_upperarm = np.linalg.norm(upperarm)
    length_vec_forearm = np.linalg.norm(forearm)
    inner_product = np.inner(upperarm, forearm)
    angle_rad = np.arccos(
        inner_product / (length_vec_upperarm * length_vec_forearm))
    angle_deg = np.rad2deg(angle_rad)
    if cross[2] < 0:
        angle_deg = 360 - angle_deg
    return angle_deg


def draw_armpose(canvas, xy):
    """
    腕の位置にポイントを描画
    """
    points = []

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
    if p_num >= 1:
        for j in range(3):
            x, y = xy[0][j][0:2]
            points.append((x, y))
            cv2.circle(canvas, (int(x), int(y)), 4, colors[j+2], thickness=-1)
        for i in range(2):
            cv2.line(canvas, (int(points[i][0]), int(points[i][1])), (int(
                points[i+1][0]), int(points[i+1][1])), (0, 0, 0), thickness=1)

    return canvas

    # 複数人数の点も描画
""" for i in range(p_num):
        for j in range(3):
            x, y = xy[i][j][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[j+2], thickness=-1) """


def load_data():
    """
    各種ファイルからデータを取得する
    """
    # json fileの読み込み部分
    args = sys.argv
    json_path = args[1]
    dir = os.path.split(os.path.abspath(json_path))[0]
    with open(json_path) as f:
        config = json.load(f)
        # print(config)
    color_path = os.path.join(dir, config["color_file"])
    depth_path = os.path.join(dir, config["depth_file"])
    frequency = config["frequency"]

    # データの読み込み
    video = cv2.VideoCapture(color_path)
    depth_frames = np.load(depth_path)
    depth_frames = depth_frames[depth_frames.files[0]]
    return video, depth_frames, frequency, json_path


def elbow_distance_change_analysis(XYZ_file_path, angles_file_path, fluctuation=50):
    """
    csvファイルから肘の角度が±fluctuationまで変化した時の角度の変動をプロットし、最大・最小値等を表示する

    Parameters:
    ----------
    XYZ_file_path : str
        Estimated coordinate file path
    angles_file_path : str
        Estimated angles file path

    fluctuation : int
        Fluctuation range
    """
    XYZ = pd.read_csv(XYZ_file_path)
    angles = pd.read_csv(angles_file_path)

    # anglesから角度が最小、最大のframeを取得
    min_frame = angles.iat[angles.idxmin()["angle"], 0]
    max_frame = angles.iat[angles.idxmax()["angle"], 0]
    XYZ_min = XYZ[XYZ["frame"] == min_frame]
    XYZ_max = XYZ[XYZ["frame"] == max_frame]
    l_min = XYZ_min.values.tolist()
    l_max = XYZ_max.values.tolist()
    l = [l_min, l_max]
    m = ["min", "max"]
    for i in range(2):
        print(m[i])
        world_XYZ = [[]]
        angle_list = [[], []]
        for j in range(3):
            world_XYZ[0].append(l[i][j][2:5])
        for j in range(-(fluctuation), fluctuation + 1):
            new_world_XYZ = copy.deepcopy(world_XYZ)
            new_world_XYZ[0][1][2] = world_XYZ[0][1][2] + j
            result1 = calculate_angle_360(new_world_XYZ)
            angle_list[0].append(new_world_XYZ[0][1][2])
            angle_list[1].append(result1)
        plt.plot(angle_list[0], angle_list[1])
        plt.axvline(x=world_XYZ[0][1][2])
        plt.show()

        print(min(angle_list[1]), max(angle_list[1]), calculate_angle(
            world_XYZ), world_XYZ[0][1][2])
        print(
            "-----------------------------------------------------------------------------")
