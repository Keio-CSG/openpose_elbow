import os
import cv2
import copy
import numpy as np
import pyrealsense2 as rs
from src import util
from src.body import Body


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
        if len(ind[i]) >= 1:
            if -1 in ind[i]:
                del ind[i]

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
    for i in range(p_num):
        for j in range(3):
            x, y = xy[i][j][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[j+2], thickness=-1)
    return canvas


#INPUT_FILE_NAME = "image.png"

if __name__ == "__main__":

    body_estimation = Body('model/body_pose_model.pth')
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    pro = profile.get_stream(rs.stream.depth)
    intr = pro.as_video_stream_profile().get_intrinsics()

    align_to = rs.stream.color
    align = rs.align(align_to)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    try:
        while(True):
            #ret, frame = cap.read()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            color_image_s = cv2.resize(color_image, (640, 360))

            candidate, subset = body_estimation(color_image_s)
            ind_test = get_arm_index(subset)
            xy = get_xy(ind_test, candidate)
            canvas = draw_armpose(color_image_s, xy)
            if len(xy) >= 1:
                world_XYZ = pixel_to_world_XYZ(intr, xy, depth_image)
                angle_deg = calculate_angle(world_XYZ)
                cv2.putText(canvas, f"{int(angle_deg)} degree",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
            cv2.imshow("frame", canvas)

            if cv2.waitKey(1) & 0xff == 27:  # ESCで終了
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

    # target_image_path = 'images/' + INPUT_FILE_NAME
    # oriImg = cv2.imread(target_image_path)  # B,G,R order
    #candidate, subset = body_estimation(oriImg)

    #canvas = copy.deepcopy(oriImg)
    #canvas = draw_armpose(canvas, xy)

    #basename_name = os.path.splitext(os.path.basename(target_image_path))[0]

    #result_image_path = "./test_result.jpg"
    #cv2.imwrite(result_image_path, canvas)
    # print("Complete")
