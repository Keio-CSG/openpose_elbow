from src.body import Body
from src import util
from elbow_func import util_elbow
import pyrealsense2 as rs
import numpy as np
import os
import cv2
import copy
import time
import sys
import json


#INPUT_FILE_NAME = "image.png"

if __name__ == "__main__":
    # モデルの読み込みとデータの取得
    video, depth_frames, frequency = util_elbow.load_data()
    body_estimation = Body('model/body_pose_model.pth')

    # 変数類の定義
    frame_count = 0
    angles = []
    flag1 = False
    counter1 = 0

    past_frame_time = time.time()
    sec_per_frame = 1.0 / frequency

    # 本研究室のD435のintrinsicsを利用
    intr = rs.pyrealsense2.intrinsics()
    intr.width = 640
    intr.height = 360
    intr.ppx = 321.2279968261719
    intr.ppy = 176.7667999267578
    intr.fx = 318.4465637207031
    intr.fy = 318.4465637207031
    intr.model = rs.pyrealsense2.distortion.brown_conrady
    intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    # ループ部　推定を含む
    try:
        while True:
            ret, color_frame = video.read()
            depth_frame = depth_frames[frame_count, :, :]

            color_image_s = cv2.resize(color_frame, (640, 360))

            current_time = time.time()
            time.sleep(max(0, sec_per_frame - (current_time - past_frame_time)))
            past_frame_time = current_time

            candidate, subset = body_estimation(color_image_s)
            ind_test = util_elbow.get_arm_index(subset)
            xy = util_elbow.get_xy(ind_test, candidate)
            canvas = util_elbow.draw_armpose(color_image_s, xy)
            if len(xy) >= 1:
                world_XYZ = util_elbow.pixel_to_world_XYZ(
                    intr, xy, depth_frame)
                angle_deg = util_elbow.calculate_angle(world_XYZ)
                if flag1:
                    if angle_deg <= angles[-1] + 45 and angle_deg >= angles[-1] - 45:
                        angles.append(angle_deg)
                else:
                    angles.append(angle_deg)
                counter1 += 1
                if counter1 >= 5:
                    flag1 = True
                cv2.putText(canvas, f"min: {int(min(angles))} deg",
                            (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
                cv2.putText(canvas, f"max: {int(max(angles))} deg",
                            (410, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
                cv2.putText(canvas, f"{int(angle_deg)} degree",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
            cv2.imshow("frame", canvas)

            k = cv2.waitKey(1)
            if k & 0xff == 27:  # ESCで終了
                cv2.destroyAllWindows()
                break

            frame_count += 1
            if frame_count >= depth_frames.shape[0]:
                cv2.destroyAllWindows()
                break
    finally:
        print(int(min(angles)), int(max(angles)))
        video.release()
