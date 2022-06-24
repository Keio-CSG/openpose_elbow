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
import csv
import math


#INPUT_FILE_NAME = "image.png"

if __name__ == "__main__":
    # モデルの読み込みとデータの取得
    video, depth_frames, frequency, json_path = util_elbow.load_data()
    body_estimation = Body('model/body_pose_model.pth')

    # 変数類の定義
    frame_count = 0
    angles = []
    angles_csv = []
    shoulder_list = []
    elbow_list = []
    wrist_list = []
    flag1 = False
    counter1 = 1

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

    start_time = time.time()
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

                # CSV用listの生成
                shoulder_list.append(world_XYZ[0][0][0:3])
                shoulder_list[-1].insert(0, counter1)
                shoulder_list[-1].insert(1, "shoulder")
                elbow_list.append(world_XYZ[0][1][0:3])
                elbow_list[-1].insert(0, counter1)
                elbow_list[-1].insert(1, "elbow")
                wrist_list.append(world_XYZ[0][2][0:3])
                wrist_list[-1].insert(0, counter1)
                wrist_list[-1].insert(1, "wrist")

                angle_deg = util_elbow.calculate_angle_360(world_XYZ)
                angles_csv.append([counter1, angle_deg])
                if flag1:
                    previous_angle = (angles[-1] + angles[-2]) / 2
                    if angle_deg <= previous_angle + 45 and angle_deg >= previous_angle - 45:
                        angles.append(angle_deg)
                else:
                    angles.append(angle_deg)
                counter1 += 1
                if counter1 >= 6:
                    flag1 = True
                cv2.putText(canvas, f"min: {int(min(angles))} deg",
                            (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
                cv2.putText(canvas, f"max: {int(max(angles))} deg",
                            (410, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
                if not math.isnan(angle_deg):
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
        end_time = time.time()
        json_name = os.path.splitext(os.path.basename(json_path))[0]

        if not os.path.exists(f"./results"):
            os.mkdir(f"./results")
        if not os.path.exists(f"./results/{json_name}_openpose_360"):
            os.mkdir(f"./results/{json_name}_openpose_360")
        with open(f"./results/{json_name}_openpose_360/XYZ.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "part", "X", "Y", "Z"])
            writer.writerows(shoulder_list)
            writer.writerows(elbow_list)
            writer.writerows(wrist_list)
        with open(f"./results/{json_name}_openpose_360/angles.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "angle"])
            writer.writerows(angles_csv)

        os.system("cls")
        measurement_time = end_time - start_time
        print("Completed")
        print("------------------------------------")
        print(json_path)
        print(f"min:{int(min(angles))}")
        print(f"max:{int(max(angles))}")
        print(f"time:{measurement_time}")
        video.release()
