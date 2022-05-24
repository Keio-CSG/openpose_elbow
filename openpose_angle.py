import os
import cv2
import copy
import numpy as np
import pyrealsense2 as rs
from elbow_func import util_elbow
from src import util
from src.body import Body


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
            ind_test = util_elbow.get_arm_index(subset)
            xy = util_elbow.get_xy(ind_test, candidate)
            canvas = util_elbow.draw_armpose(color_image_s, xy)
            if len(xy) >= 1:
                world_XYZ = util_elbow.pixel_to_world_XYZ(
                    intr, xy, depth_image)
                angle_deg = util_elbow.calculate_angle(world_XYZ)
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
