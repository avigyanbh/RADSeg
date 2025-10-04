"""(Deprecated) Converts ZED X ros bag files to .npz numpy files.

This script is very rough. ROS2 bags only get RGB images extracted at the moment.
time synchronization is ignored completely and frames are roughly aligned by
publishing order.
"""

import argparse
import pathlib
from collections import defaultdict
import os

import numpy as np
import tqdm

def save_frames(frames_data, path):
  for k,v in frames_data.items():
    frames_data[k] = np.array(frames_data[k])
  np.savez_compressed(path, **frames_data)


def main(args=None):
  if args.bag.endswith(".txt"):
    with open(args.bag, "r") as f:
      bag_paths = [x.strip() for x in f.readlines() if not x.strip().startswith("#")]
    if not args.combine:
      os.makedirs(args.out, exist_ok=True)
  else:
    bag_paths = [args.bag]

  if args.version == 1:
    import rospy
    import rosbag
    from cv_bridge import CvBridge


    bridge = CvBridge()
    topic_list = ["/zedx/zed_node/disparity/disparity_image",
                  "/zedx/zed_node/left/image_rect_color", 
                  "/zedx/zed_node/pose",
                  "/zedx/zed_node/left/camera_info"]

    frames_data = defaultdict(list)

    for bag_path in tqdm.tqdm(bag_paths):
      print(f"Loading {bag_path}")
      bag = rosbag.Bag(bag_path, "r")
      print(f"Loaded {bag_path}")
      for topic, msg, t in bag.read_messages(topics=topic_list):
        if topic == "/zedx/zed_node/disparity/disparity_image":
          disparity_np = bridge.imgmsg_to_cv2(msg.image, desired_encoding="32FC1")
          frames_data["disparity_img"].append(disparity_np)
          frames_data["focal_length"].append(msg.f)
          frames_data["stereo_baseline"].append(msg.T)
          frames_data["min_disparity"].append(msg.min_disparity)
          frames_data["max_disparity"].append(msg.max_disparity)
          frames_data["delta_d"].append(msg.delta_d)

        elif topic == "/zedx/zed_node/left/image_rect_color":
          color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
          frames_data["rgb_img"].append(color_image)
        elif topic == "/zedx/zed_node/pose":
          tx,ty,tz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
          qw,qx,qy,qz = msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z
          frames_data["pose_t"].append([tx,ty,tz])
          frames_data["pose_q_wxyz"].append([qw,qx,qy,qz])

        elif topic == "/zedx/zed_node/left/camera_info":
          frames_data["width"].append(msg.width)
          frames_data["height"].append(msg.height)
          frames_data["distortion_model"].append(msg.distortion_model)
          frames_data["intrinsics_3x3"].append(msg.K)
          frames_data["distortion_params"].append(msg.D)


          if len(bag_paths) > 1 and not args.combine:
              bag_path = pathlib.Path(bag_path)
              out_path = os.path.join(args.out,bag_path.with_suffix(".npz").name)
              save_frames(frames_data, out_path)
              frames_data = defaultdict(list)
  
  elif args.version == 2:
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    import ros_utils

    frames_data = defaultdict(list)

    for bag_path in tqdm.tqdm(bag_paths):
      storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
      converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                                      output_serialization_format='cdr')


      reader = rosbag2_py.SequentialReader()
      reader.open(storage_options, converter_options)

      topic_types = reader.get_all_topics_and_types()

      # Create a map for quicker lookup
      type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

      # Set filter for topic of string type
      storage_filter = rosbag2_py.StorageFilter(topics=['/camera/image'])
      reader.set_filter(storage_filter)

      while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        bgr_img = ros_utils.image_to_numpy(msg)
        rgb_img = np.flip(bgr_img, axis=-1)
        frames_data["rgb_img"].append(rgb_img)
        frames_data["ts"].append(msg.header.stamp.sec)

      if len(bag_paths) > 1 and not args.combine:
        bag_path = pathlib.Path(bag_path)
        out_path = os.path.join(args.out,bag_path.with_suffix(".npz").name)
        save_frames(frames_data, out_path)
        frames_data = defaultdict(list)

  else:
    raise Exception("ROS version unrecognized !")

  for k,v in frames_data.items():
    frames_data[k] = np.array(frames_data[k])

  if args.combine or len(bag_paths) == 1:
    np.savez_compressed(args.out, **frames_data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("ROS to Numpy")
  parser.add_argument("--version", "-v", default=2, choices=[1, 2], type=int,
                      help="Which ROS version to use")
  parser.add_argument("--bag", "-b", type=str, required=True,
                      help="Path to bag file to convert. Or a .txt file with list of bags")
  parser.add_argument("--combine", "-c", action="store_true",
                      help="If multiple bag files were specified, this option "
                           "determines if one file is to be outputted")
  parser.add_argument("--out", "-o", type=str, default=None,
                      help="Path to output .npz file")
  args = parser.parse_args()

  main(args)

