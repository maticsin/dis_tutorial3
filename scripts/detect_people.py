#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pickle
from rclpy.duration import Duration
import json

from geometry_msgs.msg import Quaternion, PointStamped
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from ultralytics import YOLO

import math
from sklearn.decomposition import PCA

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		marker_topic = "/people_marker"
		stop_marker_topic = "/stop_point"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.stop_marker_pub = self.create_publisher(Marker, stop_marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []
		self.rbounds = []
		self.lbounds = []

		self.id = 0

		self.markers = []
		self.stops = []

		self.distTresh = 0.75

		self.stopDist = 0.9
		
		self.tf_buffer = Buffer()
		
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def rgb_callback(self, data):

		self.faces = []
		self.rbounds = []
		self.lbounds = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue

				self.get_logger().info(f"Person has been detected!")
				bbox = bbox[0]
 
				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)
				cv_image = cv2.circle(cv_image, (int(bbox[0]),cy), 5,(0,255,0), -1)
				cv_image = cv2.circle(cv_image, (int(bbox[2]),cy), 5,(0,255,0), -1)

				self.rbounds.append((int(bbox[2]),cy))
				self.lbounds.append((int(bbox[0]),cy))
				self.faces.append((cx,cy))

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def marker_nearby(self, pointin):
		d = np.array([pointin.point.x, pointin.point.y])
		for marker in self.markers:
			ar = marker
			self.get_logger()
			# dist = np.linalg.norm(ar - d)
			dist = np.linalg.norm(ar[:2] - d[:2])
			# if(abs(float(d[0]) - p.x) < self.distTresh and abs(float(d[1]) - p.y) < self.distTresh and abs(float(d[2]) - p.z) < self.distTresh):
			# 	return True
			if dist < self.distTresh:
				self.get_logger().info(f"The distance to other markers{dist}")
				return True
		return False
	
	def computeStopPoint(self, center, bounds):
		edge = np.array(bounds)
		center = np.array(center)
		dist = np.sqrt(np.sum((center-edge)**2))
		ang = np.arctan(dist/self.stopDist)
		hipDist = np.sqrt(dist**2+self.stopDist**2)

		x_end = edge[0] + hipDist * np.cos(ang)
		y_end = edge[1] + hipDist * np.sin(ang)

		return (int(x_end), int(y_end))
	
	def createMarker(self, data, pos, color, loc):
		# create marker
		marker = Marker()

		marker.header.frame_id = loc
		marker.header.stamp = data.header.stamp

		marker.type = marker.SPHERE
		marker.id = self.id 
		self.id += 1

		# Set the scale of the marker
		scale = 0.3
		marker.scale.x = scale
		marker.scale.y = scale
		marker.scale.z = scale

		# Set the color
		marker.color.r = float(color[0])
		marker.color.g = float(color[1])
		marker.color.b = float(color[2])
		marker.color.a = 1.0

		# Set the pose of the marker
		marker.pose.position.x = float(pos[0])
		marker.pose.position.y = float(pos[1])
		marker.pose.position.z = float(pos[2])

		return marker
	
	def robot2map(self, data, point):
		# a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
		# a = a.reshape((data.height,data.width,3))

		point_in_robot_frame = PointStamped()
		point_in_robot_frame.header.frame_id = "/base_link"
		point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

		# print(point[1])

		point_in_robot_frame.point.x = float(point[0])
		point_in_robot_frame.point.y = float(point[1])
		point_in_robot_frame.point.z = float(point[2])

		time_now = rclpy.time.Time()

		trans = self.tf_buffer.lookup_transform("map", "base_link", time_now)
		point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)

		return point_in_map_frame

	def get_wall_normal(self, points):

		points = np.array(points)
		pca = PCA(n_components=3)
		pca.fit(points)

		normal = pca.components_[-1]
		return normal
	


	def compute_perpendicular_point(self, face_position, wall_normal, offset=0.5):
		normal = wall_normal / np.linalg.norm(wall_normal)
		perpendicular_point = np.array(face_position) + offset * normal
		return perpendicular_point
	

	def get_wall_points_near_face(self, points, x, y, radius=1.0):
		wall_points = []
		for i in range(max(0, y - radius), min(y + radius, len(points))):
			for j in range(max(0, x - radius), min(x + radius, len(points[i]))):
				wall_points.append(points[i, j, :])
		
		return wall_points
	
	def YawToQuaternion(self, angle_z = 0.):
		quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
		quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
		return quat_msg
	
	def compute_yaw_from_face(self, robot_position, face_position):
		dx = face_position[0] - robot_position[0]
		dy = face_position[1] - robot_position[1]
		yaw = math.atan2(dy, dx)
		quaternion = self.YawToQuaternion(yaw)
		return quaternion

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width     
		point_step = data.point_step
		row_step = data.row_step		

		# iterate over face coordinates
		for (cx,cy),(bx,by),(dx,dy) in zip(self.faces, self.rbounds, self.lbounds):

			try:

				# get 3-channel representation of the poitn cloud in numpy format
				a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
				a = a.reshape((height,width,3))
				
				# center
				C = a[cy, cx, :]


				point_in_robot_frame = PointStamped()
				point_in_robot_frame.header.frame_id = "/base_link"
				point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()


				# d = a[y,x,:]

				point_in_robot_frame.point.x = float(C[0])
				point_in_robot_frame.point.y = float(C[1])
				point_in_robot_frame.point.z = float(C[2])

				time_now = rclpy.time.Time()

				trans = self.tf_buffer.lookup_transform("map", "base_link", time_now)
				point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)

				# read center coordinates

				if self.marker_nearby(point_in_map_frame):
					self.get_logger().info(f"Face already saved")
					continue


				# bound
				B = a[by, bx, :]
				D = a[dy, dx, :]
				# vector in image plane
				vec = D - B
				vec2d = np.array([vec[0], vec[1]])
				# 90 degrees rotation
				n = np.array([-vec2d[1], vec2d[0]])
				norm_n = np.linalg.norm(n)
				if norm_n < 1e-5:
					self.get_logger().warn("Bounding box premajhen ali degeneriran vektor!")
					continue
				n = n / norm_n 

				C2d = np.array([C[0], C[1]]) 
				stop_2d = C2d + self.stopDist * n

				
				stop_3d = np.array([stop_2d[0], stop_2d[1], C[2]])

	

				self.get_logger().info(f"New marker {point_in_map_frame.point.x} {point_in_map_frame.point.y}")

				marker = self.createMarker(data, C, [1,0,0], "/base_link")

				# marker.lifetime = Duration(seconds=lifetime).to_msg()
				temp = [point_in_map_frame.point.x,point_in_map_frame.point.y]
				self.markers.append(temp)
				self.marker_pub.publish(marker)


				point_in_robot_frame.point.x = float(stop_3d[0])
				point_in_robot_frame.point.y = float(stop_3d[1])
				point_in_robot_frame.point.z = float(stop_3d[2])

				#marker_stop = self.createMarker(data, stop_3d, [0,1,0], "/base_link")  # zelena barva
				
				stop_point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)

				# self.stop_marker_pub.publish(marker_stop)
				coords = [
					stop_point_in_map_frame.point.x,
					stop_point_in_map_frame.point.y,
					stop_point_in_map_frame.point.z
				]
				
				stopMarker = self.createMarker(data,coords,[0,1,0], "/map")

				stopMarker.pose.orientation = self.compute_yaw_from_face(coords, temp)
				self.stop_marker_pub.publish(stopMarker)


				# dist_org = np.sqrt(point_in_robot_frame.point.x**2 + point_in_robot_frame.point.y**2)
				# dist_small = dist_org - 0.5
				# razmerje = dist_small / dist_org

				# point_stop = []

				# point_stop.append(point_in_robot_frame.point.x * razmerje)
				# point_stop.append(point_in_robot_frame.point.y * razmerje)
				# point_stop.append(point_in_robot_frame.point.z * razmerje)
				
				# marker_stop = self.createMarker(data,point_stop,[0,1,0], "/base_link")
				# self.marker_pub.publish(marker_stop)


				# stop = self.robot2map(data, point_stop)
				# temp = [stop.point.x,stop.point.y]
				# self.stops.append(temp)
				# stopTemp = [stop.point.x,stop.point.y,stop.point.z]
				# stop_marker = self.createMarker(data,stopTemp,[1,0,0], "/map")
				# self.marker_pub.publish(stop_marker)


				# boundMap = self.robot2map(data, (b1,b2))

				# stopPoint = self.computeStopPoint((point_in_map_frame.point.x,point_in_map_frame.point.y),(boundMap.point.x,boundMap.point.y))

				# stopPoint = [stopPoint[0], stopPoint[1], 0.0]

				# stopMarker = self.createMarker(data,stopPoint,[0,1,0], "/map")
				# self.stop_marker_pub.publish(stopMarker)

				# stop_coordinates = np.array([])
				# distance = math.sqrt(marker.pose.position.x**2 + marker.pose.position.y**2)

				# if distance > 0.5:
				# 	direction_x = marker.pose.position.x / distance
				# 	direction_y = marker.pose.position.y / distance

				# 	stop_coordinates[0] = (direction_x * 0.5)
				# 	stop_coordinates[1] = (direction_y * 0.5)

				# else:

				# 	stop_coordinates[0] = 0
				# 	stop_coordinates[1] = 0

				
				# stop_point_in_map_frame = tfg.do_transform_point(d, trans)
				

				# with open("faces.json", "a") as f: 
				# 	json.dump({"x": stop.point.x, "y": stop.point.y}, f)
				# 	f.write("\n")
				# 	self.get_logger().info(f"NEW FACE SAVED")


				# if (len(self.markers) > 2):
				# 	with open("faces.pkl", "wb") as f:
				# 		pickle.dump(self.markers, f)
				# 	exit()
			except TransformException as te:
					self.get_logger().info(f"Cound not get the transform: {te}")


def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
