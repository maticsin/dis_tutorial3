#!/usr/bin/env python3

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sklearn.decomposition import PCA
from collections import Counter
import webcolors
import tf2_geometry_msgs as tfg
import time


class DetectRings(Node):
	def __init__(self):
		super().__init__('detect_rings')
		self.declare_parameters('', [('device', '')])
		self.device = self.get_parameter('device').get_parameter_value().string_value

		# Topics
		self.marker_topic = "/ring_marker"

		# ROS Setup
		self.bridge = CvBridge()
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# Subscriptions
		self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, qos_profile_sensor_data)

		# Synchronized Subscribers
		self.rgb_sub = Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.pc_sub = Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")
		self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.pc_sub], queue_size=10, slop=0.1)
		self.ts.registerCallback(self.synced_callback)

		self.marker_pub = self.create_publisher(Marker, self.marker_topic, QoSReliabilityPolicy.BEST_EFFORT)



		# State
		self.rings = []
		self.id = 0
		self.latest_cloud = None
		self.distTresh = 0.75
		self.get_logger().info("Node initialized")
	
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
	
	def marker_nearby(self, pointin):
		d = np.array([pointin.point.x, pointin.point.y])
		for marker in self.rings:
			ar = marker
			self.get_logger()
			dist = np.linalg.norm(ar[:2] - d[:2])
			if dist < self.distTresh:
				self.get_logger().info(f"The distance to other markers{dist}")
				return True
		return False
	


	def closest_colour(self, requested_colour):
		distances = {}
		for name in webcolors.names():
			r_c, g_c, b_c = webcolors.name_to_rgb(name)
			rd = (r_c - requested_colour[0]) ** 2
			gd = (g_c - requested_colour[1]) ** 2
			bd = (b_c - requested_colour[2]) ** 2
			distances[name] = rd + gd + bd
		return min(distances, key=distances.get)

	def get_human_readable_color_name(self, requested_colour):
		try:
			closest_name = webcolors.rgb_to_name(requested_colour)
		except ValueError:
			closest_name = self.closest_colour(requested_colour)
		return closest_name

	def scale_ellipse(self, ellipse, scale_x, scale_y):
		center, axes, angle = ellipse
		new_axes = (axes[0] * scale_x, axes[1] * scale_y)
		return (center, new_axes, angle)

	def depth_callback(self, data):
		try:
			depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
		except CvBridgeError as e:
			print(e)
			return

		depth_image[depth_image == np.inf] = 0

		image_1 = depth_image / 65536.0 * 255
		image_1 = image_1 / np.max(image_1) * 255
		image_viz = np.array(image_1, dtype=np.uint8)

		cv2.imshow("Depth window", image_viz)
		cv2.waitKey(1)

	def synced_callback(self, rgb_msg, pc_msg):
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
		except CvBridgeError as e:
			self.get_logger().error(f"CV Bridge error: {e}")
			return

		height, width = rgb_image.shape[:2]
		ellipses = self.extract_ellipses_from_image(rgb_image)
		candidates = self.find_ellipse_pairs(ellipses)

		rings_detected_img = rgb_image.copy()
		rings_found_this_frame = False
		processed_centers = []

		all_rings = np.copy(rings_detected_img)

		for le, se in candidates:
			cv2.ellipse(all_rings, le, (255, 0, 0), 2)
			center_x, center_y = int(le[0][0]), int(le[0][1])
			if self.filter_duplicate_centers(center_x, center_y, processed_centers):
				continue
			processed_centers.append((center_x, center_y))

			valid, ring_type, ring_color_name, coords_3d, bottom_x, bottom_y = self.analyze_ring_candidate(
    		le, se, center_x, center_y, pc_msg, rgb_image
)

			cv2.circle(rings_detected_img, (bottom_x, bottom_y), 5, (255, 0, 0), -1)

			if not valid:
				continue

			rings_found_this_frame = True

			if ring_type == "3D":
				cv2.ellipse(rings_detected_img, le, (0, 0, 255), 2)
				cv2.ellipse(rings_detected_img, se, (0, 0, 255), 2)
				cv2.putText(
					rings_detected_img,
					f"{ring_color_name}",
					(center_x + 10, center_y),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 255, 0),
					2,
					cv2.LINE_AA
				)
				self.get_logger().info(f"Detected {ring_type} ring "
									   f"of color '{ring_color_name}' near ({center_x}, {center_y})")
				
				point_in_cam_frame = PointStamped()
				point_in_cam_frame.header.frame_id = "/base_link"
				point_in_cam_frame.header.stamp = self.get_clock().now().to_msg()

				point_in_cam_frame.point.x = float(coords_3d[0])
				point_in_cam_frame.point.y = float(coords_3d[1])
				point_in_cam_frame.point.z = float(coords_3d[2])

				try:
					transform_stamped = self.tf_buffer.lookup_transform(
						"map", 
						"base_link",
						rclpy.time.Time()
					)

					point_in_map_frame = tfg.do_transform_point(point_in_cam_frame, transform_stamped)
					ring_marker = self.createMarker(
						rgb_msg,
						[point_in_map_frame.point.x,
						point_in_map_frame.point.y,
						point_in_map_frame.point.z],	
						[1, 0, 1],
						"map"
					)
					if self.marker_nearby(point_in_map_frame):
						self.get_logger().info(f"Ring already saved")
						continue
					temp = [point_in_map_frame.point.x,point_in_map_frame.point.y]
					self.rings.append(temp)
					self.marker_pub.publish(ring_marker)
					self.get_logger().info(
						f"Ring at map coords: "
						f"({point_in_map_frame.point.x:.2f}, "
						f"{point_in_map_frame.point.y:.2f}, "
						f"{point_in_map_frame.point.z:.2f})"
					)

				except Exception as e:
					self.get_logger().warn(f"Transform to map failed: {e}")
					continue


		display_image = rings_detected_img if rings_found_this_frame else rgb_image
		cv2.imshow("All rings", all_rings)
		cv2.waitKey(1)
		cv2.imshow("Detected 3d rings", display_image)
		cv2.waitKey(1)

	def extract_ellipses_from_image(self, rgb_image):
		gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		edges = cv2.Canny(blurred, 50, 150)
		contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		ellipses = []
		for cnt in contours:
			if cnt.shape[0] >= 5:
				try:
					ellipses.append(cv2.fitEllipse(cnt))
				except cv2.error:
					pass
		return ellipses

	def find_ellipse_pairs(self, ellipses):
		candidates = []
		for n in range(len(ellipses)):
			for m in range(n + 1, len(ellipses)):
				e1, e2 = ellipses[n], ellipses[m]
				dist = np.hypot(e1[0][0] - e2[0][0], e1[0][1] - e2[0][1])
				angle_diff = min(abs(e1[2] - e2[2]), 180.0 - abs(e1[2] - e2[2]))
				if dist >= 10 or angle_diff > 10:
					continue

				e1_axes = sorted(e1[1])
				e2_axes = sorted(e2[1])
				if e1_axes[1] >= e2_axes[1] and e1_axes[0] >= e2_axes[0]:
					le, se = e1, e2
				elif e2_axes[1] >= e1_axes[1] and e2_axes[0] >= e1_axes[0]:
					le, se = e2, e1
				else:
					continue

				aspect1 = e1_axes[1] / e1_axes[0] if e1_axes[0] > 0 else float('inf')
				aspect2 = e2_axes[1] / e2_axes[0] if e2_axes[0] > 0 else float('inf')
				if abs(aspect1 - aspect2) > 0.5:
					continue

				if any(ax[0] == 0 for ax in (le[1], se[1])):
					continue

				le_ar = max(le[1]) / min(le[1])
				se_ar = max(se[1]) / min(se[1])
				if le_ar > 1.5 or se_ar > 1.8:
					continue

				border_major = (max(le[1]) - max(se[1])) / 2.0
				border_minor = (min(le[1]) - min(se[1])) / 2.0
				if border_minor <= 0 or abs(border_major - border_minor) > 5:
					continue

				candidates.append((le, se))
		return candidates

	def filter_duplicate_centers(self, x, y, seen_centers, threshold_sq=25**2):
		return any((x - px) ** 2 + (y - py) ** 2 < threshold_sq for px, py in seen_centers)

	def analyze_ring_candidate(self, le, se, center_x, center_y, pc_msg, rgb_image):
		height, width = rgb_image.shape[:2]
		a = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z")).reshape((height, width, 3))

		# ROI izračun (kot prej)
		x, y, w, h = cv2.boundingRect(
			cv2.ellipse2Poly(
				(int(se[0][0]), int(se[0][1])),
				(int(se[1][0] / 2), int(se[1][1] / 2)),
				int(se[2]), 0, 360, 1
			)
		)
		x_min, y_min = max(0, x), max(0, y)
		x_max, y_max = min(width, x + w), min(height, y + h)

		if x_min >= x_max or y_min >= y_max:
			self.get_logger().warn(f"Invalid ROI: [{x_min}:{x_max}, {y_min}:{y_max}]")
			return (False, None, None, None, None, None)

		if not (x_min <= center_x < x_max and y_min <= center_y < y_max):
			self.get_logger().warn(f"Center point ({center_x}, {center_y}) outside ROI")
			return (False, None, None, None, None, None)

		centerDepth = a[center_y, center_x, 2]
		depth = a[y_min:y_max, x_min:x_max, 2]

		if depth.size == 0:
			return (False, None, None, None, None, None)

		z_min = np.min(depth)
		if z_min > 2:
			return (False, None, None, None, None, None)

		z_range = centerDepth - z_min
		ring_type = "3D" if z_range > 0.5 else "2D"

		# Ustvarimo masko in preberemo barvo (kot prej)
		ring_mask = np.zeros((height, width), dtype=np.uint8)
		cv2.ellipse(ring_mask, self.scale_ellipse(le, 0.9, 0.9), color=255, thickness=-1)
		cv2.ellipse(ring_mask, self.scale_ellipse(se, 1.04, 1.04), color=0, thickness=-1)
		ring_pixels = rgb_image[ring_mask == 255]

		if ring_pixels.size == 0:
			self.get_logger().warn("No ring pixels found in mask.")
			return (False, None, None, None, None, None)

		# Določimo dominantno barvo
		rgb_pixels = [(int(p[2]), int(p[1]), int(p[0])) for p in ring_pixels]  # BGR -> RGB
		dominant_rgb, _ = Counter(rgb_pixels).most_common(1)[0]
		ring_color_name = self.get_human_readable_color_name(dominant_rgb)

		# -- Poiščemo “spodnjo” točko elipse (lowest y) --
		(cx, cy) = (int(le[0][0]), int(le[0][1]))
		(axes_x, axes_y) = (int(le[1][0] / 2), int(le[1][1] / 2))
		angle = int(le[2])
		ellipse_points = cv2.ellipse2Poly((cx, cy), (axes_x, axes_y), angle, 0, 360, 1)
		if ellipse_points.size == 0:
			self.get_logger().warn("ellipse2Poly returned no points.")
			return (False, None, None, None, None, None)

		best_y = -1
		best_pt_2d = None
		for px, py in ellipse_points:
			if not (0 <= px < width and 0 <= py < height):
				continue
			z = a[py, px, 2]
			if not np.isfinite(z):
				continue
			if py > best_y:
				best_y = py
				best_pt_2d = (px, py)

		if best_pt_2d is None:
			self.get_logger().warn("No valid bottom point found on ellipse.")
			return (False, None, None, None, None, None)

		bottom_x, bottom_y = best_pt_2d
		coords_3d = a[bottom_y, bottom_x, :]  # (X, Y, Z) v kamerinem okviru
		if not np.isfinite(coords_3d).all():
			self.get_logger().warn("Bottom ellipse point has invalid (NaN/Inf) coords.")
			return (False, None, None, None, None, None)

		# Vrnemo 6 elementov: (valid, ring_type, ring_color, coords_3d, bottom_x, bottom_y)
		return (True, ring_type, ring_color_name, coords_3d, bottom_x, bottom_y)



def main(args=None):
	rclpy.init(args=args)
	node = DetectRings()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
