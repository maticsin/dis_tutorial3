#!/usr/bin/env python3

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs_py import point_cloud2 as pc2
from sklearn.decomposition import PCA
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs as tfg
from message_filters import Subscriber, ApproximateTimeSynchronizer
import webcolors


class DetectRings(Node):
	def __init__(self):
		super().__init__('detect_rings')
		self.declare_parameters('', [('device', '')])
		self.device = self.get_parameter('device').get_parameter_value().string_value

		# Topics
		self.marker_topic = "/ring_marker"

		# Basic ROS stuff
		timer_frequency = 2
		timer_period = 1/timer_frequency

		# ROS Setup
		self.bridge = CvBridge()
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# Subscriptions
		# self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		# self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
		self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, qos_profile_sensor_data)
		# # Synchronized subscribers
		# self.rgb_sub = Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		# self.pc_sub = Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")

		# # Approximate sync (adjust slop as needed)
		# self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.pc_sub], queue_size=10, slop=0.1)
		# self.ts.registerCallback(self.synced_callback)

		# Subscribers with synchronization
		self.rgb_sub = Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.pc_sub = Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")

		self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.pc_sub], queue_size=10, slop=0.1)
		self.ts.registerCallback(self.synced_callback)

		# Publishers

		# State
		self.rings = []
		self.id = 0
		self.latest_cloud = None

		self.get_logger().info(f"Node initialized")
		
	def closest_colour(self,requested_colour):
		distances = {}
		for name in webcolors.names():
			r_c, g_c, b_c = webcolors.name_to_rgb(name)
			rd = (r_c - requested_colour[0]) ** 2
			gd = (g_c - requested_colour[1]) ** 2
			bd = (b_c - requested_colour[2]) ** 2
			distances[name] = rd + gd + bd
		return min(distances, key=distances.get)

	def get_human_readable_color_name(self,requested_colour):
		try:
			closest_name = webcolors.rgb_to_name(requested_colour)
		except ValueError:
			closest_name = self.closest_colour(requested_colour)
		return closest_name

# --------------------- CALLBACKS ---------------------
	
	def depth_callback(self,data):
		try:
			depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
		except CvBridgeError as e:
			print(e)

		depth_image[depth_image==np.inf] = 0
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
		image_1 = depth_image / 65536.0 * 255
		image_1 = image_1/np.max(image_1)*255

		image_viz = np.array(image_1, dtype= np.uint8)

		cv2.imshow("Depth window", image_viz)
		cv2.waitKey(1)

	def synced_callback(self, rgb_msg, pc_msg):
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
		except CvBridgeError as e:
			self.get_logger().error(f"CV Bridge error: {e}")
			return

		height, width = rgb_image.shape[:2]
		gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		edges = cv2.Canny(blurred, 50, 150)
		contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		# --- Keep visualization if needed ---
		# cv2.imshow("Binary Image", edges)
		# gray_contours = gray.copy()
		# cv2.drawContours(gray_contours, contours, -1, (255, 0, 0), 3)
		# cv2.imshow("Detected contours", gray_contours)
		# cv2.waitKey(1)
		# --- End visualization ---

		elps = []
		for cnt in contours:
			# Ensure contour has enough points for fitEllipse
			if cnt.shape[0] >= 5: # fitEllipse requires at least 5 points
				try:
					ellipse = cv2.fitEllipse(cnt)
					elps.append(ellipse)
				except cv2.error as e:
					# Sometimes fitEllipse can fail even with >= 5 points
					# self.get_logger().warn(f"cv2.fitEllipse failed: {e}")
					pass # Ignore contours where ellipse fitting fails


		candidates = []
		# --- Ellipse pairing logic (seems okay, but check thresholds) ---
		for n in range(len(elps)):
			for m in range(n + 1, len(elps)):
				e1 = elps[n]
				e2 = elps[m]
				dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
				# Angle difference needs careful handling due to wrap-around (0/180 degrees)
				angle_diff = np.abs(e1[2] - e2[2])
				angle_diff = min(angle_diff, 180.0 - angle_diff) # Normalize angle difference

				# Adjust thresholds based on expected ring size and perspective
				if dist >= 10: # Increased threshold slightly
					continue

				if angle_diff > 10: # Increased threshold slightly
					continue

				e1_minor_axis = min(e1[1]) # Use min/max for clarity
				e1_major_axis = max(e1[1])
				e2_minor_axis = min(e2[1])
				e2_major_axis = max(e2[1])

				# Ensure one ellipse is reasonably contained within the other
				if e1_major_axis >= e2_major_axis and e1_minor_axis >= e2_minor_axis:
					le = e1
					se = e2
				elif e2_major_axis >= e1_major_axis and e2_minor_axis >= e1_minor_axis:
					le = e2
					se = e1
				else:
					continue # Not contained

				# Optional: Check aspect ratios are similar?
				aspect1 = e1_major_axis / e1_minor_axis if e1_minor_axis > 0 else float('inf')
				aspect2 = e2_major_axis / e2_minor_axis if e2_minor_axis > 0 else float('inf')
				if abs(aspect1 - aspect2) > 0.5: # Example threshold
					continue

				# Optional: Check border width consistency (can be sensitive to noise)
				border_major = (max(le[1]) - max(se[1])) / 2.0
				border_minor = (min(le[1]) - min(se[1])) / 2.0
				if border_minor <= 0: continue # Inner ellipse larger than outer?
				border_diff = np.abs(border_major - border_minor)
				if border_diff > 5: # Example threshold
					continue

				candidates.append((le, se)) # Store consistently (larger, smaller)
		# --- End Ellipse pairing ---

		rings_detected_img = rgb_image.copy() # Draw on a copy
		rings_found_this_frame = False

		# Process candidates *outside* the ellipse generation loops
		processed_centers = [] # Avoid processing nearly identical rings multiple times
		center_threshold_sq = 25**2 # Pixel distance squared threshold

		for c in candidates:
			le = c[0] # Larger ellipse
			se = c[1] # Smaller ellipse

			# --- Check if this ring center is too close to one already processed ---
			center_x, center_y = int(le[0][0]), int(le[0][1])
			is_duplicate = False
			for px, py in processed_centers:
				if (center_x - px)**2 + (center_y - py)**2 < center_threshold_sq:
					is_duplicate = True
					break
			if is_duplicate:
				continue
			processed_centers.append((center_x, center_y))
			# --- End duplicate check ---


			# Drawing the ellipses on the image
			cv2.ellipse(rings_detected_img, le, (0, 255, 0), 2)
			cv2.ellipse(rings_detected_img, se, (0, 255, 0), 2)
			rings_found_this_frame = True

			# --- ROI Calculation ---
			# Use bounding rect of the *larger* ellipse for ROI
			# Add some padding
			padding = 0
			x, y, w, h = cv2.boundingRect(cv2.ellipse2Poly((int(se[0][0]), int(se[0][1])), (int(se[1][0]/2), int(se[1][1]/2)), int(se[2]), 0, 360, 1))

			x_min = max(0, x - padding)
			y_min = max(0, y - padding)
			x_max = min(width, x + w + padding)  # Use image width
			y_max = min(height, y + h + padding) # Use image height

			# Ensure ROI dimensions are valid
			if x_min >= x_max or y_min >= y_max:
				self.get_logger().warn(f"Invalid ROI calculated: [{x_min}:{x_max}, {y_min}:{y_max}]")
				continue

			roi_rgb = rgb_image[y_min:y_max, x_min:x_max]

			# --- Get 3D Points for ROI using pc2.read_points with uvs ---
			# Create list of pixel coordinates (u, v) for the ROI
			
			a = pc2.read_points_numpy(pc_msg, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))
			roi_3d = a[y_min:y_max, x_min:x_max, :]
			depth = roi_3d[:, :, 2]
			if center_x <=x_min or center_x >=x_max or center_y <=y_min or center_y >=y_max:
				self.get_logger().warn(f"Center point ({center_x}, {center_y}) is outside the ROI [{x_min}:{x_max}, {y_min}:{y_max}]")
				continue
			centerDepth = a[int(center_y), int(center_x), 2]

			# --- Get Color ---
			# --- Get average color between ellipses (ring area) ---

			# Create a blank mask
			ring_mask = np.zeros((height, width), dtype=np.uint8)

			# Draw filled outer ellipse
			cv2.ellipse(ring_mask, le, color=255, thickness=-1)

			# Draw filled inner ellipse in black (removes from mask)
			cv2.ellipse(ring_mask, se, color=0, thickness=-1)

			# Mask the RGB image
			ring_pixels = rgb_image[ring_mask == 255]

			if ring_pixels.size == 0:
				self.get_logger().warn("No ring pixels found in mask.")
				ring_color_name = "unknown"
			else:
				avg_bgr = np.mean(ring_pixels, axis=0)
				avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))  # BGR → RGB
				ring_color_name = self.get_human_readable_color_name(avg_rgb)

			# --- Analyze 3D points ---
			if len(depth)>0: # Check if any valid points were collected
				z_values = depth
				z_min = np.min(z_values)
				if(z_min > 1):
					continue
				z_range = centerDepth - z_min # Peak-to-peak (max-min)
				if(z_range > 0.5):
					cv2.ellipse(rings_detected_img, le, (0, 0, 255), 2)
					cv2.ellipse(rings_detected_img, se, (0, 0, 255), 2)
				# Adjust threshold based on expected ring thickness and distance
				ring_type = "3D" if z_range > 0.5 else "2D"
			else:
				# No valid 3D points found in the ROI (either due to error or no points)
				# The warning is now implicitly handled by the empty roi_points array
				# self.get_logger().warn(f"No valid 3D points found in ROI [{x_min}:{x_max}, {y_min}:{y_max}]")
				ring_type = "Unknown (No Points)"
				z_range = 0
			if ring_type == "3D":
				self.get_logger().info(f"Detected {ring_type} ring (z-range: {z_range:.4f}) of color '{ring_color_name}' near ({center_x}, {center_y})")

		if rings_found_this_frame:
			cv2.imshow("Detected rings", rings_detected_img)
			cv2.waitKey(1)
		else:
			# Optionally clear the window or show the original image if no rings found
			cv2.imshow("Detected rings", rgb_image)
			cv2.waitKey(1)

	# --------------------- MAIN FUNCTION ---------------------

def main(args=None):
	rclpy.init(args=args)
	node = DetectRings()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
