#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, qos_profile_sensor_data, QoSReliabilityPolicy

from sklearn.neighbors import KNeighborsClassifier

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        self.depth_image = None
        self.rings = []
        self.ring_published = []
        self.parking_spots = []
        self.colors = []
        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1
        self.marker_num2 = 1

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.depth_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        self.when_to_detect = self.create_subscription(Bool, "/when_to_detect_rings", self.when_to_detect_callback, 1)
        self.is_it_the_sae_ring = self.create_subscription(Bool, "/is_it_the_same_ring", self.is_it_the_same_ring_callback, 1)

        self.ok_to_detect = False
        # Publiser for the visualization markers
        self.marker_pub = self.create_publisher(Marker, "/ring_marker", QoSReliabilityPolicy.BEST_EFFORT)
        
        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

        self.rgb_values = [
            [255, 0, 0],     # Red
            [204, 71, 65],
            [130, 120, 120],
            [116, 69, 68],
            [147, 53, 49],
            [154, 143, 142],
            [62, 37, 36],
            [96, 33, 30],
            [134, 134, 134],
            [159, 153, 152],
            [137, 105, 105],
            [190, 101, 100],
            [161, 159, 159],
            [0, 255, 0],     # Green
            [63, 137, 62],
            [25, 124, 23],
            [128, 136, 128],
            [158, 165, 157],
            [139, 150, 139],
            [117, 131, 116],
            [102, 132, 98],
            [126, 146, 125],
            [80, 93, 79],
            [46, 92, 45],
            [84, 110, 84],
            [0, 0, 255],     # Blue
            [38, 62, 84],
            [26, 42, 58],
            [124, 129, 133],
            [79, 119, 155],
            [115, 121, 131],
            [123, 149, 173],
            [150, 152, 154],
            [145, 161, 177],
            [135, 145, 153],
            [78, 69, 90],
            [44, 59, 77],
            [255, 255, 0],   # Yellow
            [170, 170, 170],  # Gray
            [100, 100, 100],
            [147, 145, 135],
            [44, 45, 44], # Black
            [79, 80, 80],
            [73, 73, 73],
            [135, 135, 135],
            [76, 77, 76],
            [64, 65, 65],
            [136, 136, 136],
            [147, 147, 147],
            [5, 6, 5],
            [71, 71, 71],
            [119, 120, 119],
            [125, 125, 125]
        ]

        # Corresponding color labels
        self.color_labels = ["red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red",
                             "green", "green", "green", "green", "green", "green", "green", "green", "green", "green", "green", "green",
                             "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue",  "blue", "blue", "blue",
                             "yellow",
                             "gray", "gray", "gray",
                             "black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black"]

        # Train the classifier
        self.classifier = KNeighborsClassifier(n_neighbors=1)
        self.classifier.fit(self.rgb_values, self.color_labels)

    def is_it_the_same_ring_callback(self, data):
        # take the last color from the list if the message is true
        if data.data:
            self.colors.pop()
            self.rings.pop()
            self.ring_published.pop()
        
        print("BRISANJE BARVVV")
        print(self.colors)

    def when_to_detect_callback(self, data):
        self.ok_to_detect = True
        print("I am ready to detect rings!")

    def image_callback(self, data):
        #self.get_logger().info(f"I got a new image! Will try to find rings...")
        if not self.ok_to_detect:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if self.depth_image is not None:
            depth_image = self.depth_image
        else:
            return

        blue = cv_image[:,:,0]
        green = cv_image[:,:,1]
        red = cv_image[:,:,2]

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # gray = red

        # Apply Gaussian Blur
        #gray = cv2.GaussianBlur(gray,(3,3),0)

        # Do histogram equalization
        #gray = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
        cv2.imshow("Binary Image", gray)
        cv2.waitKey(1)

        # Canny edge detection
        edges = cv2.Canny(thresh, 100, 200)

        # Extract contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Detected contours", gray)
        cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)


        # Find two elipses with same centers
        candidates_3D = []
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
                if dist >= 3:
                    continue

                # The rotation of the elipses should be whitin 4 degrees of eachother
                if angle_diff>4:
                    continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                
                # # The widths of the ring along the major and minor axis should be roughly the same
                border_major = (le[1][1]-se[1][1])/2
                border_minor = (le[1][0]-se[1][0])/2
                border_diff = np.abs(border_major - border_minor)

                if border_diff > 4:
                    continue

                # Get the depth of the center of the ellipses
                if int(se[0][1]) >= 240 or int(se[0][0]) >= 320:
                    continue
                depth_center = depth_image[int(se[0][1]), int(se[0][0])]
                if depth_center == 0:
                    candidates_3D.append((e1,e2))
                    candidates.append((e1,e2))

            
                #self.get_logger().info(f"{depth_center}")

                #if int(se[0][1]) > 195:
                #    self.parking_spots.append((int(se[0][0]), int(se[0][1])))
                #    candidates.append((e1,e2))
                #candidates.append((e1,e2))

        #print("Processing is done! found", len(candidates), "candidates for rings")

        # Plot the rings on the image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]
            if c in candidates_3D:
                # Get a few points along the perimeter of the smaller ellipse
                e2_points = cv2.ellipse2Poly((int(e2[0][0]), int(e2[0][1])), (int(e2[1][0] / 2), int(e2[1][1] / 2)),
                                            int(e2[2]), 0, 360, 15)
                sampled_points = e2_points[np.random.choice(e2_points.shape[0], min(15, e2_points.shape[0]), replace=False)]

                center_x = int(e2[0][0])
                center_y = int(e2[0][1])

                # Get a bounding box, around the first ellipse ('average' of both elipsis)
                size = (e1[1][0]+e1[1][1])/2
                center = (e1[0][1], e1[0][0])

                if (size / 2) > 26:
                    continue

                # check if points near center are allredy in the list
                if any(abs(center_x - x) < 0.75 and abs(center_y - y) < 0.75 for x, y, c in self.rings):
                    continue

                # Extract color information at sampled points
                color_average = [0, 0, 0]
                for point in sampled_points:
                    x, y = point
                    if x < 320 and y < 240:
                        b = blue[y, x]
                        g = green[y, x]
                        r = red[y, x]
                        #self.get_logger().info(f"colors at {point}: {r}, {g}, {b}")
                        
                        #self.get_logger().info(f"color at {point}: {color}")

                        # calculate the average of the points
                        color_average[0] += r
                        color_average[1] += g
                        color_average[2] += b

                color_average[0] = color_average[0] / len(sampled_points)
                color_average[1] = color_average[1] / len(sampled_points)
                color_average[2] = color_average[2] / len(sampled_points)

            if color_average[0] != 0 and color_average[1] != 0 and color_average[2] != 0:
               
                color = self.get_color(color_average[0], color_average[1], color_average[2])
                if color is not None and color not in self.colors:
                    # put center of the ellipse in the list
                    x = int(e2[0][0])
                    y = int(e2[0][1])
        
                    if x >= 320 and y >= 240:
                        continue
                    
                    print(f'color average: {color_average}')
                    print("CLOR", color)
                    
                    self.rings.append((x, y, color))
                    self.ring_published.append(False)
                    self.colors.append(color)
            else:
                continue    

            
            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

            if len(candidates) > 0:
                cv2.imshow("Detected rings", cv_image)
                cv2.waitKey(1)

            
    def get_color(self, r, g, b):
        pred = self.classifier.predict([[r,g,b]])
        if pred[0] == "gray":
            return None
        return pred[0]

    def pointcloud_callback(self, data):
        height = data.height
        width = data.width	

		# iterate over ring coordinates
        for x,y,c in self.rings:
            
            if self.ring_published[self.rings.index((x,y,c))]:
                continue
            # get 3-channel representation of the point cloud in numpy format
            a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
            a = a.reshape((height,width,3))

            # check if the ring is in the image
            if x >= 320 or y >= 240:
                continue

            # read center coordinates
            d = a[y,x,:]
            if float(d[0]) == float('inf') and float(d[1]) == float('inf') and float(d[2]) == float('inf'):
                continue

            point_x = d[0]
            point_y = d[1]
            point_z = d[2]
            
            # create marker
            marker = Marker()

            marker.header.frame_id = "/base_link"
            marker.header.stamp = data.header.stamp

            marker.type = 2
            marker.id = self.marker_num
            self.marker_num += 1

            # Set the scale of the marker
            scale = 0.1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            rgb = self.get_rgb_values(c)
            marker.color.r = rgb[0]
            marker.color.g = rgb[1]
            marker.color.b = rgb[2]
            marker.color.a = 1.0

            # Set the pose of the marker
            marker.pose.position.x = float(d[0])
            marker.pose.position.y = float(d[1])
            marker.pose.position.z = -float(d[2])

            print(f"Ring at {x}, {y} is at {d[0]}, {d[1]}, {d[2]}")
            self.ring_published[self.rings.index((x,y,c))] = True
            self.marker_pub.publish(marker) 
                  

    def get_rgb_values(self, str):
        if str == "red":
            return [1., 0., 0.]
        elif str == "green":
            return [0., 1., 0.]
        elif str == "blue":
            return [0., 0., 1.]
        elif str == "yellow":
            return [1., 1., 0.]
        elif str == "gray":
            return [0.5, 0.5, 0.5]
        else:
            return [0., 0., 0.]

    def depth_callback(self, data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        depth_image[depth_image==np.inf] = 0
        
        # store the image
        self.depth_image = depth_image        
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()