#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import time
import math
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus
from std_srvs.srv import Trigger

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from visualization_msgs.msg import Marker


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None

        self.faces = 0
        self.rings = 0

        self.Markers = None
        self.image_cli = self.create_client(Trigger, 'request_image')
        while not self.image_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for 'request_image' service...")

        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)
        self.next_marker_pub = self.create_publisher(Marker, "/aaaa", QoSReliabilityPolicy.BEST_EFFORT)
        self.tts_pub = self.create_publisher(String, '/tts', 10)
        self.spoken = False
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.stop_marker_sub = self.create_subscription(
            Marker, 
            '/stop_point',
            self.face_marker_callback, 
            qos_profile_sensor_data
        )
        self.ring_marker_sub = self.create_subscription(
            Marker, 
            '/ring_stop_marker',
            self.ring_marker_callback, 
            qos_profile_sensor_data
        )
        self.next_move = []
        self.get_logger().info(f"Robot commander has been initialized!")


    def chain_sort_next_move(self):
        # Če imamo manj kot 2 elementa, ni kaj sortirati
        if len(self.next_move) < 2:
            return

        # 1) Shranimo prvi element (trenutni cilj), ki se ne spreminja
        first_item = self.next_move[0]

        # 2) Preostale točke damo v 'points'
        points = self.next_move[1:]

        # 3) Počistimo originalni seznam in vrnemo first_item nazaj
        self.next_move = [first_item]

        # 4) current_position = pozicija first_item, ker želimo,
        #    da se naslednje točke razvrstijo glede na njo in vse naslednje
        current_position = (first_item[0], first_item[1])

        ordered_path = []

        # 5) Dokler ima 'points' točke, iščemo najbližjo k 'current_position'
        while points:
            nearest_idx = None
            nearest_dist = float('inf')

            for i, item in enumerate(points):
                # item je (x, y, orientation, type, (optionally text))
                dx = item[0] - current_position[0]
                dy = item[1] - current_position[1]
                dist = math.hypot(dx, dy)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i

            # Najbližjo odstranimo iz 'points' in jo dodamo v 'ordered_path'
            best_item = points.pop(nearest_idx)
            ordered_path.append(best_item)

            # Posodobimo 'current_position'
            current_position = (best_item[0], best_item[1])

        # 6) Na koncu 'verižnega' iskanja dodamo 'ordered_path' na konec self.next_move
        self.next_move.extend(ordered_path)

        
    def face_marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        if math.isnan(x) or math.isnan(y):
            self.get_logger().warn("Marker has NaN position, ignoring!")
            return
        print(f'Received a marker: ID={msg.id}, position=({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}), yaw= {msg.pose.orientation}')

        self.faces += 1

        #if self.faces == 3:
            #self.erasePath()

        #check for nex available spot in next move

        marker_tuple = (msg.pose.position.x, msg.pose.position.y, msg.pose.orientation, "face")

        self.next_move.append(marker_tuple)
        if hasattr(self, 'current_pose'):
            self.chain_sort_next_move()
            print(self.next_move)

    def ring_marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        if math.isnan(x) or math.isnan(y):
            self.get_logger().warn("Marker has NaN position, ignoring!")
            return
        print(f'Received a marker: ID={msg.id}, position=({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}), yaw= {msg.pose.orientation}')

        self.rings += 1

        #if self.rings == 4:
            #self.erasePath()

        #check for nex available spot in next move

        marker_tuple = (msg.pose.position.x, msg.pose.position.y, msg.pose.orientation, "ring", msg.text) 

        self.next_move.append(marker_tuple)
        if hasattr(self, 'current_pose'):
            self.chain_sort_next_move()
            print(self.next_move)


    def erasePath(self):
        self.next_move = [
            item for item in self.next_move
            if item[3] != 'path'
        ]


    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)

        self.goal_handle = send_goal_future.result()
        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
    def toPose(self, move):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(move[0])
        goal_pose.pose.position.y = float(move[1])
        goal_pose.pose.orientation = move[2]
        return goal_pose
    
    def createMarker(self, pos, color, loc, id):
        marker = Marker()
        marker.header.frame_id = loc
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.type = marker.SPHERE
        marker.id = id

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
        marker.pose.position.z = float(0)

        return marker

    
def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()
    rc.waitUntilNav2Active()

    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    if rc.is_docked:
        rc.undock()

    rc.next_move = [(-0.15, -1.5, rc.YawToQuaternion(4), "path"), 
                    (-0.8, -0.5, rc.YawToQuaternion(3), "path"), #
                    (-1.3, 0.5, rc.YawToQuaternion(3), "path"), 
                    ( 0 , 2, rc.YawToQuaternion(0), "path"),
                    ( -1.5 , 4.5, rc.YawToQuaternion(2), "path"), #
                    ( 0 , 3.3, rc.YawToQuaternion(4), "path"),
                    ( 1.5 , 3.3, rc.YawToQuaternion(4), "path"),
                    ( 2.2 , 2, rc.YawToQuaternion(3), "path"),
                    ( 1 , 0, rc.YawToQuaternion(5), "path"),
                    ( 2.5 , -1, rc.YawToQuaternion(2), "path"),
                    ( 2 , -1.8, rc.YawToQuaternion(5), "path"),
                    ( 1 , -2, rc.YawToQuaternion(2), "path"),
                    ( 0 , -2, rc.YawToQuaternion(2), "path"),]
    


    while len(rc.next_move):
        nextMarker = rc.createMarker(rc.next_move[0], [0, 0, 1], "/map", 0)
        rc.next_marker_pub.publish(nextMarker)


        rc.goToPose(rc.toPose(rc.next_move[0]))
        while not rc.isTaskComplete():
            # rc.info("Waiting for the task to complete...")
            # print(len(rc.next_move))
            # rclpy.spin_once(rc)
            if len(rc.next_move) >= 2:
                if rc.next_move[1][3] == "face" and rc.next_move[0][3] == "path":
                    #save current position to come back and move the next point not implemented
                    curr_pose_stamped = PoseStamped()
                    curr_pose_stamped.header.frame_id = 'map'
                    curr_pose_stamped.header.stamp = rc.get_clock().now().to_msg()
                    curr_pose_stamped.pose = rc.current_pose.pose

                    curr_pos = (
                        curr_pose_stamped.pose.position.x,
                        curr_pose_stamped.pose.position.y,
                        curr_pose_stamped.pose.orientation,
                        "path_back"
                    )

                    curr_next_pos = rc.next_move[0]

                    i = 1
                    while i < len(rc.next_move) and rc.next_move[i][3] == "face":
                        i += 1

                    if(len(rc.next_move) > i):
                        rc.next_move.insert(i, curr_pos)
                        rc.next_move.insert(i+1, curr_next_pos)
                    else: 
                        rc.next_move.append(curr_pos)
                        rc.next_move.append(curr_next_pos)

                    print(len(rc.next_move))

                    for a in rc.next_move:
                        print(a[1])

                    rc.cancelTask()
                    print("canceled")
                elif rc.next_move[1][3] == "face" and rc.next_move[0][3] == "path_back":
                    curr_next_pos = rc.next_move[0]

                    i = 1
                    while i < len(rc.next_move) and rc.next_move[i][3] == "face":
                        i += 1

                    if(len(rc.next_move) > i):
                        rc.next_move.insert(i, curr_next_pos)
                    else: 
                        rc.next_move.append(curr_next_pos)

                    print(len(rc.next_move))

                    for a in rc.next_move:
                        print(a[1])

                    rc.cancelTask()
                    print("canceled")

        #stop say hello
        if (rc.next_move[0][3] == "face"):
            
            msg = String()
            msg.data = "Hello, there stranger."
            rc.tts_pub.publish(msg)
            rc.spoken = True
            rc.get_logger().info("Message published to /tts: Hello ") 
            print("Hello")
            req = Trigger.Request()
            future = rc.image_cli.call_async(req)
            time.sleep(3)
        elif (rc.next_move[0][3] == "ring"):
            msg = String()
            msg.data = rc.next_move[0][4] + "ring"
            rc.tts_pub.publish(msg)
            rc.spoken = True
            rc.get_logger().info("Message published to /tts: Hello ") 
            print(rc.next_move[0][4])
            time.sleep(3)
        else:
            print("finished next move")
            
        rc.next_move.pop(0)

        if (len(rc.next_move) > 0):
            print(rc.next_move[0][1])


    msg = String()
    msg.data = "I did the task."
    rc.tts_pub.publish(msg)
    time.sleep(3)
    

    # rclpy.spin(rc)


    rc.destroyNode()

    # And a simple example
if __name__=="__main__":
    main()
