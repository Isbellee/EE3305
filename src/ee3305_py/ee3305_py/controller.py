from math import hypot, atan2, inf, cos, sin

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


class Controller(Node):

    def __init__(self, node_name="controller"):
        # Node Constructor =============================================================
        super().__init__(node_name)

        # Parameters: Declare
        self.declare_parameter("frequency", float(20))
        self.declare_parameter("lookahead_distance", float(0.3))
        self.declare_parameter("lookahead_lin_vel", float(0.1))
        self.declare_parameter("stop_thres", float(0.1))
        self.declare_parameter("max_lin_vel", float(0.2))
        self.declare_parameter("max_ang_vel", float(2.0))

        # LiDAR Parameters - 360° Scanning
        self.declare_parameter("obstacle_detection_threshold", float(0.3))  # meters
        self.declare_parameter("enable_obstacle_avoidance", bool(True))
        self.declare_parameter("min_obstacle_distance", float(0.1))  # meters

        # Parameters: Get Values
        self.frequency_ = self.get_parameter("frequency").value
        self.lookahead_distance_ = self.get_parameter("lookahead_distance").value
        self.lookahead_lin_vel_ = self.get_parameter("lookahead_lin_vel").value
        self.stop_thres_ = self.get_parameter("stop_thres").value
        self.max_lin_vel_ = self.get_parameter("max_lin_vel").value
        self.max_ang_vel_ = self.get_parameter("max_ang_vel").value
        
        # LiDAR Parameters: Get Values
        self.obstacle_detection_threshold_ = self.get_parameter("obstacle_detection_threshold").value
        self.enable_obstacle_avoidance_ = self.get_parameter("enable_obstacle_avoidance").value
        self.min_obstacle_distance_ = self.get_parameter("min_obstacle_distance").value

        # Handles: Topic Subscribers
        # !TODO: path subscriber
        self.sub_path_ = self.create_subscription(
            Path,
            "path",
            self.callbackSubPath_,
            10, 
        )    
        # !TODO: odometry subscriber
        self.sub_odom_ = self.create_subscription(
            Odometry,
            "odom",
            self.callbackSubOdom_,
            10, 
        )  
        # Adding a LiDAR
        self.sub_laser_ = self.create_subscription(
            LaserScan,
            'scan',
            self.callbackSubLaser_,
            qos_profile_sensor_data,
        )

        # Handles: Topic Publishers
        # !TODO: command velocities publisher
        self.pub_cmd_vel_ = self.create_publisher(
            TwistStamped,
            "cmd_vel",
            10,
        )
        # !TODO: lookahead point publisher
        self.pub_lookahead_ = self.create_publisher(
            PoseStamped,
            "lookahead",
            10,
        )
        # Handles: Timers
        self.timer = self.create_timer(1.0 / self.frequency_, self.callbackTimer_)

        # Other Instance Variables
        self.received_odom_ = False
        self.received_path_ = False
        self.received_scan_ = False

        # LiDAR Instance Variables
        self.min_distance_360_ = inf  # minimum distance detected in any direction
        self.obstacle_detected_ = False
        # self.emergency_stop_ = False

    # Callbacks =============================================================
    
    # Path subscriber callback
    def callbackSubPath_(self, msg: Path):
        if len(msg.poses) == 0:  # not msg.poses is fine but not clear
            self.get_logger().warn(f"Received path message is empty!")
            return  # do not update the path if no path is returned. This will ensure the copied path contains at least one point when the first non-empty path is received.

        # !TODO: copy the array from the path
        self.path_poses_ = msg.poses

        self.received_path_ = True

    # Odometry subscriber callback
    def callbackSubOdom_(self, msg: Odometry):
        # !TODO: write robot pose to rbt_x_, rbt_y_, rbt_yaw_
        self.rbt_x_ = msg.pose.pose.position.x
        self.rbt_y_ = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        delta_y = 2 * (q.x * q.y + q.w * q.z)
        delta_x = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.rbt_yaw_ = atan2(delta_y, delta_x)

        self.received_odom_ = True

    # LiDAR subscriber callback
    def callbackSubLaser_(self, msg: LaserScan):
        # Process LiDAR scan data to detect obstacles in all directions (360°).
        # This is the simple version that checks the entire scan.
        if not self.enable_obstacle_avoidance_:
            return

        # Find the minimum distance in all directions (360°)
        min_distance = inf
        
        for range_val in msg.ranges:
            # Only consider valid range readings (not inf or nan)
            if msg.range_min < range_val < msg.range_max:
                if range_val < min_distance:
                    min_distance = range_val

        # Update instance variables
        self.min_distance_360_ = min_distance

        # Check for obstacles
        if min_distance < self.obstacle_detection_threshold_:
            self.obstacle_detected_ = True
            self.get_logger().info(f"Obstacle detected at {min_distance:.2f}m")
        else:
            self.obstacle_detected_ = False

        self.received_scan_ = True

    # Gets the lookahead point's coordinates based on the current robot's position and planner's path
    # Make sure path and robot positions are already received, and the path contains at least one point.
    def getLookaheadPoint_(self):
        # Ensure path exists
        if not self.path_poses_:
            self.get_logger().warn("No path received yet. Cannot compute lookahead point.")
            return None, None
        
        # Find the point along the path that is closest to the robot
        min_dist = inf
        closest_idx = 0
        for i, pose_stamped in enumerate(self.path_poses_):
            path_x = pose_stamped.pose.position.x
            path_y = pose_stamped.pose.position.y
            dist = hypot(path_x - self.rbt_x_, path_y - self.rbt_y_)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # From the closest point, iterate towards the goal and find the first point that is at least a lookahead distance away.
        # Return the goal point if no such lookahead point can be found
        lookahead_idx = len(self.path_poses_) - 1

        for i in range(closest_idx, len(self.path_poses_)):
            path_x = self.path_poses_[i].pose.position.x
            path_y = self.path_poses_[i].pose.position.y
            dist = hypot(path_x - self.rbt_x_, path_y - self.rbt_y_)
            if dist >= self.lookahead_distance_:
                lookahead_idx = i
                self.get_logger().info(f"distance: {dist:.3f}")
                break

        # Get the lookahead coordinates
        lookahead_pose = self.path_poses_[lookahead_idx]
        lookahead_x = lookahead_pose.pose.position.x
        lookahead_y = lookahead_pose.pose.position.y

        # Publish the lookahead coordinates
        msg_lookahead = PoseStamped()
        msg_lookahead.header.stamp = self.get_clock().now().to_msg()
        msg_lookahead.header.frame_id = "map"
        msg_lookahead.pose.position.x = lookahead_x
        msg_lookahead.pose.position.y = lookahead_y
        self.pub_lookahead_.publish(msg_lookahead)

        # Return the coordinates
        return lookahead_x, lookahead_y

    # Implement the pure pursuit controller here
    def callbackTimer_(self):
        if not self.received_odom_ or not self.received_path_:
            return  # return silently if path or odom is not received.

        # # Emergency stop: If obstacle is too close anywhere, stop immediately
        # if self.emergency_stop_:
        #     msg_stop = TwistStamped()
        #     msg_stop.header.stamp = self.get_clock().now().to_msg()
        #     msg_stop.twist.linear.x = 0.0
        #     msg_stop.twist.angular.z = 0.0
        #     self.pub_cmd_vel_.publish(msg_stop)
        #     self.get_logger().error("EMERGENCY STOP - Obstacle too close (360° detection)!")
        #     return
        
        # get lookahead point
        lookahead_x, lookahead_y = self.getLookaheadPoint_()

        # get distance to lookahead point (not to be confused with lookahead_distance)
        dist_to_lookahead = hypot(lookahead_x - self.rbt_x_, lookahead_y - self.rbt_y_)
        
        # stop the robot if close to the point.
        if dist_to_lookahead < self.stop_thres_:
                msg_stop = TwistStamped()
                msg_stop.header.stamp = self.get_clock().now().to_msg()
                msg_stop.twist.linear.x = 0.0
                msg_stop.twist.angular.z = 0.0
                self.pub_cmd_vel_.publish(msg_stop)
                self.get_logger().info("Near looksahead. Stopping the robot.")
                return
            
        # get curvature
        diff_x = lookahead_x - self.rbt_x_
        diff_y = lookahead_y - self.rbt_y_

        x_prime = (diff_x * cos(self.rbt_yaw_)) + (diff_y * sin(self.rbt_yaw_))
        y_prime = (diff_y * cos(self.rbt_yaw_)) - (diff_x * sin(self.rbt_yaw_))

        d = hypot(x_prime, y_prime)   # distance to lookahead in robot frame

        if d == 0:
            curvature = 0.0
        else:       
            curvature = (2.0 * y_prime) / (d * d)  # curvature formula

        # calculate velocities
        # lin_vel = self.lookahead_lin_vel_
        # Check sign of x_prime to determine determine direction
        if x_prime < 0:
            lin_vel = -1 * self.lookahead_lin_vel_  # Negative lin_vel to reverse
        else:
            lin_vel = self.lookahead_lin_vel_  # Positive lin_vel to move forward

        ang_vel = lin_vel * curvature  # ang_vel will change direction accordingly w.r.t. the sign of lin_vel

        # Obstacle avoidance: Reduce speed if obstacle detected anywhere (360°)
        if self.obstacle_detected_ and self.enable_obstacle_avoidance_:
            # Calculate speed reduction factor based on distance to obstacle
            speed_factor = (self.min_distance_360_ - self.min_obstacle_distance_) / (self.obstacle_detection_threshold_ - self.min_obstacle_distance_)  # linear interpolation: (current - min) / (max - min)
            speed_factor = max(0.0, min(1.0, speed_factor))  # Clamp to [0%, 100%]

            lin_vel *= speed_factor
            self.get_logger().info(f"Reducing speed to {speed_factor*100:.0f}% due to obstacle at {self.min_distance_360_:.2f}m (360°)")

        # saturate velocities. The following can result in the wrong curvature,
        # but only when the robot is travelling too fast (which should not occur if well tuned).
        lin_vel = max(-self.max_lin_vel_, min(self.max_lin_vel_, lin_vel))
        ang_vel = max(-self.max_ang_vel_, min(self.max_ang_vel_, ang_vel))

        # publish velocities
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.header.stamp = self.get_clock().now().to_msg()
        msg_cmd_vel.twist.linear.x = lin_vel
        msg_cmd_vel.twist.angular.z = ang_vel
        self.pub_cmd_vel_.publish(msg_cmd_vel)


# Main Boiler Plate =============================================================
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Controller())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    
