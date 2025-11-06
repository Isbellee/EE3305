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

        # Parameters: Get Values
        self.frequency_ = self.get_parameter("frequency").value
        self.lookahead_distance_ = self.get_parameter("lookahead_distance").value
        self.lookahead_lin_vel_ = self.get_parameter("lookahead_lin_vel").value
        self.stop_thres_ = self.get_parameter("stop_thres").value
        self.max_lin_vel_ = self.get_parameter("max_lin_vel").value
        self.max_ang_vel_ = self.get_parameter("max_ang_vel").value

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
        
        # Handles: Topic Publishers
        # !TODO: command velocities publisher
        self.pub_cmd_vel_ = self.create_publisher(
            TwistStamped,
            "cmd_vel",
            10
        )
        
        # !TODO: lookahead point publisher
        self.pub_lookahead_ = self.create_publisher(
            PoseStamped,
            "lookahead",
            10
        )
        
        # Handles: Timers
        self.timer = self.create_timer(1.0 / self.frequency_, self.callbackTimer_)

        # Other Instance Variables
        self.received_odom_ = False
        self.received_path_ = False

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
        self.rbt_yaw_ = msg.pose.pose.position.yaw

        q = msg.pose.pose.orientation
        delta_y = 2 * (q.x * q.y + q.w * q.z)
        delta_x = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.rbt_yaw_ = atan2(delta_y, delta_x)
        
        self.received_odom_ = True

    # Gets the lookahead point's coordinates based on the current robot's position and planner's path
    # Make sure path and robot positions are already received, and the path contains at least one point.
    def getLookaheadPoint_(self):
        # Find the point along the path that is closest to the robot

        # From the closest point, iterate towards the goal and find the first point that is at least a lookahead distance away.
        # Return the goal point if no such lookahead point can be found
        lookahead_idx = len(self.path_poses_) - 1

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

        dx = (diff_x * cos(self.rbt_yaw_)) + (diff_y * sin(self.rbt_yaw_))
        dy = (diff_y * cos(self.rbt_yaw_)) - (diff_x * sin(self.rbt_yaw_))
        d = hypot(dx, dy)   # distance to lookahead in robot frame

        if d == 0:
            curvature = 0.0
        else:       
            curvature = (2.0 * dy) / (d * d)  # curvature formula

        # calculate velocities
        lin_vel = self.lookahead_lin_vel_
        ang_vel = lin_vel * curvature

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
