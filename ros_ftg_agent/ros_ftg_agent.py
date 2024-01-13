import rclpy
from rclpy.node import Node
import f110_agents
import numpy as np
import pickle as pkl
from f110_agents import agents_numpy
from f110_agents.pure_pursuit import StochasticContinousPPAgent
from f110_agents.agent import Agent
# import laser scan message
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from f110_agents.rewards import Progress
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool
from tf_transformations import euler_from_quaternion
from reward_config.config_new import Config as RewardConfig
import math
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import json
# import ros imu message vesc_msgs
# from vesc_msgs.msg import VescImuStamped
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
class Raceline():
    def __init__(self):
        self.xs = []
        self.ys = []
        self.vxs = []

class Track():
    def __init__(self, file):
        self.track = self.load_track(file)
        self.centerline = Raceline()
        self.centerline.xs = self.track[:,0]
        self.centerline.ys = self.track[:,1]

        self.centerline.vxs = self.track[:,2]

    def load_track(self,file):
        # open and read in the track, a csv with x_m, x_y, vx_mps, delta_rad 
        
        track = np.loadtxt(file, delimiter=',')
        return track

class AgentRollout(Node):
    
    def __init__(self, agent, reset_agent, track_path, num_starting_points = 25):
        super().__init__('agent_rollouts')
        self.get_logger().info("Rollout agent node started!")
        # agent is a numpy agent!
        # TODO! adjust and make argument?
        print(track_path)
        self.track = Track(track_path)
        self.progress = Progress(self.track, lookahead=200)
        model_name = str(agent)
        with open(f"dataset/{model_name}", 'w') as f:
            pass
        self.initial = False
        self.timestep = 0
        self.current_lidar_occupancy = None
        self.current_imu = None
        self.laser_scan_sub = self.create_subscription(LaserScan, 'scan', self.get_laser_scan, 10)
        self.current_pose = None
        self.pose_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.get_pose, 10)
        # subscribe to intial pose
        self.reset_publisher = self.create_publisher(Bool, 'reset', 10)
        self.inital_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.inital_pose_callback, 10)
        self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.marker_publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.waypoint_publisher = self.create_publisher(Marker, 'waypoint_marker', 10)
        self.start_publisher = self.create_publisher(Marker, 'start_marker', 10)
        self.imu_subscriber = self.create_subscription(Imu, 'phidgets/imu/data_raw', self.get_imu, 10)
        timer_period = 1/20  # seconds (20 Hz)
        self.timer = self.create_timer(timer_period, self.execute_agent)
        self.agent = agent
        self.reset_agent = reset_agent
        self.state = "inital"
        self.terminate = False


        self.current_speed = 0.0
        self.current_angle = 0.0
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.subscription = self.create_subscription(
            Joy,
            'joy',  # Replace with your topic name
            self.joy_callback,
            10  # Queue size
        )

        # print("Running agent")
        self.TRUNCATION_TIMESTEP = 250
        self.trajectory_num = 0
        
        self.ttc_min = 0.1
        self.deceleration = 18.0
        self.engaged = False
        self.lidar_angle_increment = 0
        self.lidar_angle_min = 0
        self.drop_curr_pose = 0
        self.deaccelerate = False
        self.num_starting_points = num_starting_points
        self.starting_points_progress = np.linspace(0, 1, self.num_starting_points + 1)[1:] 
        self.target_start = None
        self.i = 0

    
    def joy_callback(self, msg):
        # Check if button 6 (index 5) is pressed
        self.terminate = True
        if len(msg.buttons) > 5 and msg.buttons[5] == 1:
            # self.get_logger().info('Button 5 is pressed')
            self.terminate = False
        if len(msg.buttons)>5 and msg.buttons[5] == 1:
            self.running = True
        else:
            self.current_angle =0.0
            self.current_speed = 0.0
        if self.terminate and self.state=="recording":
            self.get_logger().info('Terminating trajectory')



    def publish_state_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"  # or your relevant frame
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 1.0  # Adjust the size of the sphere
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.pose.position = Point(x=-2.0, y=-2.0, z=0.0)  # Set the position of the sphere

        # Set the color based on self.state
        if self.state == "recording":
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue
        elif self.state == "reset":
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        elif self.state =="decelerate":
            marker.color = ColorRGBA(r=1.0, g=0.647, b=0.0, a=1.0)  # Orange
        elif self.state == "resetting":
            marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.5) # Other Orange 

        # Publish the marker
        self.marker_publisher.publish(marker)
        
    def publish_waypoint(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"  # or your relevant frame
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2  # Adjust the size of the sphere
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.pose.position = Point(x=x, y=y, z=0.0)  # Set the position of the sphere

        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green

        # Publish the marker
        self.waypoint_publisher.publish(marker)
    
    def publish_start_point(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"  # or your relevant frame
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2  # Adjust the size of the sphere
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.pose.position = Point(x=x, y=y, z=0.0)  # Set the position of the sphere

        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Green

        # Publish the marker
        self.start_publisher.publish(marker)


    def inital_pose_callback(self, msg):
        # reset the agent
        # print("Resetting agent")
        self.agent.reset()
        self.get_logger().info("resetting pose")
        self.get_logger().info(f"Pose {msg.pose.pose.position.x} {msg.pose.pose.position.y}")
        self.get_logger().info(f"progress {self.progress.previous_closest_idx}")
        self.progress.reset([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.get_logger().info(f"progress {self.progress.previous_closest_idx}")
        self.current_pose = None
        self.current_speed = 0.0
        self.current_angle = 0.0
        self.timestep = 0
        self.engaged = False
        self.state = "resetting"
        self.publish_raceline()

    def publish_raceline(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"  # or your relevant frame
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.1  # width of the line

        # Assign points and colors
        max_speed = max(self.track.centerline.vxs)
        min_speed = min(self.track.centerline.vxs)

        for x, y, vx in zip(self.track.centerline.xs, self.track.centerline.ys, self.track.centerline.vxs):
            # Normalize speed to range [0, 1]
            norm_speed = (vx - min_speed) / (max_speed - min_speed)
            color = ColorRGBA()
            color.r = 1.0 - norm_speed  # More red for slower speed
            color.g = norm_speed        # More green for faster speed
            color.b = 0.0
            color.a = 1.0  # Alpha value

            # Add point and corresponding color
            marker.points.append(Point(x=x, y=y, z=0.0))
            marker.colors.append(color)

        marker_array.markers.append(marker)
        self.publisher.publish(marker_array)


    def get_pose(self, msg):
        # TODO! disable this on real car again!!
        #if self.drop_curr_pose % 10 == 0:
        self.current_pose = msg
        self.drop_curr_pose = 0
        self.drop_curr_pose += 1

    
    def get_imu(self, msg):
        #self.get_logger().info("getting imu pose")
        if self.current_imu is None:
            self.current_imu = []
        self.current_imu.append(msg)

    def detect_collision(self):
        pass

    def should_stop(self, scan, speed):
        if self.engaged:
            print("Safety brake engaged")
            return True
        #print(scan.ranges)
        self.ttc = 200
        self.distance = 0
        # print(scan)
        time_to_stop = speed / self.deceleration
        for i, range in enumerate(scan):
            angle = self.lidar_angle_min + i * self.lidar_angle_increment
            distance = max(speed * math.cos(angle), 0)
            # print(distance)
            if distance > 0:
                ttc = range / distance
                
                if ttc < self.ttc:
                    self.ttc = ttc

                if ttc < self.ttc_min or ttc < time_to_stop:
                    print("safety_brake: speed={} ttc={} tts={} STOP".format(speed, ttc, time_to_stop))
                    self.engaged = True
                    return True
        #print("ttc", self.ttc)
        #print("time to stop", time_to_stop)
        return False

    def find_and_remove_closest(self, array, x, wrap_around=0.1):
        # Adjust x by wrap_around amount and handle cyclic behavior
        target_value = (x + wrap_around) % 1

        # Compute the cyclic differences and find the closest index
        differences = (array - target_value)
        # check if all differences are negative
        if (differences < 0).all():
            closest_idx = np.argmin(array)
        else:
            # the smallest positive number
            # set differences to 1 if negative
            differences[differences < 0] = 1
            closest_idx = np.argmin(differences)
        # Get the closest value and remove it from the array
        closest_value = array[closest_idx]
        array = np.delete(array, closest_idx)
        # Recreate the array if empty
        #if array.size == 0:
        #    array = np.linspace(0, 1, num=10)  # Assuming original linspace parameters
        return closest_value, array
    
    def assemble_obs_dict(self):
        ######## HANDLE LIDAR DATA ########

        lidar_data, timestamp_lidar = self.current_lidar_occupancy
        lidar_data = lidar_data.copy()
        # subsample
        lidar_data = lidar_data[:1080] # saveguard againts having more than 1080 points (e.g. 1081 like on the real car -.-)
        lidar_data = lidar_data[::self.agent.subsample]
       
        # normalize the lidar data
        lidar_data = lidar_data / 10.0 # TODO make this not hardcoded
        # remove lidar data larger than 1   
        lidar_data = np.clip(lidar_data, 0.0, 1.0)
        lidar_data = np.array(lidar_data, dtype=np.float32)
        
        # create an observation dictionary
        obs = {'lidar_occupancy': lidar_data}
        ######## HANDLE PREV ACTION DATA ########
        # obs['prev_action'] = [[self.current_angle, self.current_speed]]
        ######## HANDLE POSE DATA ########
        pose_data = self.current_pose
        x = pose_data.pose.pose.position.x
        y = pose_data.pose.pose.position.y

        orientation_q = pose_data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        theta = yaw


        obs['poses_x'] = np.array([x],dtype=np.float32)
        obs['poses_y'] = np.array([y],dtype=np.float32)
        obs['linear_vels_x'] = np.array([pose_data.twist.twist.linear.x],dtype=np.float32)
        obs['linear_vels_y'] = np.array([pose_data.twist.twist.linear.y],dtype=np.float32)
        obs['ang_vels_z'] = np.array([pose_data.twist.twist.angular.z],dtype=np.float32)
        obs['theta_sin'] = np.array([np.sin(theta)],dtype=np.float32)
        obs['theta_cos'] = np.array([np.cos(theta)],dtype=np.float32)
        # print("Pose", x, y)
        #self.get_logger().info(f"Pose {x} {y}")
        new_progress = self.progress.get_progress(np.array([[x, y]]))
        #self.get_logger().info(f"progress {new_progress}")
        # print("new progress", new_progress)
        obs['progress_sin'] = np.array(np.sin(new_progress*2 * np.pi),dtype=np.float32)
        obs['progress_cos'] = np.array(np.cos(new_progress*2 * np.pi),dtype=np.float32)
        obs['previous_action_steer'] = np.array([self.current_angle],dtype=np.float32)
        obs['previous_action_speed'] = np.array([self.current_speed],dtype=np.float32)
        # add to all obs one dimension at 0
        #for key in obs.keys():
        #    obs[key] = np.expand_dims(obs[key], axis=0)
        obs['lidar_occupancy'] = np.expand_dims(obs['lidar_occupancy'], axis=0)
        ######## FINISHED CREATING OBSERVATION DICTIONARY ########
        #print(obs['lidar_occupancy'])
        #print(obs['lidar_occupancy'].shape)
        assert obs['lidar_occupancy'].shape == (1, 1080//self.agent.subsample)
        return obs , new_progress, timestamp_lidar, pose_data

    def execute_agent(self):
        # print(self.current_pose)
        
        # print(self.current_lidar_occupancy)
        if self.current_pose is None or self.current_lidar_occupancy is None or self.current_imu is None:
            #print("waiting for pose and lidar data")
            self.get_logger().info(f"waiting for pose, lidar data and imu {self.current_pose is None}, {self.current_lidar_occupancy is None}, {self.current_imu is None}")
            
            return
        
        if self.state == "inital":
            pose_data = self.current_pose
            x = pose_data.pose.pose.position.x
            y = pose_data.pose.pose.position.y
            self.progress.reset(np.array([x, y]))
            self.initial = False
            self.current_angle = 0.0
            self.current_speed = 0.0
            self.state = "resetting"


        # self.get_logger().info(f"{self.state}")
        #########################################
        ### Create the observation dictionary ###
        #########################################
        obs, new_progress, timestamp_lidar, pose_data = self.assemble_obs_dict()
        
        # self.state = "recording"
        ######## HANDLE ACTION ########
        # A bit of a mess with all the global variables - ups 
        #self.state = "recording"
        #if self.timestep % 20 == 0:
        #    self.get_logger().info(f"current progress {new_progress}")
        if self.state == "resetting":
            
            info, action, log_prob = self.reset_agent(obs, deaccelerate=False)#self.deaccelerate)
            waypoint = info[0] % len(self.track.centerline.xs)
            waypoint_debug_x = float(self.track.centerline.xs[waypoint])
            waypoint_debug_y = float(self.track.centerline.ys[waypoint])
            self.publish_waypoint(waypoint_debug_x,waypoint_debug_y)
            
            if self.target_start is None:
                # compute the target start
                self.get_logger().info(f"Available starting points {self.starting_points_progress}")
                self.target_start, self.starting_points_progress = self.find_and_remove_closest(self.starting_points_progress,new_progress, wrap_around=0.2)
                self.get_logger().info(f"picked starting point {self.target_start}")
                # transform target start to a index in the track
                start_index = int(self.target_start * len(self.track.centerline.xs))% len(self.track.centerline.xs)
                s_x = float(self.track.centerline.xs[start_index])
                s_y = float(self.track.centerline.ys[start_index])
                self.publish_start_point(s_x, s_y)

                if self.starting_points_progress.size == 0:
                    # reshuffle and start from the begining
                    self.starting_points_progress = np.linspace(0, 1, self.num_starting_points + 1)[1:] 
            # if we are within 0.05 of the target start, set deaccelerate to true
            if (abs(self.target_start - new_progress) < 0.05):
                self.state = "decelerate"
                self.timestep = 0
                info, action, log_prob = self.reset_agent(obs, deaccelerate=True)

        elif self.state == "decelerate":
            info, action, log_prob = self.reset_agent(obs, deaccelerate=True)
            # decelerate for one second
            if self.timestep > 20:
                self.state = "reset"
                self.timestep = 0

        elif self.state == "reset":
            # remain stationary for one second
            self.current_speed = 0.0
            self.current_angle = 0.0
            action = np.array([[0.0, 0.0]])
            log_prob = np.array([0.0])
            if self.timestep > 20:
                self.timestep = 0
                self.target_start = None
                self.state = "recording"
                self.trajectory_num += 1
                self.get_logger().info(f"Recording trajectory: {self.trajectory_num}")

        # Immediately start, dont wait until the next timestep
        if self.state =="recording":
            values, action, log_prob = self.agent(obs, timestep=np.array([self.timestep]))
            #self.get_logger().info(f"[delta_angles, target_angles, current_angles]: {values}")
        
        
        action_out = action.copy()
        action = action[0] #* 0.15 # hardcoded scaling, TODO!
        
        
        self.current_angle += action[0]
        self.current_speed += action[1]
        if self.terminate:
            self.current_angle = 0.0
            self.current_speed = 0.0
        #self.get_logger().info()
        #print(action[1])
        #self.get_logger().info(f"current speed {self.current_speed}: {action[1]}")
        self.current_speed = np.clip(self.current_speed, 0.0, 7.0)
        # assert self.current_speed >= 0.0, "Speed is negative!, it is {}".format(self.current_speed)
        
        # publish the action to ackerman drive
        ackermann_command = AckermannDriveStamped()
        timestamp = self.get_clock().now().to_msg()
        ackermann_command.header.stamp = timestamp
        ackerman_drive = AckermannDrive()
        ackerman_drive.speed = self.current_speed
        ackerman_drive.steering_angle = self.current_angle
        ackermann_command.drive = ackerman_drive
        self.ackermann_pub.publish(ackermann_command)        
        # now lets save all the stuff
        truncated = False

        if self.state== "recording":
            
            action_raw = np.array([[self.current_angle, self.current_speed]])
            model_name = str(self.agent)
            # add the timestap of the lidar and of the pose to the obs
            timestamp_lidar_float = timestamp_lidar.sec + timestamp_lidar.nanosec * 1e-9
            pose_stamp_float = pose_data.header.stamp.sec + pose_data.header.stamp.nanosec * 1e-9

            truncated = self.timestep >= self.TRUNCATION_TIMESTEP
            # terminate = False # TODO listen to termination topic

            time_infos = {}
            time_infos["lidar_timestamp"] = timestamp_lidar_float
            time_infos["pose_timestamp"] = pose_stamp_float
            time_infos["action_timestamp"] = timestamp.sec + timestamp.nanosec * 1e-9
            imu_data = dict()
            imu_data["lin_vel_x"] = []
            imu_data["lin_vel_y"] = []
            imu_data["lin_vel_z"] = []
            imu_data["ang_vel_x"] = []
            imu_data["ang_vel_y"] = []
            imu_data["ang_vel_z"] = []
            imu_data["timestamp"] = []
            for msg_imu in self.current_imu:
                imu_data["lin_vel_x"].append(msg_imu.linear_acceleration.x)
                imu_data["lin_vel_y"].append(msg_imu.linear_acceleration.y)
                imu_data["lin_vel_z"].append(msg_imu.linear_acceleration.z)
                imu_data["ang_vel_x"].append(msg_imu.angular_velocity.x)
                imu_data["ang_vel_y"].append(msg_imu.angular_velocity.y)
                imu_data["ang_vel_z"].append(msg_imu.angular_velocity.z)
                imu_data["timestamp"].append(msg_imu.header.stamp.sec + msg_imu.header.stamp.nanosec * 1e-9)
            
            #time_infos["imu"] = self.current_imu 
            #print("action, timestamp", )
            if True:
                # print(model_name)
                #self.get_logger().info(f"Imu data length {len(imu_data['timestamp'])}")
                #self.get_logger().info(f"terminate {self.terminate}")
                with open(f"dataset/{model_name}", 'ab') as f:
                    collision=self.terminate
                    # if terminate log info
                    if self.terminate:
                        self.get_logger().info(f"Terminated trajectory: {self.trajectory_num}")
                    
                    pkl.dump((action_out, obs, 0.0,
                            self.terminate,
                            truncated,
                            log_prob,
                            self.timestep,
                            model_name,
                            collision,
                            action_raw, 
                            time_infos,
                            imu_data), f)
        if truncated:
            self.state = "resetting"
        
        self.current_imu = [] # clear the imu
        self.timestep += 1
        self.publish_state_marker()


    def get_laser_scan(self, msg):
        # convert to numpy array
        # save scans, timestamp
        #print("begore", msg.ranges)
        #self.get_logger().info("Getting laserscan")
        self.current_lidar_occupancy = (np.array(msg.ranges, dtype=np.float32), msg.header.stamp)
        self.lidar_angle_increment = msg.angle_increment 
        self.lidar_angle_min = msg.angle_min
        # self.laser_raw = msg
        #print("here", self.current_lidar_occupancy)
    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    config_file_path = "/home/rindt/racecar_ws/src/f110_ros_wrapper/config/config_infsaal.json" #"/sim_ws/src/f1tenth_gym_ros/config2/config_fallstudien.json" #"/home/rindt/racecar_ws/src/f110_ros_wrapper/config/config.json"  # Update with the actual path
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # Extract configurations
    reward_config_path = config["reward_config_path"]
    agent_config_path = config["agent_config_path"]
    reset_agent_path = config["reset_agent_path"]
    
    track_path = config["track_path"]
    # "/sim_ws/src/agent_configs/ftg_fast_5.json"
    reward_config = RewardConfig(reward_config_path) 
    agents = Agent()
    agent = agents.load(agent_config_path)
    #reset_agent_path = "/sim_ws/src/agent_configs/reset_agent.json" 
    reset_agent = agents.load(reset_agent_path)

    rolloutNode = AgentRollout(agent, reset_agent, track_path, num_starting_points=10)
    try:
        rolloutNode.run()
    except KeyboardInterrupt:
        pass

    rolloutNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
