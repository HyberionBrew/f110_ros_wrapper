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
from vesc_msgs.msg import VescImuStamped
from sensor_msgs.msg import Joy

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
        self.imu_subscriber = self.create_subscription(VescImuStamped, 'sensors/imu', self.get_imu, 10)
        timer_period = 1/20  # seconds (20 Hz)
        self.timer = self.create_timer(timer_period, self.execute_agent)
        self.agent = agent
        self.reset_agent = reset_agent
        self.state = "resetting"
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

        print("Running agent")
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


    
    def joy_callback(self, msg):
        # Check if button 6 (index 5) is pressed
        self.terminate = False
        if len(msg.buttons) > 4 and msg.buttons[4] == 1:
            self.get_logger().info('Button 4 is pressed')
            self.current_speed = 0.0
            self.current_angle = 0.0
            self.terminate = True
        if len(msg.buttons)>5 and msg.buttons[5] == 1:
            self.running = True
        else:
            self.current_angle =0.0
            self.current_speed = 0.0



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
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        elif self.state == "crash_imminent":
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
        elif self.state == "resetting":
            marker.color = ColorRGBA(r=1.0, g=0.647, b=0.0, a=1.0)  # Orange

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
        print("Resetting agent")
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
        self.current_imu = msg

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

    def execute_agent(self):
        if self.current_pose is None or self.current_lidar_occupancy is None or self.current_imu is None:
            #print("waiting for pose and lidar data")
            self.get_logger().info("waiting for pose and lidar data")
            return
        
        if self.initial == True:
            self.progress.reset(np.array([x, y]))
            self.initial = False
        self.get_logger().info(f"{self.state}")
        #########################################
        ### Create the observation dictionary ###
        #########################################

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
        #self.get_logger().info(f"progress {self.progress.previous_closest_idx}")
        # print("new progress", new_progress)
        obs['progress_sin'] = np.array(np.sin(new_progress),dtype=np.float32)
        obs['progress_cos'] = np.array(np.cos(new_progress),dtype=np.float32)
        obs['previous_action'] = np.array([[self.current_angle, self.current_speed]])
        # add to all obs one dimension at 0
        #for key in obs.keys():
        #    obs[key] = np.expand_dims(obs[key], axis=0)
        obs['lidar_occupancy'] = np.expand_dims(obs['lidar_occupancy'], axis=0)
        ######## FINISHED CREATING OBSERVATION DICTIONARY ########
        #print(obs['lidar_occupancy'])
        #print(obs['lidar_occupancy'].shape)
        assert obs['lidar_occupancy'].shape == (1, 1080//self.agent.subsample)
        
        # self.state = "recording"
        ######## HANDLE ACTION ########
        # A bit of a mess with all the global variables - ups 
        #self.state = "recording"
        if self.timestep % 20 == 0:
            self.get_logger().info(f"current progress {new_progress}")
        
        
        if self.state == "resetting":
            
            info, action, log_prob = self.reset_agent(obs, deaccelerate=self.deaccelerate)
            waypoint = info[0] % len(self.track.centerline.xs)
            xx = float(self.track.centerline.xs[waypoint])
            yy = float(self.track.centerline.ys[waypoint])
            self.get_logger().info(f"{xx} {yy}")
            self.publish_waypoint(xx,yy)
            
            if self.target_start is None:
                
                self.get_logger().info(f"Available starting points {self.starting_points_progress}")
                self.target_start, self.starting_points_progress = self.find_and_remove_closest(self.starting_points_progress,new_progress, wrap_around=0.2)
                self.get_logger().info(f"picked starting point {self.target_start}")
                # transform target start to a index in the track
                start_index = int(self.target_start * len(self.track.centerline.xs))% len(self.track.centerline.xs)
                s_x = float(self.track.centerline.xs[start_index])
                s_y = float(self.track.centerline.ys[start_index])
                self.publish_start_point(s_x, s_y)


                if self.starting_points_progress.size == 0:
                    self.starting_points_progress = np.linspace(0, 1, self.num_starting_points + 1)[1:] 
                #print("remaining", self.starting_points_progress)
            # if we are within 0.05 of the target start, set deaccelerate to true
            else:
                if (abs(self.target_start - new_progress) < 0.05) and not self.deaccelerate:
                    self.deaccelerate = True
                    # start timer to wait for one second, after which we swap to recording
                    self.timestep = 0
                    self.current_speed = 0.0
                    self.current_angle = 0.0
                    #self.deaccelerate_timer = self.timestep
                if self.deaccelerate and self.timestep > 20:
                    self.deaccelerate = False
                    self.state = "recording"
                    self.get_logger().info(f"Recording trajectory: {self.trajectory_num}")
                    #print(f"Recording trajectory: {self.trajectory_num}")
                    self.timestep = 0
                    self.trajectory_num += 1
                    self.target_start = None
                    self.current_speed = 0.0
                    self.current_angle = 0.0
                    #if self.trajectory_num == 100:
                    #    exit()
                #if self.deaccelerate:
                    #print("waiting",self.timestep)
                    #print(action)
            #print(curr_progress)
            #print("vs")
            #if self.timestep % 30 == 0:
            #    print(new_progress, self.target_start, )
            # select next starting point
            
            #print("..........")
            #print(self.current_speed)
            #print(reset_almost_done)
            #if obs['linear_vels_x'][0] < 0.05:

            #    self.state = "recording"
            #    print(f"Recording trajectory: {self.trajectory_num}")
            #    self.timestep = 0
            #    self.trajectory_num += 1
        
        if self.state=="recording":
            values , action, log_prob = self.agent(obs, timestep=np.array([self.timestep]))
            #self.get_logger().info(f"[delta_angles, target_angles, current_angles]: {values}")
            
        elif self.state == "crash_imminent":
            print("crash iminent!!")
            self.current_speed = 0.0
            self.current_angle = 0.0
            action = np.array([[0.0, 0.0]])
            log_prob = np.array([[0.0, 0.0]])

        #assert (action<=1.0).all() 
        #assert (action>=-1.0).all()
        #print(action)
        action_out = action.copy()
        action = action[0] *0.3 #* 0.15 # hardcoded scaling, TODO!
        
        
        self.current_angle += action[0]
        self.current_speed += action[1]
        
        self.current_speed = np.clip(self.current_speed, 0.0, 2.0)
        assert self.current_speed >= 0.0, "Speed is negative!, it is {}".format(self.current_speed)
        
        # publish the action to ackerman drive
        ackermann_command = AckermannDriveStamped()
        timestamp = self.get_clock().now().to_msg()
        ackermann_command.header.stamp = timestamp
        ackerman_drive = AckermannDrive()
        ackerman_drive.speed = self.current_speed
        ackerman_drive.steering_angle = self.current_angle
        ackermann_command.drive = ackerman_drive
        self.ackermann_pub.publish(ackermann_command)
        # check if we will be crashing soon
        terminate = False #self.should_stop(self.current_lidar_occupancy[0], 
        #          obs['linear_vels_x'][0])
        
        if terminate == True:
            pass
            #self.state = "crash_imminent"
        
        # now lets save all the stuff
        truncated = False

        if self.state=="recording":
            
            action_raw = np.array([[self.current_angle, self.current_speed]])
            model_name = str(self.agent)
            # add the timestap of the lidar and of the pose to the obs
            timestamp_lidar_float = timestamp_lidar.sec + timestamp_lidar.nanosec * 1e-9
            pose_stamp_float = pose_data.header.stamp.sec + pose_data.header.stamp.nanosec * 1e-9
            #obs['lidar_timestamp'] = np.array([timestamp_lidar_float],dtype=np.float32)
            #obs['pose_timestamp'] = np.array([pose_stamp_float],dtype=np.float32)
            
            #if self.timestep % 20 == 0:
            #    print("Timestep", self.timestep)
            
            # self.timestep == 1000
            truncated = self.timestep >= self.TRUNCATION_TIMESTEP - 1
            # terminate = False # TODO listen to termination topic

            time_infos = {}
            time_infos["lidar_timestamp"] = timestamp_lidar_float
            time_infos["pose_timestamp"] = pose_stamp_float
            time_infos["action_timestamp"] = timestamp.sec + timestamp.nanosec * 1e-9
            #time_infos["imu"] = self.current_imu 
            #print("action, timestamp", )
            done = terminate
            if True:
                # print(model_name)
                with open(f"dataset/{model_name}", 'ab') as f:
                    collision=terminate
                    pkl.dump((action_out, obs, 0.0,
                            self.terminate,
                            truncated,
                            log_prob,
                            self.timestep,
                            model_name,
                            collision,
                            action_raw, 
                            time_infos), f)
        if truncated or terminate:
            #self.state = "resetting"
            if terminate:
                self.state = "crash_imminent"
            else:
                self.state = "resetting"
        self.timestep += 1
        self.publish_state_marker()


    def get_laser_scan(self, msg):
        # convert to numpy array
        # save scans, timestamp
        #print("begore", msg.ranges)
        self.current_lidar_occupancy = (np.array(msg.ranges, dtype=np.float32), msg.header.stamp)
        self.lidar_angle_increment = msg.angle_increment 
        self.lidar_angle_min = msg.angle_min
        # self.laser_raw = msg
        #print("here", self.current_lidar_occupancy)
    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    config_file_path = "/home/rindt/racecar_ws/src/f110_ros_wrapper/config/config_fallstudien.json" #"/home/rindt/racecar_ws/src/f110_ros_wrapper/config/config.json"  # Update with the actual path
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

    rolloutNode = AgentRollout(agent, reset_agent, track_path)
    try:
        rolloutNode.run()
    except KeyboardInterrupt:
        pass

    rolloutNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
