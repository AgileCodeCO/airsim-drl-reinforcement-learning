import airsimdroneracinglab as airsim
import numpy as np
import cv2
import threading
import time
import random
import math
#import open3d as o3d
import os

class DRLEnvironment(object):

    action_space = (2,)
    max_axis_velocity = 2.0
    
    def __init__(
        self,
        drone_name="drone_1",
        viz_image_cv2=True,
        observation_type="images"
    ):
        self.drone_name = drone_name
        self.viz_image_cv2 = viz_image_cv2
        
        self.airsim_client = airsim.MultirotorClient()        
        self.airsim_client_images = airsim.MultirotorClient()        
        self.airsim_client_odom = airsim.MultirotorClient()
               
        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = (
            10  # see https://github.com/microsoft/AirSim-Drone-Racing-Lab/issues/38
        )
        
        self.observation_type = observation_type
        
        # reward
        self.max_distance = 12
        self.previous_distance = 2       
        self.next_gate = 0
        self.has_collission = False
        self.has_finished = False
        
        # timer
        self.start_time = time.time()
        
        if self.observation_type == "lidar" and self.viz_image_cv2:
            self.vis = o3d.visualization.Visualizer()
            
        self.is_log_monitor_thread_active = False
        
    def get_observation_space(self):
        if self.observation_type == "images":
            return  (3, 240, 320)
        else:
            return (3000, 3)
    
    def start_log_monitor_callback_thread(self):
        if not self.is_log_monitor_thread_active:
            self.is_log_monitor_thread_active = True
                    
            self.log_monitor_callback_thread = threading.Thread(
                target=self.repeat_log_monitor_callback
            )
                        
            self.log_monitor_callback_thread.start()

    def stop_log_monitor_callback_thread(self):
        if self.is_log_monitor_thread_active:
            self.is_log_monitor_thread_active = False
    
    def open_log_file(self):
        path = "C:\\Repos\\air-sim\ADRL\\Saved\\Logs\\RaceLogs"
        #path = "/home/JorgeGonzalez/ADRL/ADRL/ADRL/Saved/Logs/RaceLogs"
        files = os.listdir(path)
        list_of_files = [os.path.join(path, basename) for basename in files if basename.endswith(".log")]
        latest_file = max(list_of_files, key=os.path.getctime)        
        return open(latest_file, "r+")
            
    def follow_log_file(self, filename):
        filename.seek(0, 2)
        while self.is_log_monitor_thread_active:
            line = filename.readline()
            if not line:
                time.sleep(0.25)
                continue
            yield line
            
    def check_colission(self, line):
        tokens = line.split()
        #print(line)
        if tokens[0] == self.drone_name and tokens[3] == "penalty":      
            if int(tokens[4]) > 0:
                self.has_collission = True       
                
        if tokens[0] == self.drone_name and tokens[3] == "finished":  
            self.has_finished = True
                            
    def repeat_log_monitor_callback(self):        
        f = self.open_log_file()
        for line in self.follow_log_file(f):
            self.check_colission(line)
    
    def calculate_reward(self):
        
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        position = drone_state.kinematics_estimated.position
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        
        reward = 0
        done = False
        
        lastGatePassed = self.airsim_client_odom.simGetLastGatePassed(self.drone_name)

        if lastGatePassed > 100:
            lastGatePassed = -1
        
        #print(lastGatePassed, self.next_gate)
        if lastGatePassed == self.next_gate:
            reward += 5
            self.next_gate = lastGatePassed + 1
            self.previous_distance = self.max_distance
        else:   
            if lastGatePassed > self.next_gate:
                done = True
                
            if lastGatePassed < len(self.gate_poses_ground_truth) - 1:            
                
                gate = self.gate_poses_ground_truth[lastGatePassed+1]
                
                drone_position = np.array([position.x_val, position.y_val, position.z_val])
                gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
                
                direction_vector = gate_position - drone_position
                
                distance = position.distance_to(gate.position)
                
                direction = direction_vector.dot(linear_velocity.to_numpy_array())
                
                if self.previous_distance > distance and distance < 2:
                    reward += 0#1
                    
                    if direction >= 0 and direction < 1:
                        reward += 1
                else:
                    reward += 0#-1
                    
                    if direction < 0:
                        reward += 0#-1

                if distance > self.max_distance:
                    #print(distance, self.max_distance)
                    reward = 0#-5
                    done = True
                    
                self.previous_distance = distance            
            
        isDisqualified = self.airsim_client_odom.simIsRacerDisqualified(self.drone_name)
        if isDisqualified:
           reward = -1#0
           done = True
           
        if self.has_collission:
           reward = -1#0
           done = True
           
        if self.has_finished:           
           done = True
           
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 300:
            reward = 0
            done = True
        
        #print(elapsed_time, reward, done)
        return (reward, done)
            
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=1):
        
        # self.start_image_callback_thread()
        
        self.airsim_client.simStartRace(tier)        
        
        self.start_time = time.time()
        self.start_log_monitor_callback_thread();
        
        self.initialize_drone()
        self.takeoff()
        self.get_ground_truth_gate_poses()
        
        return self.get_observation()

    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )

        self.airsim_client.setTrajectoryTrackerGains(
            traj_tracker_gains, vehicle_name=self.drone_name
        )
        time.sleep(0.2)
    
    def takeoff(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        takeoff_waypoint = airsim.Vector3r(
            start_position.x_val,
            start_position.y_val,
            start_position.z_val - takeoff_height,
        )

        self.airsim_client.moveOnSplineAsync(
            [takeoff_waypoint],
            vel_max=15.0,
            acc_max=5.0,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            vehicle_name=self.drone_name,
        ).join()
    
    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        self.gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)            
            counter = 0
            while (
                math.isnan(curr_pose.position.x_val)
                or math.isnan(curr_pose.position.y_val)
                or math.isnan(curr_pose.position.z_val)
            ) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                #print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(
                curr_pose.position.x_val
            ), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.y_val
            ), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.z_val
            ), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)
            #print(gate_name, curr_pose.position)
                    
    def init_race_environment(self):
        self.airsim_client.confirmConnection()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom.confirmConnection()
        
        self.load_level(level_name="Soccer_Field_Easy")
        
        if self.observation_type == "lidar" and self.viz_image_cv2:
            self.vis.create_window(width=640,height=480)
    
    def get_observation(self):    
        if self.observation_type == "images":    
            return self.get_camera_image()
        else:
            return self.get_lidar_points()
    
    def get_camera_image(self):
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        
        if self.viz_image_cv2:
            cv2.imshow("Drone FPV", img_rgb)
            cv2.waitKey(1)
        
        img_rgb = np.moveaxis(img_rgb, [2], [0])
        
        #print(img_rgb.shape)
        
        return img_rgb
    
    def get_lidar_points(self):
        lidar_data = self.airsim_client_odom.getLidarData(lidar_name="LidarSensor1", vehicle_name=self.drone_name)
        
        if self.viz_image_cv2:     
            request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
            response = self.airsim_client_images.simGetImages(request)
            img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        
        complete_points = np.zeros(self.get_observation_space())
        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))

        # if self.get_observation_space()[0] < len(points):
        #      print(len(points))        
        
        if(len(points)) < 3:
            return complete_points
        
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        points_shape = np.shape(points);
        
        complete_points[:points_shape[0],:points_shape[1]] = points     
        
        if self.viz_image_cv2:        
            self.vis.clear_geometries()
            template_ = o3d.geometry.PointCloud() 
            template_.points = o3d.utility.Vector3dVector(complete_points)
            self.vis.add_geometry(template_)
            ctr = self.vis.get_view_control()
            ctr.rotate(-200,0)
            
            self.vis.poll_events()
            self.vis.update_renderer()
             
            cv2.imshow("Drone FPV", img_rgb)
            cv2.waitKey(1)
        
        return complete_points
           
           
    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset(self):
        self.airsim_client.simResetRace()
        self.stop_log_monitor_callback_thread()
        # self.stop_image_callback_thread()
        
        # if self.observation_type == "lidar":
        #     self.vis.destroy_window()

        self.previous_distance = 2  
        self.next_gate = 0
        self.has_collission = False
        self.has_finished = False
    
    
    def step(self, action):
        
        #print(action)
        x = np.clip(action[0], -self.max_axis_velocity, self.max_axis_velocity).astype(np.float) 
        y = np.clip(action[1], -self.max_axis_velocity, self.max_axis_velocity).astype(np.float)

        #Read current state of the drone
        drone_state = self.airsim_client_odom.getMultirotorState()                    
        #Get quaternion to calculate rotation angle in Z axis (yaw)  
        q = drone_state.kinematics_estimated.orientation
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val) , 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        
        #Rotate dimensions using rotation matrix in 2D
        x_rotated = x * math.cos(angle) - y * math.sin(angle)
        y_rotated = y * math.cos(angle) + x * math.sin(angle)    
        
        self.airsim_client.moveByVelocityZAsync(x_rotated
                                                , y_rotated
                                                , z = 2
                                                , duration = 2
                                                , drivetrain= airsim.DrivetrainType.ForwardOnly
                                                , yaw_mode= airsim.YawMode(is_rate=False)
                                                , vehicle_name=self.drone_name).join()
                
        done = False
        (reward, done) = self.calculate_reward()
     
        return self.get_observation(), reward, done
            

    