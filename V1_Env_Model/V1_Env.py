"""
    V1 training
    Objective:  Training basic walking gait along relatively flat ground. Most effecient movement from
                point A to B, symmetrical movement to the best of its ability. 
"""

import os, math, random, time, sys, json, copy
import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import register, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
class RobotEnv(gym.Env):
    def __init__(self, connect_type="GUI"):
        self.step_count = 0
        self.episodes = 0
        self.times_moved_forwards = 0
        self.accumulated_episode_rewards = 0
        p.connect(p.DIRECT if connect_type == "DIRECT" else p.GUI)
        self.reset()
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        self.non_moves = 0
        self.prev_pos = []
        self.times_moved_forwards = 0
        self.times_penalized = 0
        self.accumulated_episode_rewards = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

        self.target_pos = (random.choice([random.uniform(-5, -3), random.uniform(3, 5)]), 
                           random.choice([random.uniform(-5, -3), random.uniform(3, 5)]))
        
        p.addUserDebugLine(
            lineFromXYZ=[self.target_pos[0], self.target_pos[1], 0],  # Start at ground level
            lineToXYZ=[self.target_pos[0], self.target_pos[1], 1],    # End slightly above
            lineColorRGB=[1, 0, 0],  # Red color
            lineWidth=2
        )
        self.step_count = 0
        
        # Loading robot
        startPos = [random.randrange(-2, 3), random.randrange(-2, 3), 3] # (x, y, z)
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(f'{os.getcwd()}/V1_Env_Model/Model/robot.urdf',
                                startPos, startOrientation,
                                flags=flags, useFixedBase=False)
        p.changeDynamics(self.robot, -1, restitution=0, linearDamping=0.3, angularDamping=0.3,
                         contactStiffness=1e5, contactDamping=1000, lateralFriction=5)
        p.setPhysicsEngineParameter(fixedTimeStep=0.002, maxNumCmdPer1ms=0)
        
        # Loading terrain
        self.terrain = self.createRandomHeightfield() # Load floor
        # Set friction, no bounce, no sinking (contactStiffness)
        p.changeDynamics(self.terrain, -1, contactStiffness=1e10, contactDamping=1e10, 
                         lateralFriction=5, spinningFriction=1.2, rollingFriction=0.5 , restitution=0) 
        
        # Initialize joints
        self.joints = {}
        for i in range (p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_id, name, lower_lim, upper_lim, current_pos = joint_info[0], joint_info[1].decode("utf-8"), joint_info[8], joint_info[9], 0
            if name in "BL_J1 FR_J1 BR_J2 FL_J2 BL_J4 FR_J4":
                current_pos = upper_lim
            elif name in "BR_J1 FL_J1 BL_J2 FR_J2 BR_J4 FL_J4":
                current_pos = lower_lim
            self.joints[name] = (joint_id, lower_lim, upper_lim, current_pos) # (Id, lower limit, upper limit, current pos)
            
        for i in range(30): # Wait for robot to touch the ground
            p.stepSimulation()
        self.prev_pos = p.getBasePositionAndOrientation(self.robot)[0][:2]
            
        # Training vars
        obs = self.getObservation()
        return obs, {}
        
    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        
        target_rots = self.setJoints(action.tolist())
        p.stepSimulation()
        
        current_pos, _ = p.getBasePositionAndOrientation(self.robot)
        current_pos = current_pos[:2]
        current_x, current_y = current_pos
        if (self.target_pos[0] - 0.25 <= current_x <= self.target_pos[0] + 0.25) and (self.target_pos[1] - 0.25 <= current_y <= self.target_pos[1] + 0.25):
            terminated = True
            self.episodes += 1
            print("TERMINATED: Target reached  ", "  Times Forward:", self.times_moved_forwards, "   Steps ran: ", self.step_count)
        
        if (self.step_count >= 1000):
            terminated = True
            self.episodes += 1
            print("TERMINATED: Timeout  ", "  Times Forward:", self.times_moved_forwards, "   Steps ran: ", self.step_count)
            
        if self.isFlipped():
            terminated = True
            self.episodes += 1
            print("TERMINATED: Flipped  ", "  Times Forward:", self.times_moved_forwards, "   Steps ran: ", self.step_count)
        
        reward = self.evaluateReward()
        self.accumulated_episode_rewards += reward
        
        if self.non_moves >= 5:
            terminated = True
            self.episodes += 1
            print("TERMINATED: Stationary  ", "Times Forward:", self.times_moved_forwards, "   Steps ran: ", self.step_count)
        
        self.prev_pos = copy.copy(current_pos)
        obs = self.getObservation()
        return obs, reward, terminated, False, {}
    
    def getObservation(self):
        obs = []
        # Joint angles
        for joint_id in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, joint_id)[1].decode("utf-8") in {"FL_J1", "FR_J1", "BL_J1", "BR_J1", "FL_J3", "FR_J3", "BL_J3", "BR_J3"}:
                obs.append(p.getJointState(self.robot, joint_id)[0])
        
        # Target position, base position and orientation (roll/pitch/yaw)
        base_pos, orn = p.getBasePositionAndOrientation(self.robot)
        obs.extend(base_pos[:2])
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        obs.extend([roll, pitch, yaw])
        obs.extend(self.target_pos)
        
        # Velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        obs.extend([lin_vel[0], lin_vel[1]]) # (x, y)
        obs.append(ang_vel[2]) # Yaw angular vel
        
        if np.isnan(obs).any():
            print("NAN DETECTED", obs)
        
        return np.array(obs, dtype=np.float32)
        
    def createRandomHeightfield(self):
        """Creates a random heightfield to replace the flat plane."""
        # heightfield_data = np.random.uniform(-0.05, 0.05, 128 * 128).astype(np.float32) 
        heightfield_data = np.zeros(128 * 128, dtype=np.float32)  # Completely flat surface
        terrain_collision = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[1.5, 1.5, 5],  # Adjust scale for realistic terrain
            heightfieldTextureScaling=128,
            heightfieldData=heightfield_data,
            numHeightfieldRows=128,
            numHeightfieldColumns=128
        )
        terrain_body = p.createMultiBody(0, terrain_collision)
        p.resetBasePositionAndOrientation(terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.changeVisualShape(terrain_body, -1, textureUniqueId=-1, rgbaColor=[0.85, 0.85, 0.85, 1])  # Set color

        return terrain_body

    def setJoints(self, actions, tolerance=0.01):
        moving_joints = {}  # Track joints still in motion
        target_rots = {}
        
        i = 0
        for name, (joint_id, lower_lim, upper_lim, current_pos) in self.joints.items():
            if name in {"FL_J1", "FR_J1", "BL_J1", "BR_J1", "FL_J3", "FR_J3", "BL_J3", "BR_J3"}:
                target_rots[name] = (upper_lim + lower_lim)/2 + actions[i] * ((upper_lim - lower_lim) / 2)
                i += 1
            else:
                target_rots[name] = current_pos

            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, target_rots[name])
            moving_joints[joint_id] = target_rots[name]

        # Run simulation steps until all joints reach targets
        for i in range(50):
            joints_to_remove = []
            for joint_id, target in moving_joints.items():
                current_position = p.getJointState(self.robot, joint_id)[0]
                if abs(current_position - target) < tolerance:
                    joints_to_remove.append(joint_id)  # Mark joint as reached

            for joint_id in joints_to_remove:
                moving_joints.pop(joint_id)
            p.stepSimulation() 
     
        for name in target_rots:
            current_pos = target_rots[name]
            joint_id, lower_lim, upper_lim, _ = self.joints[name]
            self.joints[name] = (joint_id, lower_lim, upper_lim, current_pos)
        
        return target_rots
        
    def evaluateBalanceScore(self):
        # Compute CoM
        com_pos = np.array([0., 0., 0.])
        total_mass = 0

        for i in range(-1, p.getNumJoints(self.robot)):
            if i == -1:
                pos, _ = p.getBasePositionAndOrientation(self.robot)
            else:
                pos = p.getLinkState(self.robot, i)[0]

            mass = p.getDynamicsInfo(self.robot, i)[0]
            com_pos += np.array(pos) * mass
            total_mass += mass

        com_pos = com_pos / total_mass if total_mass > 0 else np.array([0., 0., 0.])
        com_x, com_y, _ = com_pos
        
        contacts = p.getContactPoints(self.robot, self.terrain)
        if len(contacts) <= 1: # Penalize if not all arms are touching ground
            return -5 
        contact_positions = [contact[5] for contact in contacts]
        contact_x = [pos[0] for pos in contact_positions]
        contact_y = [pos[1] for pos in contact_positions]
        min_x, max_x = min(contact_x), max(contact_x)
        min_y, max_y = min(contact_y), max(contact_y)

        # Check if CoM is inside the bounding box of contact points
        balance_score = -1 # Penalize if outside area
        if min_x <= com_x <= max_x and min_y <= com_y <= max_y:
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            max_dist = max(max_x - min_x, max_y - min_y) / 2
            current_dist = np.linalg.norm([com_x - center_x, com_y - center_y])
            balance_score = 1 - current_dist / max_dist

        return round(balance_score, 2)
    
    def evaluateReward(self):
        reward = 0
      
        current_pos, orn = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y = current_pos[:2]
        target_x, target_y = self.target_pos
        _, _, robot_yaw = p.getEulerFromQuaternion(orn)  # Extract yaw (in radians)
        
        # Directional
        desired_angle = np.arctan2(target_y - robot_y, target_x - robot_x)
        angle_deviation = desired_angle - robot_yaw
        angle_deviation = (angle_deviation + np.pi) % (2 * np.pi) - np.pi
        angle_deviation_degrees = np.degrees(angle_deviation) # angle deviation -25 to 25 is acceptable, where 0 = facing direct
                
        # Closing in / moving away
        current_distance = self.distance(self.target_pos, current_pos[:2])
        prev_distance = self.distance(self.target_pos, self.prev_pos)

        # Reward for turning toward the target (smooth scaling up to 90°)
        direction_reward = max(0, 1 - abs(angle_deviation_degrees) / 90)
        reward += 5 * direction_reward
            
        if prev_distance > current_distance + 0.1:
            self.times_moved_forwards += 1
            reward += 150 * (prev_distance - current_distance)
        
        if round(prev_distance, 3) != round(current_distance, 3):
            # Balance rewards, ONLY if the robot is moving forwards
            reward -= 5 if len(p.getContactPoints(self.robot, self.terrain)) < 1 else 0
            reward +=  1.25 * self.evaluateBalanceScore() # Balance score out of 10, * 0.5
        else:
            print('!!!')
            reward -= 10
            self.non_moves += 1
        
        reward -= 0.1
        reward -= 700 if self.isFlipped() else 0 # Penalty for flipping over

        if current_distance < 0.2:
            reward += 1500  # Large final reward

        return reward

    def isFlipped(self):
        _, orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)

        # If the roll or pitch is beyond 85 degrees, assume it's flipped
        flipped = abs(roll_deg) > 85 or abs(pitch_deg) > 85
        return flipped

    def distance(self, coord1, coord2):
        (x1, y1), (x2, y2) = coord1, coord2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def close(self):
        p.disconnect()
        sys.exit()

if __name__ == "__main__":
    env = RobotEnv("GUI")
    
    obs, _ = env.reset()
    model = PPO.load(f'{os.getcwd()}/V1_Env_Model/V1B_PPOSpiderRobot.zip')

    for episode in range(10):
        obs, _ = env.reset() 
        episode_reward = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=False) 
            print(action)
            obs, reward, terminated, _, _ = env.step(action) 
            print(reward)
            time.sleep(0.01)

            if terminated:
                print(f"Episode {episode + 1} finished")
                break
    env.close()
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=os.getcwd(),
        name_prefix="PPO"
    )
    
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, vf_coef=0.6, ent_coef=0.0015, tensorboard_log=f'{os.getcwd()}', device='auto', )
    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=checkpoint_callback)
    model.save('PPOSpiderRobot')
    evaluate_policy(model, env, n_eval_episodes=10)
    """
    # tensorboard --logdir=PPO_2