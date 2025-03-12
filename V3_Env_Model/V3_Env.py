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
#from stable_baselines3 import PPO
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.callbacks import CheckpointCallback
#from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, VecNormalize
#from stable_baselines3.common.monitor import Monitor

class RobotEnv(gym.Env):
    def __init__(self, connect_type="DIRECT"):
        self.step_count = 0
        self.total_steps = 0
        self.episodes = 0

        p.connect(p.DIRECT if connect_type == "DIRECT" else p.GUI)
        self.reset()
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        self.prev_pos = []
        self.step_count = 0

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

        # Target visualization
        self.target_pos = (random.choice([random.uniform(-15, -10), random.uniform(10, 15)]), 
                           random.choice([random.uniform(-15, -10), random.uniform(10, 15)]))
        p.addUserDebugLine(
            lineFromXYZ=[self.target_pos[0], self.target_pos[1], 0], 
            lineToXYZ=[self.target_pos[0], self.target_pos[1], 1], 
            lineColorRGB=[1, 0, 0],
            lineWidth=2
        )
        
        # Loading robot
        startPos = [random.randrange(-2, 3), random.randrange(-2, 3), 1.5] # (x, y, z)
        yaw_angle = np.arctan2(self.target_pos[1] - startPos[1], self.target_pos[0] - startPos[0]) + (np.pi / 4) + np.pi
        startOrientation = p.getQuaternionFromEuler([0, 0, yaw_angle])
        flags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(f'{os.getcwd()}/V3_Env_Model/Model/robot.urdf',
                                startPos, startOrientation,
                                flags=flags, useFixedBase=False)
        p.changeDynamics(self.robot, -1, restitution=0, linearDamping=1, angularDamping=1,
                         contactStiffness=5e10, contactDamping=1000, lateralFriction=10, rollingFriction=5, spinningFriction=5)
        p.setPhysicsEngineParameter(fixedTimeStep=0.002, maxNumCmdPer1ms=0, contactBreakingThreshold=0.001)
        
        # Loading terrain
        self.terrain = self.createRandomHeightfield()
        # Set friction, no bounce, no sinking (contactStiffness)
        p.changeDynamics(self.terrain, -1, contactStiffness=5e10, contactDamping=1000, 
                         lateralFriction=10, spinningFriction=5, rollingFriction=5, restitution=0) 
        
        # Initialize joints
        self.joint_poses = {}
        for i in range (p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_id, name, lower_lim, upper_lim, current_pos = joint_info[0], joint_info[1].decode("utf-8"), joint_info[8], joint_info[9], 0
            
            if name in "BL_J4 FR_J4 BL_J1 BR_J1":
                current_pos = upper_lim
            elif name in "BL_J2 FR_J2":
                current_pos = np.radians(5)
            elif name in "BR_J2 FL_J2":
                current_pos = np.radians(-5)
            elif name in "BR_J4 FL_J4 FL_J1 FR_J1":
                current_pos = lower_lim
          
            self.joint_poses[name] = {"id": joint_id, "pos": current_pos, "lLim": lower_lim, "uLim": upper_lim}
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, self.joint_poses[name]["pos"], maxVelocity=10, force=1e6)

        for i in range(250): # Wait for robot to touch the ground
            p.stepSimulation()
        self.prev_pos = p.getBasePositionAndOrientation(self.robot)[0][:2]
        
        # Training vars
        obs = self.getObservation()
        return obs, {}

    def applyDefinedGait(self):
        def get_joint_info(name):
            midpoint = (self.joint_poses[name]["lLim"] + self.joint_poses[name]["uLim"])/2
            lower = self.joint_poses[name]["lLim"]
            upper = self.joint_poses[name]["uLim"]
            return midpoint, lower, upper
        
        def FL_J1_map():
            m, f, b = get_joint_info("FL_J1")
            return [f, f, b, b, f, f, b, b] 
        def FR_J1_map():
            m, b, f = get_joint_info("FR_J1")
            return [b, b, f, f, b, b, f, f]  
        def BL_J1_map():
            m, f, b = get_joint_info("BL_J1")
            return [b, b, f, f, b, b, f, f]  
        def BR_J1_map():
            m, b, f = get_joint_info("BR_J1")
            return [f, f, b, b, f, f, b, b] 
        def FL_J3_map():
            d, _, u = get_joint_info("FL_J3")
            return [u, d, d, d, u, d, d, d] 
        def FR_J3_map():
            d, u, _ = get_joint_info("FR_J3")
            return [d, d, u, d, d, d, u, d] 
        def BL_J3_map():
            d, u, _ = get_joint_info("BL_J3")
            return [d, d, u, d, d, d, u, d] 
        def BR_J3_map():
            d, _, u = get_joint_info("BR_J3")
            return [u, d, d, d, u, d, d, d] 

        all_actions = {
            "FL_J1": FL_J1_map(),  # f, f, b, b, f, f, b, b
            "FL_J3": FL_J3_map(),  # u, d, d, d, u, d, d, d
            "BR_J1": BR_J1_map(),  # f, f, b, b, f, f, b, b
            "BR_J3": BR_J3_map(),  # u, d, d, d, u, d, d, d
            "FR_J1": FR_J1_map(),  # b, b, f, f, b, b, f, f
            "FR_J3": FR_J3_map(),  # d, d, u, d, d, d, u, d
            "BL_J1": BL_J1_map(),  # b, b, f, f, b, b, f, f
            "BL_J3": BL_J3_map(),  # d, d, u, d, d, d, u, d
        }

        taking_actions = {
            "FL_J1": all_actions["FL_J1"][self.step_count % 8],
            "FR_J1": all_actions["FR_J1"][self.step_count % 8],
            "BL_J1": all_actions["BL_J1"][self.step_count % 8],
            "BR_J1": all_actions["BR_J1"][self.step_count % 8],
            "FL_J3": all_actions["FL_J3"][self.step_count % 8],
            "FR_J3": all_actions["FR_J3"][self.step_count % 8],
            "BL_J3": all_actions["BL_J3"][self.step_count % 8],
            "BR_J3": all_actions["BR_J3"][self.step_count % 8],
        }
        return taking_actions

    def applyDefinedReset(self):
        def get_joint_info(name):
            """Returns joint midpoint, lower limit, and upper limit."""
            midpoint = (self.joint_poses[name]["lLim"] + self.joint_poses[name]["uLim"]) / 2
            lower = self.joint_poses[name]["lLim"]
            upper = self.joint_poses[name]["uLim"]
            return midpoint, lower, upper

        def FL_J1_map():
            m, f, b = get_joint_info("FL_J1")
            return [m, m, m, m, m, m, m, m]  # m, m, m, m, m

        def FL_J3_map():
            d, _, u = get_joint_info("FL_J3")
            return [u, d, d, d, d, d, d, d]  # u, d, d, d, d

        def BR_J1_map():
            c = self.joint_poses["BR_J1"]["pos"]
            m, _, _ = get_joint_info("BR_J1")
            return [c, m, m, m, m, m, m, m]  # c, m, m, m, m

        def BR_J3_map():
            d, _, u = get_joint_info("BR_J3")
            return [d, u, d, d, d, d, d, d]  # d, u, d, d, d

        def FR_J1_map():
            c = self.joint_poses["FR_J1"]["pos"]
            m, _, _ = get_joint_info("FR_J1")
            return [c, c, m, m, m, m, m, m]  # c, c, m, m, m

        def FR_J3_map():
            d, u, _ = get_joint_info("FR_J3")
            return [d, d, u, d, d, d, d, d]  # d, d, u, d, d

        def BL_J1_map():
            c = self.joint_poses["BL_J1"]["pos"]
            m, _, _ = get_joint_info("BL_J1")
            return [c, c, c, m, m, m, m, m]  # c, c, c, m, m

        def BL_J3_map():
            d, u, _ = get_joint_info("BL_J3")
            return [d, d, d, u, d, d, d, d]  # d, d, d, u, d

        # Dictionary containing all joint movement mappings
        all_actions = {
            "FL_J1": FL_J1_map(),  # m, m, m, m, m
            "FL_J3": FL_J3_map(),  # u, d, d, d, d
            "BR_J1": BR_J1_map(),  # c, m, m, m, m
            "BR_J3": BR_J3_map(),  # d, u, d, d, d
            "FR_J1": FR_J1_map(),  # c, c, m, m, m
            "FR_J3": FR_J3_map(),  # d, d, u, d, d
            "BL_J1": BL_J1_map(),  # c, c, c, m, m
            "BL_J3": BL_J3_map(),  # d, d, d, u, d
        }

        for i in range (len(all_actions["FL_J1"])):
            taking = {}
            for key, val in all_actions.items():
                taking[key] = val[i]
            self.setJoints(taking)
    
    def applyDefinedTurn(self, degree):
        def get_joint_info(name):
            """Returns joint midpoint, lower limit, and upper limit."""
            midpoint = (self.joint_poses[name]["lLim"] + self.joint_poses[name]["uLim"]) / 2
            lower = self.joint_poses[name]["lLim"]
            upper = self.joint_poses[name]["uLim"]
            return midpoint, lower, upper

        initial_poses = copy.deepcopy({
            "FL_J1": self.joint_poses["FL_J1"]["pos"],
            "FR_J1": self.joint_poses["FR_J1"]["pos"],
            "BR_J1": self.joint_poses["BR_J1"]["pos"],
            "BL_J1": self.joint_poses["BL_J1"]["pos"]
        })

        def FL_J1_map():
            c = initial_poses["FL_J1"]
            t = c - float(np.radians(25 * degree))
            m, f, b = get_joint_info("FL_J1")
            return [m, t, t, t, t, t, m]  # m, m, m, m, m

        def FL_J3_map():
            d, _, u = get_joint_info("FL_J3")
            return [u, u, d, d, d, d, d]  # u, d, d, d, d

        def BR_J1_map():
            c = initial_poses["BR_J1"]
            t = c - float(np.radians(25 * degree))
            m, _, _ = get_joint_info("BR_J1")
            return [m, t, t, t, t, t, m]  # c, m, m, m, m

        def BR_J3_map():
            d, _, u = get_joint_info("BR_J3")
            return [u, u, d, d, d, d, d]  # d, u, d, d, d

        def FR_J1_map():
            c = initial_poses["FR_J1"]
            t = c - float(np.radians(25 * degree))
            m, _, _ = get_joint_info("FR_J1")
            return [m, m, m, m, t, t, m]  # c, c, m, m, m

        def FR_J3_map():
            d, u, _ = get_joint_info("FR_J3")
            return [d, d, d, u, u, d, d]  # d, d, u, d, d

        def BL_J1_map():
            c = initial_poses["BL_J1"]
            t = c - float(np.radians(25 * degree))
            m, _, _ = get_joint_info("BL_J1")
            return [m, m, m, m, t, t, m]  # c, c, c, m, m

        def BL_J3_map():
            d, u, _ = get_joint_info("BL_J3")
            return [d, d, d, u, u, d, d]  # d, d, d, u, d

        # Dictionary containing all joint movement mappings
        all_actions = {
            "FL_J1": FL_J1_map(),  # m, t, t, t, t, t, m
            "FL_J3": FL_J3_map(),  # u, u, d, d, d, d, d
            "BR_J1": BR_J1_map(),  # m, t, t, t, t, t, m
            "BR_J3": BR_J3_map(),  # u, u, d, d, d, d, d
            "FR_J1": FR_J1_map(),  # m, m, m, m, t, t, m
            "FR_J3": FR_J3_map(),  # d, d, d, u, u, d, d
            "BL_J1": BL_J1_map(),  # m, m, m, m, t, t, m
            "BL_J3": BL_J3_map(),  # d, d, d, u, u, d, d
        }

        for i in range (len(all_actions["FL_J1"])):
            taking = {}
            for key, val in all_actions.items():
                taking[key] = val[i]
            self.setJoints(taking)
        
    def step(self, action):
        self.total_steps += 1
        self.step_count += 1
        reward = 0
        terminated = False
        
        predefined_actions = self.applyDefinedGait()
        self.setJoints(predefined_actions)
        self.applyDefinedReset()
        for i in range(10):
            self.applyDefinedTurn(-1)       
         
        reward, terminated = self.evaluateReward()
        if terminated["state"]:
            print("TERMINATED:", terminated["reason"], "\t\tSteps ran:", self.step_count)
       
        obs = self.getObservation()
        return obs, reward, terminated["state"], False, {}
    
    def getObservation(self):
        obs = []
        
        # Joint angles
        for joint_id in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, joint_id)[1].decode("utf-8") in {"FL_J1", "FR_J1", "BL_J1", "BR_J1", "FL_J3", "FR_J3", "BL_J3", "BR_J3"}:
                joint_pos, joint_vel, *_ = p.getJointState(self.robot, joint_id)
                obs.append(joint_pos)
                obs.append(joint_vel)
        
        # Base orientation (roll/pitch/yaw)
        base_pos, orn = p.getBasePositionAndOrientation(self.robot)
        obs.extend(list(p.getEulerFromQuaternion(orn))) # (x, y, z)
                
        # Distance
        current_pos = p.getBasePositionAndOrientation(self.robot)[0][:2]
        current_distance = math.sqrt((current_pos[0] - self.target_pos[0])**2 + (current_pos[1] - self.target_pos[1])**2)
        obs.append(current_distance)
        
        # Directional 
        desired_angle = np.arctan2(self.target_pos[1] - current_pos[1], self.target_pos[0] - current_pos[0])
        obs.append(np.cos(desired_angle))  # X-component of direction
        obs.append(np.sin(desired_angle))  # Y-component of direction
        
        if np.isnan(obs).any():
            print("NAN DETECTED", obs)
        
        return np.array(obs, dtype=np.float32)
        
    def createRandomHeightfield(self):
        """Creates a random heightfield to replace the flat plane."""
        # heightfield_data = np.random.uniform(-0.05, 0.05, 128 * 128).astype(np.float32) 
        heightfield_data = np.zeros(128 * 128, dtype=np.float32)  # Completely flat surface
        terrain_collision = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[1.1, 1.1, 5],  # Adjust scale for realistic terrain
            heightfieldTextureScaling=1024,
            heightfieldData=heightfield_data,
            numHeightfieldRows=128,
            numHeightfieldColumns=128
        )
        terrain_body = p.createMultiBody(0, terrain_collision)
        p.resetBasePositionAndOrientation(terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.changeVisualShape(terrain_body, -1, textureUniqueId=-1, rgbaColor=[0.85, 0.85, 0.85, 1])  # Set color

        return terrain_body

    def setJoints(self, predefined_actions, tolerance=0.01):
        moving_joints = {} # Track joints still in motion
        target_rots = {}
        
        i = 0
        for name, info in predefined_actions.items():
            if name in {"FL_J1","FR_J1", "BL_J1", "BR_J1", "FL_J3", "FR_J3", "BL_J3", "BR_J3"}:
                target_rots[name] = predefined_actions[name]
            else:
                target_rots[name] = info["pos"]
            # if name in {"FL_J2", "FR_J2", "BL_J2", "BR_J2"}:
                # target_rots[name] = info["lLim"] + ((list(model_actions)[i] + 1) / 2) * (info["uLim"] - info["lLim"])
                # target_rots[name] = min(info["uLim"], max(info["lLim"], (list(model_actions)[i] * 0.5 + target_rots[name])))
                # i += 1
            moving_joints[self.joint_poses[name]["id"]] = target_rots[name]
            p.setJointMotorControl2(self.robot, self.joint_poses[name]["id"], p.POSITION_CONTROL, target_rots[name], maxVelocity=1.5, force=1e8)

        # Run simulation steps until all joints reach targets
        while True:
            joints_to_remove = []
            for joint_id, target in moving_joints.items():
                current_position = p.getJointState(self.robot, joint_id)[0]
                if abs(current_position - target) < tolerance:
                    joints_to_remove.append(joint_id)  # Mark joint as reached

            # Remove reached joints iteration
            for joint_id in joints_to_remove:
                del moving_joints[joint_id]

            p.stepSimulation()
            if len(moving_joints.items()) == 0:
                break
        for name in target_rots:
            self.joint_poses[name]["pos"] = target_rots[name]
      
    def evaluateReward(self):
        reward = 0
        terminated = {"state": False, "reason": ""}
        
        # Current & previous position
        current_pos = p.getBasePositionAndOrientation(self.robot)[0][:2]
        height = p.getBasePositionAndOrientation(self.robot)[0][2]
        prev_distance = np.linalg.norm(np.array(self.prev_pos) - np.array(self.target_pos))
        current_distance = np.linalg.norm(np.array(current_pos) - np.array(self.target_pos))
        progress = prev_distance - current_distance  # Positive if moving toward the target

        # Heading direction reward
        desired_angle = np.arctan2(self.target_pos[1] - current_pos[1], self.target_pos[0] - current_pos[0])
        robot_yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1])[2]
        angle_deviation = (desired_angle - robot_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        direction_reward = np.cos(angle_deviation)  # -1 (wrong way) to 1 (perfect alignment)
        
        # **Heavily weighted direction reward**
        reward += 30 * direction_reward  

        # **Progress reward (higher weight)**
        reward += 15 * progress  

        # **Penalty for moving in the wrong direction**
        if progress < -0.002:  # If it's actively moving away
            reward -= 30 * abs(progress)

       # Balance reward & penalty (encourages stable posture)
        roll, pitch, _ = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1])
        roll_deg, pitch_deg = abs(np.degrees(roll)), abs(np.degrees(pitch))
        # **Reward for staying balanced (low roll/pitch)**
        balance_reward = max(0, 25 - 0.05 * (roll_deg + pitch_deg))  # Full reward if tilt is low
        reward += balance_reward
        # **Penalty for extreme tilting**
        if roll_deg > 25 or pitch_deg > 25:  # Heavier penalty beyond 25 degrees
            reward -= min(20, 0.1 * (roll_deg + pitch_deg))
        # **Stronger penalty if dangerously unstable (likely to flip)**
        if roll_deg > 50 or pitch_deg > 50:
            reward -= 20  # Heavily discourage extreme instability

        # **Big reward for reaching target**
        if current_distance < 0.25:
            reward += 2000 + 1.5 * (500 - self.step_count)
            terminated = {"state": True, "reason": "Target reached"}

        # **Flipped penalty (heavily discouraged)**
        contacts = p.getContactPoints(self.robot, self.terrain, -1)
        if self.isFlipped() or len(contacts) > 0:
            reward -= 600
            terminated = {"state": True, "reason": "Flipped/Base"}

        # **Timeout penalty**
        if self.step_count >= 500:
            reward -= 150
            terminated = {"state": True, "reason": "Timeout"}

        # Small step cost (reduces unnecessary movement)
        reward -= 0.005 * self.step_count

        self.prev_pos = copy.copy(current_pos)
        return reward, terminated


    def isFlipped(self):
        _, orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)

        # If the roll or pitch is beyond 85 degrees, assume it's flipped
        flipped = abs(roll_deg) > 35 or abs(pitch_deg) > 35
        return flipped
    
    def close(self):
        p.disconnect()
        
register(
    id="PPOSpiderRobot",
    entry_point="V2_Env:RobotEnv",
    max_episode_steps=500,
)
def make_env():
    def _init():
        return RobotEnv("DIRECT")
    return _init

if __name__ == "__main__":
    env = RobotEnv("GUI")
    while True:
        env.step(None)
    """
    env = RobotEnv("GUI")
    obs, _ = env.reset()
    model = PPO.load(f'{os.getcwd()}/PPO_320000_steps.zip')

    for episode in range(10):
        obs, _ = env.reset() 
        episode_reward = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=False) 
            obs, reward, terminated, _, _ = env.step(action) 
            time.sleep(0.01)

            if terminated:
                print(f"Episode {episode + 1} finished")
                break
    env.close()
    """
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000, 
        save_path=os.getcwd(),
        name_prefix="PPO"
    )
    policy_kwargs = dict(net_arch=[256, 256, 128])
    num_envs = 8 # Number of parallel environments
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    vec_env = VecMonitor(vec_env, filename="logs/multi_env_log")  # Monitor for logging

    #old_model = PPO.load(f'{os.getcwd()}/old_PPO_160000_steps.zip', env=vec_env)
    model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, ent_coef=0.0003, vf_coef=0.4, learning_rate=0.00025, tensorboard_log=f'{os.getcwd()}', device='auto')
    #model.set_parameters(old_model.get_parameters())
    
    model.learn(total_timesteps=500_000, progress_bar=False, callback=checkpoint_callback)
    model.save('PPOSpiderRobot')
    evaluate_policy(model, vec_env, n_eval_episodes=10)
    """
    
    