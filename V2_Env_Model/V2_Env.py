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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

class RobotEnv(gym.Env):
    def __init__(self, connect_type="GUI"):
        self.step_count = 0
        self.episodes = 0
        self.times_moved_forwards = 0
        self.moves = 0
        
        p.connect(p.DIRECT if connect_type == "DIRECT" else p.GUI)
        self.reset()
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        self.non_moves = 0
        self.prev_pos = []
        self.times_moved_forwards = 0
        self.moves = 0
        self.times_penalized = 0

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
        startPos = [random.randrange(-2, 3), random.randrange(-2, 3), 1.5] # (x, y, z)
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(f'{os.getcwd()}/V2_Env_Model/Model/robot.urdf',
                                startPos, startOrientation,
                                flags=flags, useFixedBase=False)
        p.changeDynamics(self.robot, -1, restitution=0, linearDamping=0.3, angularDamping=0.3, mass=50,
                         contactStiffness=1e5, contactDamping=1000, lateralFriction=5)
        p.setPhysicsEngineParameter(fixedTimeStep=0.002, maxNumCmdPer1ms=0)
        
        # Loading terrain
        self.terrain = self.createRandomHeightfield() # Load floor
        # Set friction, no bounce, no sinking (contactStiffness)
        p.changeDynamics(self.terrain, -1, contactStiffness=1e10, contactDamping=1e10, 
                         lateralFriction=5, spinningFriction=1.2, rollingFriction=0.5 , restitution=0) 
        
        # Initialize joints
        self.joint_poses = {}
        for i in range (p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_id, name, lower_lim, upper_lim, current_pos = joint_info[0], joint_info[1].decode("utf-8"), joint_info[8], joint_info[9], 0
            if name in "BL_J1 FR_J1 BR_J2 FL_J2 BL_J4 FR_J4":
                current_pos = upper_lim
            elif name in "BR_J1 FL_J1 BL_J2 FR_J2 BR_J4 FL_J4":
                current_pos = lower_lim
            self.joint_poses[name] = {"id": joint_id, "pos": current_pos, "lLim": lower_lim, "uLim": upper_lim}
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, self.joint_poses[name]["pos"])
        
        self.setJoints([0, 0, 0, 0, 0, 0, 0, 0])
        for i in range(200): # Wait for robot to touch the ground
            p.stepSimulation()
        self.prev_pos = p.getBasePositionAndOrientation(self.robot)[0][:2]
        
        # Training vars
        obs = self.getObservation()
        return obs, {}
        
    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        
        self.setJoints(action.tolist())
        p.stepSimulation()
        
        reward, terminated = self.evaluateReward(action)
        if terminated["state"]:
            print("TERMINATED:", terminated["reason"], "\t\tTimes Forward:", self.moves, "\t\tSteps ran:", self.step_count)
       
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
        
        # Base position and orientation (roll/pitch/yaw)
        base_pos, orn = p.getBasePositionAndOrientation(self.robot)
        obs.extend(base_pos) # (x, y, z)
        obs.extend(list(p.getEulerFromQuaternion(orn))) # (x, y, z)
        obs.extend(self.target_pos) # (x, y)
        
        # Velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        obs.extend(lin_vel) # (x, y, z)
        obs.extend(ang_vel) # (x, y, z)
        
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
        moving_joints = {}
        i = 0
        for name, info in self.joint_poses.items():
            if name in {"FL_J1", "FR_J1", "BL_J1", "BR_J1", "FL_J3", "FR_J3", "BL_J3", "BR_J3"}:
                self.joint_poses[name]["pos"] = min(info["uLim"], max(info["lLim"], self.joint_poses[name]["pos"] + actions[i]))
                p.setJointMotorControl2(self.robot, self.joint_poses[name]["id"], p.POSITION_CONTROL, self.joint_poses[name]["pos"], maxVelocity=1, force=1e8)
                moving_joints[info["id"]] = self.joint_poses[name]["pos"]
                i += 1
        
        # Run simulation steps until all joints reach targets
        for i in range(10):
            if len(moving_joints.items()) == 0:
                break
            joints_to_remove = []
            for joint_id, target in moving_joints.items():
                current_position = p.getJointState(self.robot, joint_id)[0]
                if abs(current_position - target) < tolerance:
                    joints_to_remove.append(joint_id)  # Mark joint as reached

            for joint_id in joints_to_remove:
                moving_joints.pop(joint_id)
            p.stepSimulation() 
    
    def evaluateReward(self, action):
        reward = 0
        terminated = {"state": False, "reason": ""}

        # Progress reward
        current_pos = p.getBasePositionAndOrientation(self.robot)[0][:2]
        progress = np.dot(np.array(self.target_pos) - np.array(self.prev_pos), np.array(current_pos) - np.array(self.prev_pos))
        if progress > 0:
            reward += 75 * progress * (1 + 0.1 * self.times_moved_forwards)
            self.times_moved_forwards += 1
            self.moves += 1
        else:
            self.times_moved_forwards *= 0.5
            reward += 25 * progress * (1 + min(self.episodes / 250, 4))

        reward -= 1.5 if round(progress, 3) < 0.01 else 0 
        self.prev_pos = copy.copy(current_pos)

        # Directional reward
        desired_angle = np.arctan2(self.target_pos[1] - current_pos[1], self.target_pos[0] - current_pos[0])
        angle_deviation = desired_angle - p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1])[2] # Subtracting yaw of robot
        angle_deviation = np.degrees((angle_deviation + np.pi) % (2 * np.pi) - np.pi)
        reward += 7.5 * max(0, 1 - abs(angle_deviation) / 70)
        
        # Balance reward (Penalizes excessive tilting)
        roll, pitch, _ = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1])
        balance_penalty = abs(np.degrees(roll)) + abs(np.degrees(pitch))
        reward -= 0.1 * balance_penalty  # Small penalty for tilting
        if len(p.getContactPoints(self.robot, self.terrain)) < 2:
            reward -= 5
        
        # Electricity cost
        electricity_cost = -0.25 * sum(list(map(abs, action)))
        reward -= electricity_cost
        
        # Joints at limit cost
        for name, info in self.joint_poses.items():
            reward -= 1 if (info["pos"] < info["lLim"] + 0.05 or info["pos"] > info["uLim"] - 0.05) else 0

        # Target reached
        if (self.target_pos[0] - 0.25 <= current_pos[0] <= self.target_pos[0] + 0.25) and (self.target_pos[1] - 0.25 <= current_pos[1] <= self.target_pos[1] + 0.25):
            reward += 1500
            terminated = {"state": True, "reason": "Target reached"}
            self.episodes += 1
         
        # Flipped check
        if self.isFlipped():
            terminated = {"state": True, "reason": "Flipped"}
            reward -= 500
            self.episodes += 1
            
        # Timeout
        if self.step_count > 1000:
            terminated = {"state": True, "reason": "Timeout"}
            self.episodes += 1
        
        return reward, terminated

    def isFlipped(self):
        _, orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)

        # If the roll or pitch is beyond 85 degrees, assume it's flipped
        flipped = abs(roll_deg) > 85 or abs(pitch_deg) > 85
        return flipped
    
    def close(self):
        p.disconnect()
        sys.exit()

if __name__ == "__main__":
    env = RobotEnv("GUI")
    
    obs, _ = env.reset()
    model = PPO.load(f'{os.getcwd()}/V2_Env_Model/V2B_PPOSpiderRobot.zip')

    for episode in range(10):
        obs, _ = env.reset() 
        episode_reward = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True) 
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
    
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.995, vf_coef=0.7, ent_coef=0.001, batch_size=128, tensorboard_log=f'{os.getcwd()}', device='auto', )
    model.learn(total_timesteps=2_500_000, progress_bar=True, callback=checkpoint_callback)
    model.save('PPOSpiderRobot')
    evaluate_policy(model, env, n_eval_episodes=10)
    """