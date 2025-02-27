""" 
    V1 training
    Objective:  Training basic walking gait along relatively flat ground. Most effecient movement from
                point A to B, symmetrical movement to the best of its ability. 
"""

import os, math, random, time, sys
import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import register, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env

class RobotEnv(gym.Env):
    def __init__(self):
        self.step_count = 0
        self.episodes = 0
        p.connect(p.DIRECT)
        self.reset()
        self.action_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        print("EPISODES:", self.episodes)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])
    
        self.target_pos = (random.uniform(1, 10), random.randint(1, 10))
        self.step_count = 0
        
        # Loading robot
        startPos = [0, 0, 3]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(f'{os.getcwd()}/Model/robot.urdf',
                                startPos, startOrientation,
                                flags=flags, useFixedBase=False)
        p.changeDynamics(self.robot, -1, restitution=0, linearDamping=0.3, angularDamping=0.3, mass=50)
        p.setPhysicsEngineParameter(fixedTimeStep=0.002, maxNumCmdPer1ms=0)
        
        # Loading terrain
        self.terrain = self.createRandomHeightfield() # Load floor
        # Set friction, no bounce, no sinking (contactStiffness)
        p.changeDynamics(self.terrain, -1, contactStiffness=math.inf, contactDamping=math.inf, 
                         lateralFriction=0.8, spinningFriction=0.6, rollingFriction=0.1, restitution=0) 
        
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
            
        # Training vars
        obs = self.getObservation()
        return obs, {}
        
    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        
        target_rots = self.setJoints(action.tolist())
        p.stepSimulation()
        
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y = pos[0], pos[1]
        if (self.target_pos[0] - 2 <= robot_x <= self.target_pos[0] + 2) and (self.target_pos[1] - 2 <= robot_y <= self.target_pos[1] + 2):
            terminated = True
            self.episodes += 1
            print("Target reached")
        
        if (self.step_count >= 500) or self.isFlipped():
            terminated = True
            self.episodes += 1
            
        # Reward
        directionScore, isMovingTowards = self.evaluateDirectionScore() 
        reward += self.evaluateBalanceScore()  + self.evaluateEffeciencyScore(target_rots) + + self.evaluateSpeedScore(isMovingTowards) + directionScore
        obs = self.getObservation()
        return obs, reward, terminated, False, {}
    
    def getObservation(self):
        obs = []
        # Joint angles
        for joint_id in range(p.getNumJoints(self.robot)):
            obs.append(p.getJointState(self.robot, joint_id)[0])
        
        # Target position, base position and orientation (roll/pitch/yaw)
        base_pos, orn = p.getBasePositionAndOrientation(self.robot)
        obs.extend(base_pos)
        obs.extend(p.getEulerFromQuaternion(orn))
        obs.extend(self.target_pos)
        
        # Body angular velocity
        _, angular_vel = p.getBaseVelocity(self.robot)
        obs.extend(angular_vel)
        
        # Num of contacts with ground
        contacts = p.getContactPoints(self.robot, self.terrain)
        num_contacts = len(set(contact[4] for contact in contacts))
        obs.append(num_contacts)
        
        return np.array(obs, dtype=np.float32)
        
    def createRandomHeightfield(self):
        """Creates a random heightfield to replace the flat plane."""
        heightfield_data = np.random.uniform(-0.05, 0.05, 128 * 128).astype(np.float32)  # Small variations
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
        target_rots  ={}
        
        i = 0
        for name, (joint_id, lower_lim, upper_lim, current_pos) in self.joints.items():
            target_rots[name] = lower_lim + actions[i] * (upper_lim - lower_lim)
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, target_rots[name])
            moving_joints[joint_id] = target_rots[name]
            i += 1

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
        if len(contacts) < 3: # Penalize if not all arms are touching ground
            return -5 
        contact_positions = [contact[5] for contact in contacts]
        contact_x = [pos[0] for pos in contact_positions]
        contact_y = [pos[1] for pos in contact_positions]
        min_x, max_x = min(contact_x), max(contact_x)
        min_y, max_y = min(contact_y), max(contact_y)

        # Check if CoM is inside the bounding box of contact points
        balance_score = -5 # Penalize if outside area
        if min_x <= com_x <= max_x and min_y <= com_y <= max_y:
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            max_dist = max(max_x - min_x, max_y - min_y) / 2
            current_dist = np.linalg.norm([com_x - center_x, com_y - center_y])
            balance_score = max(1, 10 * (1 - current_dist / max_dist))

        return round(balance_score, 2)

    def evaluateEffeciencyScore(self, targets):
        score = 0
        for name, (_, _, _, current_pos) in self.joints.items():
            score -= abs(targets[name] - current_pos)
        return score

    def evaluateDirectionScore(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        velocity, _ = p.getBaseVelocity(self.robot)

        target_x, target_y = self.target_pos
        robot_x, robot_y = pos[0], pos[1]
        vel_x, vel_y = velocity[0], velocity[1]

        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        movement_vector = np.array([vel_x, vel_y])

        if np.linalg.norm(movement_vector) == 0:
            return -10  # No movement at all

        target_vector /= np.linalg.norm(target_vector)
        movement_vector /= np.linalg.norm(movement_vector)

        dot_product = np.dot(target_vector, movement_vector)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if angle_deg > 90:
            return False, -10  # Moving away
        elif angle_deg < 30:
            return True, 10 - (angle_deg / 3)  # Smooth scaling from 10 to 1
        else:
            return False, max(1, 5 - (angle_deg - 30) / 15)  # Gradual decrease

    def evaluateSpeedScore(self, isMovingTowards):
        velocity, _ = p.getBaseVelocity(self.robot)
        speed = np.linalg.norm([velocity[0], velocity[1]])

        if not isMovingTowards:
            return -5  # Penalize moving in the wrong direction

        return min(10, 10 * (1 - np.exp(-speed / 2))) 

    def isFlipped(self):
        _, orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)  # Convert to Euler angles

        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)

        # If the roll or pitch is beyond 85 degrees, assume it's flipped
        flipped = abs(roll_deg) > 85 or abs(pitch_deg) > 85
        if flipped:
            print("Flipped")
        return flipped

    def close(self):
        p.disconnect()
        sys.exit()

if __name__ == "__main__":
    env = RobotEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000)
    model.save('PPOSpiderRobot')
    evaluate_policy(model, env, n_eval_episodes=10)