""" 
    V1 training
    Objective:  Training basic walking gait along relatively flat ground. Most effecient movement from
                point A to B, symmetrical movement to the best of its ability. 
"""

import os, math, random, time
import numpy as np
import pybullet as p
#from stable_baselines3 import PPO
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.evaluation import evaluate_policy

class RobotEnv():
    def __init__(self):
        p.connect(p.GUI)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])
        p.setGravity(0, 0, -9.81)
        
        # Loading robot
        startPos = [0, 0, 3]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(f'{os.getcwd()}/Model/robot.urdf',
                                startPos, startOrientation,
                                flags=flags, useFixedBase=False)
        p.changeDynamics(self.robot, -1, restitution=0, linearDamping=0.3, angularDamping=0.3)
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

        # Training vars
        self.step_count = 0
    
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

    def setJoints(self, targets, tolerance=0.01):
        moving_joints = {}  # Track joints still in motion

        for name, (joint_id, lower_lim, upper_lim, current_pos) in self.joints.items():
            targets[name] = lower_lim if random.randint(0, 1) == 0 else upper_lim
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, targets[name])
            moving_joints[joint_id] = targets[name]

        # Run simulation steps until all joints reach targets
        for i in range(50):
            joints_to_remove = []
            for joint_id, target in moving_joints.items():
                current_position = p.getJointState(self.robot, joint_id)[0]
                if abs(current_position - target) < tolerance:
                    joints_to_remove.append(joint_id)  # Mark joint as reached

            for joint_id in joints_to_remove:
                moving_joints.pop(joint_id)
            time.sleep(0.001)
            p.stepSimulation() 
     
        for name in targets:
            current_pos = targets[name]
            joint_id, lower_lim, upper_lim, _ = self.joints[name]
            self.joints[name] = (joint_id, lower_lim, upper_lim, current_pos)

    def step(self): # FIXME
        while True:
            self.setJoints({})
            p.stepSimulation()

robotEnv = RobotEnv()
robotEnv.step()
