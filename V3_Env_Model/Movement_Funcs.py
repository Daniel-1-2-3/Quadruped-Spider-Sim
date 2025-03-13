import os, math, random, time, sys, json, copy
import numpy as np
import pybullet as p

class Movements:
    def __init__(self):
        p.connect(p.GUI)
        self.target_reached = False
        self.target_pos = [0, 0]
        self.final_target = [0, 0]
        self.reset()
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.setPhysicsEngineParameter(enableConeFriction=1)
        
        # Loading robot
        startPos = [random.randrange(-2, 3), random.randrange(-2, 3), 1.5] # (x, y, z)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=startPos)
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(f'{os.getcwd()}/V3_Env_Model/Model/robot.urdf',
                                startPos, startOrientation,
                                flags=flags, useFixedBase=False)
        p.changeDynamics(self.robot, -1, restitution=0, linearDamping=1, angularDamping=1,
                         contactStiffness=5e10, contactDamping=1e5, lateralFriction=20, 
                         rollingFriction=5, spinningFriction=5)
        p.setPhysicsEngineParameter(fixedTimeStep=0.002, maxNumCmdPer1ms=0, contactBreakingThreshold=0.001)
        
        # Loading terrain
        self.terrain = self.createRandomHeightfield()
        # Set friction, no bounce, no sinking (contactStiffness)
        p.changeDynamics(self.terrain, -1, contactStiffness=5e10, contactDamping=1e5, 
                         lateralFriction=20, spinningFriction=5, rollingFriction=5, restitution=0.01) 
        
        # Initialize joints
        self.joint_poses = {}
        for i in range (p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_id, name, lower_lim, upper_lim, current_pos = joint_info[0], joint_info[1].decode("utf-8"), joint_info[8], joint_info[9], 0
            
            if name in "BL_J4 FR_J4 FL_J1 FR_J1":
                current_pos = upper_lim
            elif name in "BL_J2 FR_J2":
                current_pos = np.radians(3)
            elif name in "BR_J2 FL_J2":
                current_pos = np.radians(-3)
            elif name in "BR_J4 FL_J4 BL_J1 BR_J1":
                current_pos = lower_lim
          
            self.joint_poses[name] = {"id": joint_id, "pos": current_pos, "lLim": lower_lim, "uLim": upper_lim}
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, self.joint_poses[name]["pos"], maxVelocity=10, force=1e6)

        for i in range(250): # Wait for robot to touch the ground
            p.stepSimulation()
    
    def applyDefinedGait(self, correction=0):
        correction *= -0.5
        def get_joint_info(name):
            midpoint = (self.joint_poses[name]["lLim"] + self.joint_poses[name]["uLim"])/2
            lower = self.joint_poses[name]["lLim"]
            upper = self.joint_poses[name]["uLim"]
            return midpoint, lower, upper
        
        def FL_J1_map():
            m, f, b = get_joint_info("FL_J1")
            if correction < 0:
                f += float(np.radians(max(abs(correction), 5)))
            return [f, f, b, b] 
        def FR_J1_map():
            m, b, f = get_joint_info("FR_J1")
            if correction > 0:
                f -= float(np.radians(max(abs(correction), 5)))
            return [b, b, f, f]  
        def BL_J1_map():
            m, f, b = get_joint_info("BL_J1")
            if correction < 0:
                f += float(np.radians(max(abs(correction), 5)))
            return [b, b, f, f]  
        def BR_J1_map():
            m, b, f = get_joint_info("BR_J1")
            if correction > 0:
                f -= float(np.radians(max(abs(correction), 5)))
            return [f, f, b, b] 
        def FL_J3_map():
            d, _, u = get_joint_info("FL_J3")
            return [u, d, d, d] 
        def FR_J3_map():
            d, u, _ = get_joint_info("FR_J3")
            return [d, d, u, d] 
        def BL_J3_map():
            d, u, _ = get_joint_info("BL_J3")
            return [d, d, u, d] 
        def BR_J3_map():
            d, _, u = get_joint_info("BR_J3")
            return [u, d, d, d] 

        all_actions = {
            "FL_J1": FL_J1_map(),  # f, f, b, b
            "FL_J3": FL_J3_map(),  # u, d, d, d
            "BR_J1": BR_J1_map(),  # f, f, b, b
            "BR_J3": BR_J3_map(),  # u, d, d, d
            "FR_J1": FR_J1_map(),  # b, b, f, f
            "FR_J3": FR_J3_map(),  # d, d, u, d
            "BL_J1": BL_J1_map(),  # b, b, f, f
            "BL_J3": BL_J3_map(),  # d, d, u, d
        }
        
        for i in range(len(all_actions["FL_J1"])):
            taking = {}
            needGroundedLegs = []
            for key, val in all_actions.items():
                taking[key] = val[i]
                if "J3" in key and val[i] == 0 and val[i-1] == 0:
                    needGroundedLegs.append(key)
            self.setJoints(taking, needGroundedLegs)
    
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
        deg = degree * (1 / 0.756151)

        def FL_J1_map():
            c = initial_poses["FL_J1"]
            t = c - float(np.radians(deg))
            m, f, b = get_joint_info("FL_J1")
            return [m, t, t, t, t, t, m]  # m, m, m, m, m

        def FL_J3_map():
            d, _, u = get_joint_info("FL_J3")
            return [u, u, d, d, d, d, d]  # u, d, d, d, d

        def BR_J1_map():
            c = initial_poses["BR_J1"]
            t = c - float(np.radians(deg))
            m, _, _ = get_joint_info("BR_J1")
            return [m, t, t, t, t, t, m]  # c, m, m, m, m

        def BR_J3_map():
            d, _, u = get_joint_info("BR_J3")
            return [u, u, d, d, d, d, d]  # d, u, d, d, d

        def FR_J1_map():
            c = initial_poses["FR_J1"]
            t = c - float(np.radians(deg))
            m, _, _ = get_joint_info("FR_J1")
            return [m, m, m, m, t, t, m]  # c, c, m, m, m

        def FR_J3_map():
            d, u, _ = get_joint_info("FR_J3")
            return [d, d, d, u, u, d, d]  # d, d, u, d, d

        def BL_J1_map():
            c = initial_poses["BL_J1"]
            t = c - float(np.radians(deg))
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
            needGroundedLegs = []
            for key, val in all_actions.items():
                taking[key] = val[i]
                if "J3" in key and val[i] == 0 and val[i-1] == 0:
                    needGroundedLegs.append(key)
            self.setJoints(taking, needGroundedLegs)
    
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
            needGroundedLegs = []
            for key, val in all_actions.items():
                taking[key] = val[i]
                if "J3" in key and val[i] == 0 and val[i-1] == 0:
                    needGroundedLegs.append(key)
            self.setJoints(taking, needGroundedLegs)
    
    def groundLegs(self, needGroundedLegs):
        # Get leg tip ids
        FL_tip_id = next((i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[12].decode("utf-8") == "legjointconnector4_4"), -1)
        FR_tip_id = next((i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[12].decode("utf-8") == "legjointconnector4_2"), -1)
        BL_tip_id = next((i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[12].decode("utf-8") == "legjointconnector4_3"), -1)
        BR_tip_id = next((i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[12].decode("utf-8") == "legjointconnector4"), -1)

        for i in range(25):
            leg_states = [True]  # Assume all legs are grounded until checked
            updated_joint_values = {}  # Store what the values WOULD be

            if not self.isGrounded(FL_tip_id) and "FL_J3" in needGroundedLegs:
                new_val = min(self.joint_poses["FL_J2"]["uLim"],
                            max(self.joint_poses["FL_J2"]["lLim"], self.joint_poses["FL_J2"]["pos"] - 0.01))
                p.setJointMotorControl2(self.robot, self.joint_poses["FL_J2"]["id"], p.POSITION_CONTROL, new_val, maxVelocity=1.25, force=1e8)
                leg_states.append(self.isGrounded(FL_tip_id))
                
            if not self.isGrounded(FR_tip_id) and "FR_J3" in needGroundedLegs:
                new_val = min(self.joint_poses["FR_J2"]["uLim"],
                            max(self.joint_poses["FR_J2"]["lLim"], self.joint_poses["FR_J2"]["pos"] + 0.01))
                p.setJointMotorControl2(self.robot, self.joint_poses["FR_J2"]["id"], p.POSITION_CONTROL, new_val, maxVelocity=1.25, force=1e8)
                leg_states.append(self.isGrounded(FR_tip_id))

            if not self.isGrounded(BR_tip_id) and "BR_J3" in needGroundedLegs:
                new_val = min(self.joint_poses["BR_J2"]["uLim"],
                            max(self.joint_poses["BR_J2"]["lLim"], self.joint_poses["BR_J2"]["pos"] - 0.01))
                p.setJointMotorControl2(self.robot, self.joint_poses["BR_J2"]["id"], p.POSITION_CONTROL, new_val, maxVelocity=1.25, force=1e8)
                leg_states.append(self.isGrounded(BR_tip_id))

            if not self.isGrounded(BL_tip_id) and "BL_J3" in needGroundedLegs:
                new_val = min(self.joint_poses["BL_J2"]["uLim"],
                            max(self.joint_poses["BL_J2"]["lLim"], self.joint_poses["BL_J2"]["pos"] + 0.01))
                p.setJointMotorControl2(self.robot, self.joint_poses["BL_J2"]["id"], p.POSITION_CONTROL, new_val, maxVelocity=1.25, force=1e8)
                leg_states.append(self.isGrounded(BL_tip_id))
            
            p.stepSimulation()

            if all(leg_states):
                break

    def setJoints(self, predefined_actions, needGroundedLegs, tolerance=0.01):
        moving_joints = {} # Track joints still in motion
        target_rots = {}
        
        for name, info in predefined_actions.items():
            if name not in {"FL_J4","FR_J4", "BL_J4", "BR_J4"}:
                target_rots[name] = predefined_actions[name]
            else:
                target_rots[name] = info[name]["pos"]
           
            moving_joints[self.joint_poses[name]["id"]] = target_rots[name]
            p.setJointMotorControl2(self.robot, self.joint_poses[name]["id"], p.POSITION_CONTROL, target_rots[name], maxVelocity=1.25, force=1e8)

        # Run simulation steps until all joints reach targets
        for i in range(200):
            self.groundLegs(needGroundedLegs)
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
            if controller.getDistanceTarget() < 1:
                self.target_reached = True
                break
        for name in target_rots:
            self.joint_poses[name]["pos"] = target_rots[name]
    
    def createRandomHeightfield(self):
        """Creates a random heightfield to replace the flat plane."""
        size = 128
        stone_size = 1
        height_range = 0 # Change value here

        heightfield_data = np.zeros(size * size, dtype=np.float32)
        for i in range(0, size, stone_size):
            for j in range(0, size, stone_size):
                height = np.random.uniform(-height_range, height_range)
                for x in range(stone_size):
                    for y in range(stone_size):
                        if (i + x) < size and (j + y) < size:
                            heightfield_data[(i + x) * size + (j + y)] = height
        # heightfield_data = np.zeros(128 * 128, dtype=np.float32)  # Completely flat surface
        terrain_collision = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[1.2, 1.2, 2],  # Adjust scale for realistic terrain
            heightfieldTextureScaling=1024,
            heightfieldData=heightfield_data,
            numHeightfieldRows=128,
            numHeightfieldColumns=128
        )
        terrain_body = p.createMultiBody(0, terrain_collision)
        p.resetBasePositionAndOrientation(terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.changeVisualShape(terrain_body, -1, textureUniqueId=-1, rgbaColor=[0.85, 0.85, 0.85, 1])  # Set color

        return terrain_body

    def isGrounded(self, link_id):
        return len(p.getContactPoints(bodyA=self.robot, bodyB=self.terrain, linkIndexA=link_id)) > 0
    
    def getHeadingDeg(self):
        robot_yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1])[2]
        return np.degrees(robot_yaw) + 180
    
    def getRobotPos(self):
        return p.getBasePositionAndOrientation(self.robot)[0][:2]
    
    def getDistanceTarget(self):
        current_pos = self.getRobotPos()
        return np.linalg.norm(np.array(current_pos) - np.array(self.target_pos))
    
    def getDeviationFromTarget(self):
        current_pos = self.getRobotPos()
        desired_angle = float(np.degrees(np.arctan2(
            self.target_pos[1] - current_pos[1], self.target_pos[0] - current_pos[0])))
        heading_angle = self.getHeadingDeg()
        a = desired_angle - heading_angle
        b = 360 - abs(a)
        angle_deviation = a if abs(a) < abs(b) else b
        return angle_deviation
    
    def autoAlignTarget(self):
        deviation = - self.getDeviationFromTarget() 
        self.applyDefinedReset()
        for i in range(math.floor(abs(deviation) / 18)):
            self.applyDefinedTurn(18 * (deviation / abs(deviation)))
        self.applyDefinedTurn((deviation % 18))
        
        # Fine adjustment 
        if self.getDistanceTarget() < 2:
            while abs(self.getDeviationFromTarget()) > 5:
                self.applyDefinedTurn(5 * (deviation / abs(deviation)))
                
    def draw_targets(self, targets):
        for target in targets:
            p.addUserDebugLine(
                lineFromXYZ=[target[0], target[1], 0], 
                lineToXYZ=[target[0], target[1], 1], 
                lineColorRGB=[1, 0, 0],
                lineWidth=2
            )
        
    def toTarget(self, targets=[[0, 0], [1, 10], [20, -10]]):
        self.draw_targets(targets)
        self.final_target = targets[-1]
        
        self.target_pos = targets[0]
        self.autoAlignTarget()
        for target in targets:
            self.target_reached = False
            self.target_pos = target
            print(self.target_pos)

            if self.getDeviationFromTarget() > 40:
                self.autoAlignTarget()
            while True:
                deviation = self.getDeviationFromTarget()
                if abs(deviation) > 45 and controller.getDistanceTarget() < 4:
                    self.autoAlignTarget()
                self.applyDefinedGait(deviation)
                if self.target_reached:
                    break
        print('TARGET REACHED')
        time.sleep(3)
            
if __name__ == "__main__":
    controller = Movements()
    controller.toTarget([[-4, 5], [5, 10]])
        