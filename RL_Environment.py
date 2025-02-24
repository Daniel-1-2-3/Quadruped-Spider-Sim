import numpy as np
import pybullet as p
import pybullet_data
import time, math, os
from Simulation import Simulation


"""
    Environment setup with physics
"""
p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

terrain_collision = p.createCollisionShape( # Randomly generate heightfield
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[0.1, 0.1, 0.1],
    heightfieldTextureScaling=128,
    heightfieldData=np.random.uniform(-0.1, 0.1, 128 * 128).astype(np.float32),
    numHeightfieldRows=128,
    numHeightfieldColumns=128
)
terrain_body = p.createMultiBody(0, terrain_collision)
p.resetBasePositionAndOrientation(terrain_body, [0, 0, 0], [0, 0, 0, 1])
p.changeVisualShape(terrain_body, -1, textureUniqueId=-1, rgbaColor=[0.85, 0.85, 0.85, 1]) # Set color of terrain
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows
p.changeDynamics(terrain_body, -1, contactStiffness=5e4, contactDamping=5e4, lateralFriction=0.8, spinningFriction=0.6, rollingFriction=0.1, restitution=0) # Set friction, no bounce, no sinking (contactStiffness)

targid = p.loadURDF(f'{os.getcwd()}/Model/robot.urdf', [0, 0, 3], useFixedBase=False)

p.changeDynamics(targid, -1, restitution=0, linearDamping=0.3, angularDamping=0.3)
focus_position, _ = p.getBasePositionAndOrientation(targid)
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=focus_position)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

"""
    Robot joints param debug setup
"""
joint_count = p.getNumJoints(targid)
print("Joints count:", joint_count)
# Create sliders for joint control
joint_sliders = {}
for i in range(joint_count):
    info = p.getJointInfo(targid, i)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    lower_limit, upper_limit = info[8], info[9]

    print(f"Joint {i}: {joint_name}, Type: {joint_type}, Limits: ({lower_limit}, {upper_limit})")
    joint_sliders[i] = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0) # Create slider

""" 
    Step the simulation
"""
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
lastPrint = 0
while True:
    # Read joint sliders and update joint positions
    targets = {}
    for joint_id in range(joint_count):
        target_position = p.readUserDebugParameter(joint_sliders[joint_id])

        p.setJointMotorControl2(
            bodyUniqueId=targid,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=4.57,
            force=1000.0  
        )

        targets[joint_id] = target_position

    # Print robot joint positions every 0.05 seconds
    if time.time() - lastPrint > 0.05:
        lastPrint = time.time()
        os.system("clear")
        print("Joint Angles:")
        for joint_id, angle in targets.items():
            print(f"  Joint {joint_id}: {angle:.3f}")

    p.stepSimulation()
    time.sleep(0.01)