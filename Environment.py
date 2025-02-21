import numpy as np
import pybullet as p
import pybullet_data
import time, math, os

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -3)
p.setRealTimeSimulation(0)

terrain_collision = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[1.5, 1.5, 5],
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
p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=focus_position)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) # Toggle render

p.setTimeStep(1/120)  # Reduce update frequency for efficiency
p.setPhysicsEngineParameter(fixedTimeStep=1/120, numSolverIterations=10)  # Lighter computation

while True:
    p.stepSimulation()
    time.sleep(0.01)
