import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load ground
plane = p.loadURDF("plane.urdf")

# Define robot base
base_start_pos = [0, 0, 0.3]
base_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("r2d2.urdf", base_start_pos, base_start_orientation)  # Temporary base, replace with custom model

# Create leg segments
legs = []
for i in range(4):
    x_offset = 0.2 if i % 2 == 0 else -0.2  # Left vs right legs
    y_offset = -0.3 if i < 2 else 0.3  # Front vs back legs
    leg = p.createMultiBody(0.05, p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.2),
                            p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.2),
                            base_start_pos[0] + x_offset, base_start_pos[1] + y_offset, base_start_pos[2] - 0.1)
    legs.append(leg)

# Simple control loop
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
