import pybullet as p
import pybullet_data
import numpy as np
import time
import random

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Create Ground Heightfield
terrain_size = 256
height_data = np.random.uniform(-1, 1, (terrain_size, terrain_size)) * 2  # Random uneven terrain

# Normalize and scale heightfield
height_data = (height_data - np.min(height_data)) / (np.max(height_data) - np.min(height_data)) * 255
height_data = height_data.astype(np.int16)

terrain_shape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[0.1, 0.1, 3],  # Adjust scale to fit scene
    heightfieldData=height_data.flatten(),
    numHeightfieldRows=terrain_size,
    numHeightfieldColumns=terrain_size
)

terrain_body = p.createMultiBody(0, terrain_shape)

# Function to add random-sized rocks
def add_random_rocks(num_rocks):
    for _ in range(num_rocks):
        rock_size = random.uniform(0.3, 1.5)  # Random rock size
        rock_x = random.uniform(-10, 10)  # Random x position
        rock_y = random.uniform(-10, 10)  # Random y position
        rock_z = 2 + random.uniform(0, 2)  # Start above ground to avoid clipping

        rock_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=rock_size)
        rock_visual = p.createVisualShape(p.GEOM_SPHERE, radius=rock_size)

        p.createMultiBody(baseMass=random.uniform(5, 20),  # Random mass
                          baseCollisionShapeIndex=rock_shape,
                          baseVisualShapeIndex=rock_visual,
                          basePosition=[rock_x, rock_y, rock_z])

# Add 10 random rocks
add_random_rocks(10)

# Run Simulation
while True:
    p.stepSimulation()
    time.sleep(1/240)
