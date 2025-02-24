import math
import sys
import os
import time
import argparse
import pybullet as p
from Setup_Simulation import Simulation

sim = Simulation(f'{os.getcwd()}/Model/robot.urdf', gui=True, panels=True, fixed=False, ignore_self_collisions=False)
pos, rpy = sim.getRobotPose()
_, orn = p.getBasePositionAndOrientation(sim.robot)
sim.setRobotPose([pos[0], pos[1], pos[2]], orn)

defined_joints = [
    "FR_J1", "BL_J1", "FL_J1", "BR_J1",
    "FR_J2", "BL_J2", "FL_J2", "BR_J2", 
    "FR_J3", "BL_J3", "FL_J3", "BR_J3",
    "FR_J4", "BL_J4", "FL_J4", "BR_J4"
]

controls = {}
targets = {}
for joint_name in defined_joints:
    controls[joint_name] = None  

for name in controls.keys():
    if name.endswith('_speed'):
        controls[name] = p.addUserDebugParameter(name, -math.pi*3, math.pi*3, 0)
    else:
        infos = sim.getJointsInfos(name)
        low, high = -math.pi, math.pi
        if 'lowerLimit' in infos:
            low = infos['lowerLimit']
        if 'upperLimit' in infos:
            high = infos['upperLimit']
        controls[name] = p.addUserDebugParameter(name, low, high, 0)
        # Manually set the initial angle
        if name in "BL_J1 FR_J1 BR_J2 FL_J2 BL_J4 FR_J4":
            targets[name] = high
            controls[name] = p.addUserDebugParameter(name, low, high, high)
        elif name in "BR_J1 FL_J1 BL_J2 FR_J2 BR_J4 FL_J4":
            targets[name] = low
            controls[name] = p.addUserDebugParameter(name, low, high, low)
        
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
print(targets)
lastPrint = 0
sim.setJoints(targets)

while True:
    targets = {}
    for name in controls.keys():
        targets[name] = p.readUserDebugParameter(controls[name])
    sim.setJoints(targets)

    if time.time() - lastPrint > 0.05:
        lastPrint = time.time()
        frames = sim.getFrames()
        #print(sim.getCenterOfMassPosition())

    sim.tick()
