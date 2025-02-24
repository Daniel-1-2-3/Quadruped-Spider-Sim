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

controls = {}
for name in sim.getJoints():
    if name.endswith('_speed'):
        controls[name] = p.addUserDebugParameter(
            name, -math.pi*3, math.pi*3, 0)
    else:
        infos = sim.getJointsInfos(name)
        low = -math.pi
        high = math.pi
        if 'lowerLimit' in infos:
            low = infos['lowerLimit']
        if 'upperLimit' in infos:
            high = infos['upperLimit']
        controls[name] = p.addUserDebugParameter(name, low, high, 0)

lastPrint = 0
while True:
    targets = {}
    for name in controls.keys():
        targets[name] = p.readUserDebugParameter(controls[name])
    sim.setJoints(targets)

    if time.time() - lastPrint > 0.05:
        lastPrint = time.time()
        os.system("clear")
        frames = sim.getFrames()
        for frame in frames:
            print(frame)
            print("- x=%f\ty=%f\tz=%f" % frames[frame][0])
            print("- r=%f\tp=%f\ty=%f" % frames[frame][1])
            print("")
        print("Center of mass:")
        print(sim.getCenterOfMassPosition())

    sim.tick()
