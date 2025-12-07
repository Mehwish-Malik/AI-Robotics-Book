---
sidebar_position: 3
---

# Quick Start Guide

Get up and running with humanoid robotics concepts quickly with this hands-on introduction. This guide will walk you through your first robot simulation and control example.

## Prerequisites

Ensure you have completed the installation guide before proceeding. You should have:
- Python 3.8+ installed
- Required Python libraries installed
- Virtual environment activated (recommended)

## Your First Robot Simulation

### 1. Create a Project Directory

```bash
mkdir humanoid_robotics_tutorial
cd humanoid_robotics_tutorial
```

### 2. Create a Basic Robot Controller

Create a file called `simple_robot.py`:

```python
# simple_robot.py
import pybullet as p
import pybullet_data
import time
import numpy as np

class SimpleRobotController:
    def __init__(self):
        # Connect to PyBullet physics server
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1])

        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"Number of joints: {self.num_joints}")

    def move_to_position(self, joint_positions, max_time_step=0.01):
        """Move robot to specified joint positions"""
        for i, pos in enumerate(joint_positions):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                force=500
            )

        p.stepSimulation()
        time.sleep(max_time_step)

    def run_demo(self):
        """Run a simple movement demonstration"""
        print("Starting robot movement demo...")

        # Move to initial position
        initial_pos = [0.0] * self.num_joints
        self.move_to_position(initial_pos)

        # Move to different positions in a sequence
        positions = [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]

        for pos in positions:
            self.move_to_position(pos)
            time.sleep(1)

        print("Demo completed!")

    def disconnect(self):
        """Disconnect from physics server"""
        p.disconnect()

def main():
    controller = SimpleRobotController()

    try:
        controller.run_demo()
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    finally:
        controller.disconnect()

if __name__ == "__main__":
    main()
```

### 3. Run the Simulation

Execute your first robot simulation:

```bash
python simple_robot.py
```

You should see a simple robot (R2D2 model) in a physics simulation that moves through different joint configurations.

## Understanding the Code

Let's break down the key components:

### Robot Initialization
```python
# Connect to physics engine
self.physics_client = p.connect(p.GUI)
```
This establishes a connection to the PyBullet physics engine with a graphical interface.

### Joint Control
```python
p.setJointMotorControl2(
    bodyIndex=self.robot_id,
    jointIndex=i,
    controlMode=p.POSITION_CONTROL,
    targetPosition=pos,
    force=500
)
```
This sets each joint to move to a specific position with a controlled force.

### Simulation Loop
```python
p.stepSimulation()
time.sleep(max_time_step)
```
This advances the physics simulation by one time step and adds a delay for visualization.

## Advanced Example: Walking Pattern

Create a more complex example in `walking_robot.py`:

```python
# walking_robot.py
import pybullet as p
import pybullet_data
import time
import numpy as np

class WalkingRobot:
    def __init__(self):
        # Connect to PyBullet physics server
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Load plane and a simple humanoid model
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1])

        # Walking parameters
        self.step_size = 0.1
        self.step_height = 0.05
        self.speed = 0.1

    def simple_walk(self, steps=10):
        """Simple walking pattern"""
        for step in range(steps):
            # Move forward in a simple pattern
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)

            # Move forward
            new_pos = [base_pos[0] + self.speed * 0.1, base_pos[1], base_pos[2]]
            p.resetBasePositionAndOrientation(self.robot_id, new_pos, base_orn)

            p.stepSimulation()
            time.sleep(0.01)

    def run(self):
        """Run the walking demonstration"""
        print("Starting walking demonstration...")
        self.simple_walk(steps=50)
        print("Walking demo completed!")

    def disconnect(self):
        """Disconnect from physics server"""
        p.disconnect()

def main():
    walker = WalkingRobot()

    try:
        walker.run()
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    finally:
        walker.disconnect()

if __name__ == "__main__":
    main()
```

## Key Concepts Introduced

### 1. Forward Kinematics
Understanding how joint angles affect the position of robot parts.

### 2. Inverse Kinematics
Calculating required joint angles to achieve desired end-effector positions.

### 3. Control Theory
Using feedback to control robot movements and maintain stability.

### 4. Simulation Physics
Working with gravity, friction, and collision detection in a virtual environment.

## Troubleshooting

### Issue: Robot doesn't move
**Solution**: Check that joint indices are correct and the robot model has movable joints.

### Issue: Simulation runs too fast/slow
**Solution**: Adjust the `time.sleep()` values to control the simulation speed.

### Issue: Robot falls through the ground
**Solution**: Ensure gravity is set correctly and the robot model is properly loaded.

## Next Steps

This quick start guide introduced you to:
- Basic robot simulation setup
- Joint control mechanisms
- Simple movement patterns
- Physics simulation concepts

In the next module, we'll explore the fundamental principles of humanoid robotics in greater depth, covering the historical development and core concepts that define this fascinating field.

Continue to the next chapter to learn about the history and evolution of humanoid robotics.