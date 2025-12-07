---
sidebar_position: 3
---

# Chapter 3: Basic Concepts and Terminology

## Summary

This chapter establishes the fundamental concepts and terminology essential for understanding humanoid robotics. We'll explore coordinate systems, degrees of freedom, stability concepts, and mathematical representations that form the foundation for more advanced topics. Understanding these concepts is crucial for grasping the complexities of humanoid robot design and control.

## Learning Outcomes

By the end of this chapter, you will be able to:
- Explain different coordinate systems used in humanoid robotics
- Calculate and analyze degrees of freedom in robotic systems
- Understand static and dynamic stability concepts
- Apply mathematical representations for robot kinematics
- Identify key terminology used throughout the field

## Key Concepts

- **Coordinate Systems**: Reference frames for describing robot position and orientation
- **Degrees of Freedom (DOF)**: Independent movements a mechanical system can perform
- **Kinematic Chains**: Series of rigid bodies connected by joints
- **Static Stability**: Stability without motion
- **Dynamic Stability**: Stability maintained through active control during motion
- **Center of Mass**: Point where the total mass of the system can be considered concentrated

## Coordinate Systems in Humanoid Robotics

### World Coordinate System

The world coordinate system is a fixed reference frame that defines the global environment in which the robot operates. It typically uses a right-handed coordinate system:

- **X-axis**: Points forward (in the direction of intended movement)
- **Y-axis**: Points to the left
- **Z-axis**: Points upward (opposing gravity)

This system is essential for navigation, mapping, and understanding the robot's position relative to its environment.

### Body Coordinate System

The body coordinate system is attached to the robot's torso or base link. It moves with the robot and provides a reference for the robot's own perspective:

- **Origin**: Usually located at the robot's center of mass or at the pelvis
- **Orientation**: Typically aligned with the robot's forward direction
- **Purpose**: Simplifies control of limbs relative to the body

### Joint Coordinate Systems

Each joint has its own coordinate system that defines the range of motion for that specific joint:

- **Rotation Axes**: Define the directions in which the joint can rotate
- **Translation Axes**: Define the directions in which the joint can translate (for prismatic joints)
- **Joint Limits**: Define the mechanical constraints on movement

### End-Effector Coordinate Systems

End-effectors (hands, feet) have coordinate systems that define their orientation and position:

- **Hand Frame**: Defines the orientation of the hand for grasping
- **Foot Frame**: Defines the orientation of the foot for walking
- **Tool Frame**: Defines the orientation of any attached tools

## Degrees of Freedom and Mobility

### Definition of Degrees of Freedom

Degrees of freedom (DOF) represent the number of independent parameters that define the configuration of a mechanical system. For a rigid body in 3D space, there are 6 DOF: 3 for translation and 3 for rotation.

### Calculating DOF for Humanoid Systems

The mobility of a robotic system can be calculated using Gruebler's equation:

```
DOF = λ(N - 1) - Σ(λ - f_i)
```

Where:
- λ = degrees of freedom of the space (6 for spatial mechanisms)
- N = number of links including ground
- f_i = degrees of freedom of joint i

For a simple serial chain:
```
DOF = Σ(f_i) - constraints
```

### DOF in Humanoid Robots

A typical humanoid robot has multiple kinematic chains:

**Right Arm Chain** (7 DOF):
- Shoulder: 3 DOF (pitch, yaw, roll)
- Elbow: 1 DOF (flexion/extension)
- Wrist: 3 DOF (pitch, yaw, roll)

**Right Leg Chain** (6 DOF):
- Hip: 3 DOF (flexion/extension, abduction/adduction, internal/external rotation)
- Knee: 1 DOF (flexion/extension)
- Ankle: 2 DOF (dorsiflexion/plantarflexion, inversion/eversion)

### Redundant DOF

Humanoid robots often have redundant DOF, meaning they have more DOF than necessary to achieve a task. This redundancy provides:

- **Flexibility**: Multiple ways to achieve the same goal
- **Obstacle Avoidance**: Ability to navigate around obstacles
- **Optimization**: Capability to optimize secondary objectives (energy, comfort)

## Static vs. Dynamic Stability

### Static Stability

Static stability refers to the robot's ability to maintain balance without motion. For static balance, the center of mass (CoM) must be positioned within the support polygon (the convex hull of all contact points with the ground).

**Key Principles**:
- Center of mass must be within the support base
- Larger support base provides greater stability
- Lower center of mass improves stability

### Dynamic Stability

Dynamic stability involves maintaining balance during motion. Unlike static stability, the robot can be dynamically stable even when the center of mass is outside the support polygon, provided that appropriate control actions are taken.

**Key Principles**:
- Zero Moment Point (ZMP) must be within the support polygon
- Continuous adjustment of control parameters
- Use of momentum to maintain balance

### Stability Metrics

**Center of Pressure (CoP)**: The point where the ground reaction force acts.

**Zero Moment Point (ZMP)**: The point where the moment of the ground reaction force and gravity force equals zero.

**Capture Point**: The point where the robot must step to come to a complete stop.

## Center of Mass and Balance

### Center of Mass Calculation

The center of mass of a system of particles is given by:

```
CoM = Σ(m_i * r_i) / Σ(m_i)
```

Where:
- m_i = mass of particle i
- r_i = position vector of particle i

For continuous bodies:
```
CoM = ∫ r dm / ∫ dm
```

### Balance Control Strategies

**Ankle Strategy**: Small balance adjustments using ankle torques (effective for small disturbances)

**Hip Strategy**: Larger adjustments using hip torques (effective for medium disturbances)

**Stepping Strategy**: Taking a step to expand the support base (effective for large disturbances)

**Suspension Strategy**: Moving the center of mass to maintain balance (for dynamic situations)

## Kinematic Representations

### Forward Kinematics

Forward kinematics calculates the position and orientation of the end-effector given the joint angles:

```
T = f(θ₁, θ₂, ..., θₙ)
```

Where T is the transformation matrix and θᵢ are the joint angles.

### Inverse Kinematics

Inverse kinematics calculates the required joint angles to achieve a desired end-effector position and orientation:

```
θ₁, θ₂, ..., θₙ = f⁻¹(T)
```

### Jacobian Matrix

The Jacobian matrix relates joint velocities to end-effector velocities:

```
v = J(θ) * θ̇
```

Where:
- v = end-effector velocity vector
- J = Jacobian matrix
- θ̇ = joint velocity vector

## Mathematical Representations

### Homogeneous Transformation Matrices

Homogeneous transformation matrices represent both rotation and translation in a single 4×4 matrix:

```
T = [R  p]
    [0  1]
```

Where R is a 3×3 rotation matrix and p is a 3×1 position vector.

### Rotation Representations

**Rotation Matrices**: 3×3 orthogonal matrices
**Euler Angles**: Three angles representing sequential rotations
**Quaternions**: Four-parameter representation avoiding gimbal lock
**Axis-Angle**: Rotation around a specific axis

### Denavit-Hartenberg Parameters

The Denavit-Hartenberg (DH) convention provides a systematic way to define coordinate frames on robotic linkages:

- **aᵢ**: Link length
- **αᵢ**: Link twist
- **dᵢ**: Link offset
- **θᵢ**: Joint angle

## Practical Applications

### Robot Calibration

Understanding coordinate systems is essential for:
- Tool calibration
- Sensor integration
- Workspace definition
- Collision avoidance

### Control System Design

DOF analysis is crucial for:
- Controller design
- Trajectory planning
- Singularity avoidance
- Redundancy resolution

### Stability Analysis

Stability concepts are applied to:
- Gait planning
- Balance control
- Disturbance rejection
- Safe operation

## Challenges

### Computational Complexity

- High DOF systems require significant computational resources
- Real-time control of complex kinematic chains
- Optimization of redundant DOF

### Modeling Accuracy

- Precise modeling of complex mechanical systems
- Accounting for flexibility and non-linearities
- Parameter identification for control

### Environmental Adaptation

- Operating in unstructured environments
- Handling uncertainty in sensor data
- Adapting to changing conditions

## Figure List

1. **Figure 3.1**: Coordinate system definitions for humanoid robots
2. **Figure 3.2**: Degrees of freedom in human vs. robotic joints
3. **Figure 3.3**: Static vs. dynamic stability concepts
4. **Figure 3.4**: Center of mass and support polygon visualization
5. **Figure 3.5**: Kinematic chain representations

## Code Example: Kinematic Calculations

```python
import numpy as np
from typing import List, Tuple

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Calculate Denavit-Hartenberg transformation matrix

    Args:
        a: Link length
        alpha: Link twist
        d: Link offset
        theta: Joint angle

    Returns:
        4x4 homogeneous transformation matrix
    """
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    return np.array([
        [cos_th, -sin_th * cos_alpha, sin_th * sin_alpha, a * cos_th],
        [sin_th, cos_th * cos_alpha, -cos_th * sin_alpha, a * sin_th],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(dh_params: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Calculate forward kinematics using DH parameters

    Args:
        dh_params: List of (a, alpha, d, theta) tuples for each joint

    Returns:
        Final transformation matrix
    """
    T = np.eye(4)  # Start with identity matrix
    for a, alpha, d, theta in dh_params:
        T_joint = dh_transform(a, alpha, d, theta)
        T = T @ T_joint  # Matrix multiplication
    return T

def calculate_center_of_mass(masses: List[float], positions: List[np.ndarray]) -> np.ndarray:
    """
    Calculate center of mass for a multi-body system

    Args:
        masses: List of masses for each body
        positions: List of position vectors for each body

    Returns:
        Center of mass position vector
    """
    if len(masses) != len(positions):
        raise ValueError("Masses and positions must have the same length")

    total_mass = sum(masses)
    if total_mass == 0:
        return np.zeros(3)

    weighted_sum = np.zeros(3)
    for mass, pos in zip(masses, positions):
        weighted_sum += mass * pos

    return weighted_sum / total_mass

def is_statically_stable(com: np.ndarray, support_polygon: List[np.ndarray]) -> bool:
    """
    Check if a center of mass is within a 2D support polygon

    Args:
        com: Center of mass (x, y) coordinates
        support_polygon: List of (x, y) coordinates defining the support polygon

    Returns:
        True if CoM is within the support polygon
    """
    # Convert to 2D (ignore z-coordinate for stability check)
    x, y = com[0], com[1]

    # Ray casting algorithm to check if point is inside polygon
    n = len(support_polygon)
    inside = False

    p1x, p1y = support_polygon[0][0], support_polygon[0][1]
    for i in range(1, n + 1):
        p2x, p2y = support_polygon[i % n][0], support_polygon[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def calculate_zmp(com: np.ndarray, cop: np.ndarray, gravity: float = 9.81) -> np.ndarray:
    """
    Calculate Zero Moment Point given Center of Mass and Center of Pressure

    Args:
        com: Center of mass position (x, y, z)
        cop: Center of pressure position (x, y, z)
        gravity: Gravitational acceleration

    Returns:
        ZMP position (x, y, z)
    """
    # Simplified ZMP calculation for static case
    # In practice, ZMP involves dynamic calculations
    zmp_x = cop[0] - (com[2] - cop[2]) * (com[0] - cop[0]) / (com[2] - cop[2])
    zmp_y = cop[1] - (com[2] - cop[2]) * (com[1] - cop[1]) / (com[2] - cop[2])

    # For static case, ZMP is approximately equal to CoP
    return np.array([cop[0], cop[1], cop[2]])

# Example usage
if __name__ == "__main__":
    # Example: Simple 2-link arm
    dh_params = [
        (0.1, np.pi/2, 0.2, np.pi/4),   # Joint 1
        (0.3, 0, 0, np.pi/6)            # Joint 2
    ]

    final_transform = forward_kinematics(dh_params)
    print("Final transformation matrix:")
    print(final_transform)

    # Example: Center of mass calculation
    masses = [1.0, 0.5, 0.3]  # Mass of torso, upper arm, lower arm
    positions = [
        np.array([0.0, 0.0, 0.8]),    # Torso CoM
        np.array([0.1, 0.1, 0.7]),    # Upper arm CoM
        np.array([0.2, 0.15, 0.6])    # Lower arm CoM
    ]

    com = calculate_center_of_mass(masses, positions)
    print(f"\nCenter of mass: {com}")

    # Example: Stability check
    support_polygon = [
        np.array([-0.1, -0.1]),  # Foot corner 1
        np.array([0.1, -0.1]),   # Foot corner 2
        np.array([0.1, 0.1]),    # Foot corner 3
        np.array([-0.1, 0.1])    # Foot corner 4
    ]

    robot_com = np.array([0.05, 0.0, 0.0])  # Robot's CoM projected to ground
    stable = is_statically_stable(robot_com, support_polygon)
    print(f"Is robot statically stable? {stable}")

    # Example: ZMP calculation
    cop = np.array([0.0, 0.0, 0.0])  # Center of pressure
    zmp = calculate_zmp(com, cop)
    print(f"ZMP position: {zmp}")
```

## Exercises

1. Calculate the degrees of freedom for a humanoid robot with 6 DOF arms, 6 DOF legs, and 3 DOF head.

2. Determine if a robot with CoM at (0.05, 0.0, 0.0) is statically stable with a square foot of 0.2m × 0.2m centered at origin.

3. Implement forward kinematics for a 3-DOF planar manipulator and verify the results.

## Summary

This chapter established the fundamental concepts and terminology essential for understanding humanoid robotics. We explored coordinate systems, degrees of freedom, stability concepts, and mathematical representations that form the foundation for more advanced topics. These concepts are crucial for grasping the complexities of humanoid robot design, analysis, and control.