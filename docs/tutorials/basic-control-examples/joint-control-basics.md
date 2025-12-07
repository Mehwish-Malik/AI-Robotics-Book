---
sidebar_position: 2
---

# Joint Control Basics

## Overview

Joint control is fundamental to humanoid robotics, enabling precise movement and positioning of robot limbs. This tutorial covers the essential concepts of joint control including position, velocity, and torque control methods.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand different joint control modes
- Implement basic PID control for joint positioning
- Apply torque control for compliant behavior
- Design safety limits for joint control

## Joint Control Modes

### Position Control

Position control is the most common control mode where the controller commands a specific joint angle:

```python
import numpy as np

def position_control(current_pos, desired_pos, kp=100.0, max_torque=50.0):
    """
    Simple position control using proportional control
    """
    error = desired_pos - current_pos
    torque = kp * error
    return np.clip(torque, -max_torque, max_torque)

# Example usage
current_angle = 0.5  # radians
target_angle = 1.0   # radians
torque_command = position_control(current_angle, target_angle)
print(f"Torque command: {torque_command:.2f} Nm")
```

### Velocity Control

Velocity control commands a specific joint velocity:

```python
def velocity_control(current_vel, desired_vel, kp=10.0, max_torque=50.0):
    """
    Simple velocity control
    """
    error = desired_vel - current_vel
    torque = kp * error
    return np.clip(torque, -max_torque, max_torque)
```

### Torque Control

Torque control directly commands the output torque:

```python
def torque_control(desired_torque, max_torque=50.0):
    """
    Direct torque control with limits
    """
    return np.clip(desired_torque, -max_torque, max_torque)
```

## PID Control Implementation

PID (Proportional-Integral-Derivative) control provides more sophisticated control:

```python
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.previous_error = 0
        self.integral = 0
        self.previous_time = None

    def update(self, setpoint, measurement, dt=None):
        current_time = time.time()

        if dt is None and self.previous_time is not None:
            dt = current_time - self.previous_time
        elif dt is None:
            dt = 0.001  # Default time step

        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Store values for next iteration
        self.previous_error = error
        self.previous_time = current_time

        return output

# Example: Controlling a joint to reach a target position
pid = PIDController(kp=200, ki=50, kd=10)

def control_joint_to_position(joint_id, target_pos, current_pos, dt):
    torque = pid.update(target_pos, current_pos, dt)
    # Apply torque to joint (implementation depends on your robot interface)
    return torque
```

## Safety and Limits

Implementing safety limits is crucial for safe operation:

```python
class JointSafetyLimits:
    def __init__(self, min_pos, max_pos, max_vel, max_torque):
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.max_vel = max_vel
        self.max_torque = max_torque

    def check_safety(self, pos, vel, torque):
        """
        Check if joint state is within safe limits
        """
        if pos < self.min_pos or pos > self.max_pos:
            return False, "Position limit exceeded"

        if abs(vel) > self.max_vel:
            return False, "Velocity limit exceeded"

        if abs(torque) > self.max_torque:
            return False, "Torque limit exceeded"

        return True, "Safe"

# Example usage
safety_limits = JointSafetyLimits(
    min_pos=-np.pi,
    max_pos=np.pi,
    max_vel=5.0,  # rad/s
    max_torque=100.0  # Nm
)

def safe_joint_control(joint_pos, joint_vel, target_pos):
    # Calculate control torque
    torque = position_control(joint_pos, target_pos)

    # Check safety limits
    is_safe, message = safety_limits.check_safety(joint_pos, joint_vel, torque)

    if not is_safe:
        print(f"SAFETY ISSUE: {message}")
        return 0.0  # Return zero torque for safety

    return torque
```

## Multi-Joint Coordination

Controlling multiple joints requires coordination:

```python
class MultiJointController:
    def __init__(self, num_joints):
        self.num_joints = num_joints
        self.pid_controllers = [PIDController(100, 10, 5) for _ in range(num_joints)]
        self.safety_limits = [JointSafetyLimits(-np.pi, np.pi, 5.0, 100.0) for _ in range(num_joints)]

    def control_joints(self, current_positions, target_positions, dt=0.001):
        """
        Control multiple joints simultaneously
        """
        torques = []

        for i in range(self.num_joints):
            # Calculate torque for this joint
            torque = self.pid_controllers[i].update(target_positions[i], current_positions[i], dt)

            # Check safety limits
            is_safe, _ = self.safety_limits[i].check_safety(
                current_positions[i],
                0,  # Simplified velocity check
                torque
            )

            if not is_safe:
                torque = 0  # Emergency stop for this joint

            torques.append(torque)

        return np.array(torques)

# Example: Controlling a 6-DOF arm
arm_controller = MultiJointController(6)
current_arm_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
target_arm_pos = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

arm_torques = arm_controller.control_joints(current_arm_pos, target_arm_pos)
print(f"Arm joint torques: {arm_torques}")
```

## Practical Example: Joint Trajectory Following

Implementing smooth trajectory following:

```python
def generate_minimal_jerk_trajectory(start_pos, end_pos, duration, dt=0.001):
    """
    Generate a minimal jerk trajectory between two positions
    """
    t_total = duration
    t = np.arange(0, t_total, dt)

    # Minimal jerk trajectory formula
    trajectory = []
    for ti in t:
        if ti >= t_total:
            trajectory.append(end_pos)
        else:
            ratio = ti / t_total
            s = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
            pos = start_pos + (end_pos - start_pos) * s
            trajectory.append(pos)

    return np.array(trajectory)

# Example: Smooth joint movement
start_angle = 0.0
end_angle = np.pi/2  # 90 degrees
movement_time = 2.0  # seconds

trajectory = generate_minimal_jerk_trajectory(start_angle, end_angle, movement_time)

# Follow the trajectory
current_pos = start_angle
dt = 0.001
pid = PIDController(kp=150, ki=25, kd=8)

for i, desired_pos in enumerate(trajectory):
    torque = pid.update(desired_pos, current_pos, dt)
    # In a real system, you would apply this torque to the joint
    # and measure the new position
    print(f"Time: {i*dt:.3f}s, Desired: {desired_pos:.3f}, Torque: {torque:.3f}")

    # Update current position (simplified model)
    current_pos += torque * dt * 0.01  # Simplified dynamics
```

## Best Practices

1. **Always implement safety limits** to prevent damage to the robot or environment
2. **Use appropriate control gains** - too high can cause instability, too low can cause poor performance
3. **Consider the robot's dynamics** when designing controllers
4. **Test thoroughly** in simulation before real robot deployment
5. **Implement graceful degradation** when safety limits are reached

## Exercises

1. Implement a joint controller that smoothly moves from one position to another using a trapezoidal velocity profile.
2. Add velocity and acceleration feedback to improve the PID controller performance.
3. Create a safety system that monitors joint temperatures and reduces control gains when overheating is detected.