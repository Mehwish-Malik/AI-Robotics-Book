---
sidebar_position: 3
---

# Balance Control

## Overview

Balance control is critical for humanoid robots to maintain stability during static and dynamic activities. This tutorial covers fundamental balance control concepts including center of mass control, zero moment point (ZMP) control, and reactive balance strategies.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand the principles of static and dynamic balance
- Implement ZMP-based balance control
- Design reactive balance strategies for perturbations
- Apply whole-body control for balance maintenance

## Balance Fundamentals

### Center of Mass (CoM) and Support Polygon

For a humanoid robot to maintain balance, its center of mass must remain within the support polygon (the convex hull of all contact points with the ground).

```python
import numpy as np
from scipy.spatial import ConvexHull

def calculate_support_polygon(left_foot_pos, right_foot_pos):
    """
    Calculate the support polygon for bipedal stance
    """
    # Create points for both feet (simplified as rectangles)
    left_points = [
        [left_foot_pos[0] + 0.1, left_foot_pos[1] + 0.05],  # front-left
        [left_foot_pos[0] + 0.1, left_foot_pos[1] - 0.05],  # front-right
        [left_foot_pos[0] - 0.1, left_foot_pos[1] - 0.05],  # back-right
        [left_foot_pos[0] - 0.1, left_foot_pos[1] + 0.05],  # back-left
    ]

    right_points = [
        [right_foot_pos[0] + 0.1, right_foot_pos[1] + 0.05],
        [right_foot_pos[0] + 0.1, right_foot_pos[1] - 0.05],
        [right_foot_pos[0] - 0.1, right_foot_pos[1] - 0.05],
        [right_foot_pos[0] - 0.1, right_foot_pos[1] + 0.05],
    ]

    all_points = left_points + right_points
    hull = ConvexHull(all_points)
    return np.array(all_points)[hull.vertices]

def is_balanced(com_pos, support_polygon):
    """
    Check if center of mass is within support polygon
    """
    from matplotlib.path import Path
    path = Path(support_polygon)
    return path.contains_point(com_pos[:2])
```

### Zero Moment Point (ZMP) Control

The Zero Moment Point is the point on the ground where the moment of the ground reaction force equals zero:

```python
def compute_zmp(com_pos, com_acc, z_height=0.8, gravity=9.81):
    """
    Compute Zero Moment Point from CoM position and acceleration
    """
    # ZMP = CoM - (h/g) * CoM_acc
    zmp_x = com_pos[0] - (z_height / gravity) * com_acc[0]
    zmp_y = com_pos[1] - (z_height / gravity) * com_acc[1]
    return np.array([zmp_x, zmp_y, 0])

def zmp_stability_check(zmp, support_polygon, margin=0.05):
    """
    Check if ZMP is within support polygon with safety margin
    """
    from matplotlib.path import Path
    # Create a smaller polygon with margin
    center = np.mean(support_polygon, axis=0)
    shrunk_polygon = center + (1 - margin) * (support_polygon - center)
    path = Path(shrunk_polygon)
    return path.contains_point(zmp[:2])
```

## Simple Balance Controller

A basic balance controller that adjusts CoM position:

```python
class SimpleBalanceController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # PID controllers for balance
        self.x_pid = PIDController(kp=10.0, ki=1.0, kd=2.0)
        self.y_pid = PIDController(kp=10.0, ki=1.0, kd=2.0)

        self.previous_com_pos = np.zeros(3)
        self.previous_com_vel = np.zeros(3)

    def compute_balance_torques(self, current_com_pos, desired_com_pos, dt):
        """
        Compute torques needed for balance control
        """
        # Compute error in CoM position
        pos_error = desired_com_pos[:2] - current_com_pos[:2]

        # Use PID to compute corrective torques
        torque_x = self.x_pid.update(desired_com_pos[0], current_com_pos[0], dt)
        torque_y = self.y_pid.update(desired_com_pos[1], current_com_pos[1], dt)

        return np.array([torque_x, torque_y, 0])

    def update_com_state(self, current_com_pos, dt):
        """
        Update CoM velocity and acceleration estimates
        """
        if hasattr(self, 'previous_com_pos'):
            current_vel = (current_com_pos - self.previous_com_pos) / dt
            if hasattr(self, 'previous_com_vel'):
                current_acc = (current_vel - self.previous_com_vel) / dt
            else:
                current_acc = np.zeros(3)
        else:
            current_vel = np.zeros(3)
            current_acc = np.zeros(3)

        self.previous_com_vel = current_vel
        self.previous_com_pos = current_com_pos

        return current_vel, current_acc
```

## Inverted Pendulum Model

The inverted pendulum is a fundamental model for balance control:

```python
class InvertedPendulumModel:
    def __init__(self, height=0.8, gravity=9.81):
        self.height = height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / height)

    def integrate_motion(self, zmp_ref, dt, current_com, current_vel):
        """
        Integrate inverted pendulum dynamics
        """
        # ẍ = ω² * (x - zmp)
        com_acc = self.omega**2 * (current_com[:2] - zmp_ref)

        # Update velocity and position
        new_vel = current_vel[:2] + com_acc * dt
        new_pos = current_com[:2] + new_vel * dt + 0.5 * com_acc * dt**2

        # Return new CoM position (with z unchanged) and velocity
        new_com = np.array([new_pos[0], new_pos[1], current_com[2]])
        new_vel_full = np.array([new_vel[0], new_vel[1], current_vel[2]])

        return new_com, new_vel_full, np.concatenate([com_acc, np.array([0])])

    def compute_zmp(self, com_pos, com_acc):
        """
        Compute ZMP from CoM state
        """
        # ZMP = CoM - (h/g) * CoM_acc
        zmp = com_pos[:2] - (self.height / self.gravity) * com_acc[:2]
        return zmp
```

## Whole-Body Balance Control

Coordinating multiple joints for balance:

```python
class WholeBodyBalanceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.balance_controller = SimpleBalanceController()
        self.pendulum_model = InvertedPendulumModel()

        # Joint weighting for balance (torso and legs more important for balance)
        self.balance_weights = np.array([2.0, 2.0, 2.0,  # torso
                                        1.5, 1.5, 1.5,  # legs
                                        0.5, 0.5, 0.5,  # arms
                                        0.1, 0.1])      # head

    def compute_balance_joints(self, current_config, desired_com, current_com, dt):
        """
        Compute joint adjustments for balance maintenance
        """
        # Calculate CoM error
        com_error = desired_com[:2] - current_com[:2]

        # Map CoM error to joint space using Jacobian
        jacobian = self.robot_model.compute_com_jacobian(current_config)

        # Weighted pseudo-inverse for prioritized control
        weights = np.diag(self.balance_weights)
        weighted_jacobian = weights @ jacobian
        joint_correction = np.linalg.pinv(weighted_jacobian) @ np.concatenate([com_error, np.zeros(4)])

        # Apply limits to prevent excessive joint movements
        joint_correction = np.clip(joint_correction, -0.1, 0.1)  # Limit to 0.1 rad adjustments

        return joint_correction

    def compute_balance_strategy(self, current_state, support_polygon, dt):
        """
        Determine appropriate balance strategy based on stability margin
        """
        # Calculate distance from CoM to support polygon boundary
        com_pos = current_state['com_position']
        from matplotlib.path import Path
        path = Path(support_polygon)

        if not path.contains_point(com_pos[:2]):
            # CoM is outside support - stepping strategy needed
            return "stepping"
        else:
            # Calculate distance to boundary
            distances = []
            for i in range(len(support_polygon)):
                p1 = support_polygon[i]
                p2 = support_polygon[(i + 1) % len(support_polygon)]
                dist = self.point_to_line_distance(com_pos[:2], p1, p2)
                distances.append(dist)

            min_distance = min(distances)

            if min_distance < 0.05:  # Less than 5cm from edge
                return "hip_strategy"  # Use hip movements
            else:
                return "ankle_strategy"  # Use ankle adjustments

    def point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate distance from point to line segment
        """
        # Vector from line_start to point
        v = point - line_start
        # Vector along line
        s = line_end - line_start

        # Project v onto s
        proj_len = np.dot(v, s) / np.dot(s, s)

        if proj_len < 0:
            # Closest point is line_start
            return np.linalg.norm(point - line_start)
        elif proj_len > 1:
            # Closest point is line_end
            return np.linalg.norm(point - line_end)
        else:
            # Closest point is along line
            closest = line_start + proj_len * s
            return np.linalg.norm(point - closest)
```

## Reactive Balance Control

Reacting to external disturbances:

```python
class ReactiveBalanceController:
    def __init__(self):
        self.disturbance_threshold = 50.0  # Newtons
        self.recovery_active = False
        self.recovery_start_time = 0.0

    def detect_disturbance(self, force_sensors, torque_sensors, time):
        """
        Detect external disturbances using sensor data
        """
        total_force = np.linalg.norm(force_sensors)
        total_torque = np.linalg.norm(torque_sensors)

        return (total_force > self.disturbance_threshold or
                total_torque > self.disturbance_threshold * 0.1)

    def compute_recovery_action(self, current_state, disturbance_direction, strategy="ankle"):
        """
        Compute recovery action based on disturbance
        """
        if strategy == "ankle":
            # Ankle strategy: Use ankle torques for small disturbances
            recovery_torque = -disturbance_direction * 20  # Proportional to disturbance
            return np.array([recovery_torque[0], recovery_torque[1], 0, 0, 0, 0])  # Ankle torques
        elif strategy == "hip":
            # Hip strategy: Use hip movements for medium disturbances
            hip_torque = -disturbance_direction * 50
            return np.array([0, 0, 0, hip_torque[0], hip_torque[1], 0])  # Hip torques
        elif strategy == "stepping":
            # Stepping strategy: Plan a step to expand support base
            step_direction = -disturbance_direction
            step_size = min(0.3, np.linalg.norm(step_direction) * 0.5)  # Max 30cm step
            return step_direction / (np.linalg.norm(step_direction) + 1e-6) * step_size

    def select_recovery_strategy(self, com_state, support_polygon, disturbance_magnitude):
        """
        Select appropriate recovery strategy based on situation
        """
        from matplotlib.path import Path
        path = Path(support_polygon)

        com_in_support = path.contains_point(com_state[:2])

        if disturbance_magnitude > 100:  # Large disturbance
            return "stepping"
        elif not com_in_support or disturbance_magnitude > 50:  # Medium disturbance
            return "hip"
        else:  # Small disturbance
            return "ankle"
```

## Practical Implementation Example

Putting it all together in a practical balance controller:

```python
class PracticalBalanceController:
    def __init__(self, com_height=0.8):
        self.inverted_pendulum = InvertedPendulumModel(height=com_height)
        self.whole_body_controller = WholeBodyBalanceController(robot_model=None)  # Simplified
        self.reactive_controller = ReactiveBalanceController()

        # Balance control parameters
        self.com_reference = np.array([0.0, 0.0, com_height])
        self.max_balance_torque = 100.0
        self.balance_gains = np.array([50.0, 50.0, 20.0])  # For different directions

    def update_balance(self, sensor_data, joint_positions, dt):
        """
        Main balance control update function
        """
        # Extract sensor data
        current_com = sensor_data['com_position']
        current_com_vel = sensor_data['com_velocity']
        current_com_acc = sensor_data['com_acceleration']
        support_polygon = sensor_data['support_polygon']
        force_sensors = sensor_data.get('force_sensors', np.zeros(6))
        torque_sensors = sensor_data.get('torque_sensors', np.zeros(6))

        # Detect disturbances
        if self.reactive_controller.detect_disturbance(force_sensors, torque_sensors, 0):
            # Use reactive control for disturbances
            disturbance_direction = force_sensors[:2] / (np.linalg.norm(force_sensors[:2]) + 1e-6)
            strategy = self.reactive_controller.select_recovery_strategy(
                current_com, support_polygon, np.linalg.norm(force_sensors)
            )
            recovery_action = self.reactive_controller.compute_recovery_action(
                current_com, disturbance_direction, strategy
            )
            return recovery_action

        # Normal balance control
        else:
            # Calculate ZMP
            current_zmp = self.inverted_pendulum.compute_zmp(current_com, current_com_acc)

            # Check stability
            is_stable = zmp_stability_check(current_zmp, support_polygon)

            if not is_stable:
                # Need to adjust CoM to bring ZMP back to safe region
                center_of_support = np.mean(support_polygon, axis=0)
                desired_zmp = center_of_support[:2]

                # Use inverted pendulum model to determine CoM adjustment needed
                desired_com = current_com.copy()
                desired_com[:2] = desired_zmp + (self.inverted_pendulum.height / self.inverted_pendulum.gravity) * current_com_acc[:2]

                # Compute balance torques
                balance_torques = self.compute_balance_torques(
                    current_com, desired_com, dt
                )

                return np.clip(balance_torques, -self.max_balance_torque, self.max_balance_torque)
            else:
                # System is stable, return minimal corrective torques
                return np.zeros(6)  # No additional torque needed

    def compute_balance_torques(self, current_com, desired_com, dt):
        """
        Compute balance torques using PID control
        """
        # Calculate error
        pos_error = desired_com[:2] - current_com[:2]

        # Simple proportional control (in practice, you'd use a more sophisticated controller)
        torques = self.balance_gains[:2] * pos_error
        torques = np.append(torques, np.zeros(4))  # Add zeros for other axes

        return torques

# Example usage
def example_balance_control():
    controller = PracticalBalanceController(com_height=0.85)

    # Simulated sensor data
    sensor_data = {
        'com_position': np.array([0.02, -0.01, 0.85]),
        'com_velocity': np.array([0.01, -0.005, 0]),
        'com_acceleration': np.array([0.1, -0.05, 0]),
        'support_polygon': np.array([[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]]),
        'force_sensors': np.array([5, -3, 0, 0, 0, 0]),
        'torque_sensors': np.array([2, -1, 0, 0, 0, 0])
    }

    joint_positions = np.zeros(20)  # 20 joint robot
    dt = 0.005  # 200 Hz control rate

    balance_torques = controller.update_balance(sensor_data, joint_positions, dt)
    print(f"Balance torques: {balance_torques}")

if __name__ == "__main__":
    example_balance_control()
```

## Best Practices

1. **Start simple**: Begin with basic CoM control before implementing complex strategies
2. **Prioritize safety**: Always have emergency stop mechanisms
3. **Test incrementally**: Validate each component separately before integration
4. **Consider multiple strategies**: Use different approaches for different disturbance magnitudes
5. **Validate in simulation**: Test extensively in simulation before real robot deployment
6. **Monitor stability margins**: Keep adequate safety margins in all conditions

## Exercises

1. Implement a balance controller that switches between ankle and hip strategies based on disturbance magnitude.
2. Create a ZMP trajectory generator for walking patterns.
3. Design a balance controller that considers angular momentum for more robust stability.