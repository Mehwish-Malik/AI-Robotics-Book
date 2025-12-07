---
sidebar_position: 4
---

# Simple Locomotion

## Overview

Locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control systems to maintain balance while moving. This tutorial covers fundamental locomotion concepts including gait generation, ZMP-based walking, and basic stepping patterns.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand the basics of bipedal locomotion
- Generate simple walking patterns
- Implement ZMP-based walking control
- Create basic stepping and balance recovery patterns

## Locomotion Fundamentals

### Gait Cycle Phases

Bipedal locomotion consists of several phases:
- **Stance Phase**: When the foot is in contact with the ground
- **Swing Phase**: When the foot is off the ground moving forward
- **Double Support Phase**: When both feet are in contact
- **Single Support Phase**: When only one foot is in contact

```python
import numpy as np
import matplotlib.pyplot as plt

class GaitPhase:
    """Represents a phase in the gait cycle"""
    def __init__(self, name, start_time, end_time, support_leg):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.support_leg = support_leg  # 'left', 'right', or 'both'
        self.duration = end_time - start_time

class GaitGenerator:
    """Generate basic walking gaits"""
    def __init__(self, step_length=0.3, step_width=0.2, step_time=0.8):
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.gait_phases = self._generate_gait_phases()

    def _generate_gait_phases(self):
        """Generate the phases of a basic walking gait"""
        phases = []

        # Single support phase - left foot supports, right foot swings
        phases.append(GaitPhase("Left Support", 0, self.step_time/2, "left"))

        # Double support phase - both feet on ground
        phases.append(GaitPhase("Double Support", self.step_time/2, self.step_time/2 + 0.05, "both"))

        # Single support phase - right foot supports, left foot swings
        phases.append(GaitPhase("Right Support", self.step_time/2 + 0.05, self.step_time, "right"))

        return phases

    def get_current_phase(self, time):
        """Get the current gait phase based on time"""
        cycle_time = time % self.step_time
        for phase in self.gait_phases:
            if phase.start_time <= cycle_time <= phase.end_time:
                return phase
        return self.gait_phases[-1]  # Return last phase if none match

    def generate_foot_trajectory(self, start_pos, step_num, side="left"):
        """Generate foot trajectory for a step"""
        # Define key points in the foot trajectory
        lift_height = 0.05  # 5cm lift

        # Start position
        start_x = start_pos[0] + (step_num * self.step_length)
        start_y = start_pos[1] + (self.step_width if side == "right" else -self.step_width)

        # End position (next step)
        end_x = start_x + self.step_length
        end_y = start_pos[1] + (-self.step_width if side == "right" else self.step_width)

        # Generate trajectory points
        t = np.linspace(0, 1, 50)

        # X trajectory (smooth interpolation)
        x_traj = start_x + (end_x - start_x) * t

        # Y trajectory (return to center line)
        y_traj = start_y + (end_y - start_y) * t

        # Z trajectory (parabolic lift and place)
        z_lift = 4 * lift_height * t * (1 - t)  # Parabolic curve

        return np.column_stack([x_traj, y_traj, z_lift])

# Example usage
gait_gen = GaitGenerator(step_length=0.3, step_width=0.15, step_time=0.8)
current_phase = gait_gen.get_current_phase(0.4)
print(f"Current phase at t=0.4s: {current_phase.name}, support: {current_phase.support_leg}")
```

## Inverted Pendulum Walking Model

The linear inverted pendulum is a fundamental model for walking:

```python
class LinearInvertedPendulum:
    """Linear inverted pendulum model for walking"""
    def __init__(self, height=0.8, gravity=9.81):
        self.height = height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / height)

    def compute_com_trajectory(self, zmp_trajectory, dt=0.005):
        """Compute CoM trajectory from ZMP reference"""
        n_steps = len(zmp_trajectory)
        com_trajectory = np.zeros((n_steps, 3))
        com_velocity = np.zeros((n_steps, 3))

        # Initial conditions
        com_trajectory[0] = [0, 0, self.height]
        com_velocity[0] = [0, 0, 0]

        # Integrate inverted pendulum dynamics
        for i in range(1, n_steps):
            # ẍ = ω² * (x - zmp)
            com_acc = self.omega**2 * (com_trajectory[i-1, :2] - zmp_trajectory[i])
            com_acc_full = np.append(com_acc, [0])  # No vertical acceleration

            # Update velocity and position
            com_velocity[i] = com_velocity[i-1] + com_acc_full * dt
            com_trajectory[i] = com_trajectory[i-1] + com_velocity[i] * dt + 0.5 * com_acc_full * dt**2

            # Keep constant height
            com_trajectory[i, 2] = self.height

        return com_trajectory, com_velocity

    def compute_zmp_from_com(self, com_pos, com_acc):
        """Compute ZMP from CoM state"""
        zmp_x = com_pos[0] - (com_acc[0] * self.height / self.gravity)
        zmp_y = com_pos[1] - (com_acc[1] * self.height / self.gravity)
        return np.array([zmp_x, zmp_y])
```

## ZMP-Based Walking Controller

Implementing a ZMP-based walking controller:

```python
class ZMPWalkingController:
    """ZMP-based walking controller"""
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # Walking parameters
        self.step_length = 0.3
        self.step_width = 0.15
        self.step_time = 0.8
        self.zmp_reference_offset = 0.02  # Small offset for stability

        # PID controller for ZMP tracking
        self.zmp_x_pid = PIDController(kp=50.0, ki=5.0, kd=10.0)
        self.zmp_y_pid = PIDController(kp=50.0, ki=5.0, kd=10.0)

        # Foot placement variables
        self.left_foot_pos = np.array([0.0, self.step_width, 0.0])
        self.right_foot_pos = np.array([0.0, -self.step_width, 0.0])
        self.next_foot_placement = "left"
        self.step_count = 0

        # Walking state
        self.is_walking = False
        self.walk_time = 0.0

    def start_walking(self, step_length=0.3, step_width=0.15, step_time=0.8):
        """Initialize walking with given parameters"""
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.is_walking = True
        self.walk_time = 0.0
        self.step_count = 0

    def generate_zmp_trajectory(self, walk_time):
        """Generate reference ZMP trajectory for walking"""
        # Create a periodic ZMP pattern
        cycle_time = walk_time % (self.step_time * 2)  # Two steps per cycle

        # Basic ZMP pattern: oscillate around nominal position
        nominal_x = (self.step_count + cycle_time / (self.step_time * 2)) * self.step_length

        # Add small oscillations for stability
        oscillation = 0.01 * np.sin(2 * np.pi * cycle_time / self.step_time)

        # Y position alternates between feet
        if self.next_foot_placement == "left":
            nominal_y = -self.zmp_reference_offset
        else:
            nominal_y = self.zmp_reference_offset

        return np.array([nominal_x, nominal_y])

    def update_walking(self, current_com, current_com_vel, dt):
        """Update walking controller"""
        if not self.is_walking:
            return np.zeros(6)  # No torques when not walking

        # Update timing
        self.walk_time += dt

        # Generate reference ZMP
        zmp_ref = self.generate_zmp_trajectory(self.walk_time)

        # Compute current ZMP
        com_acc = self.estimate_com_acceleration(current_com, current_com_vel, dt)
        current_zmp = self.compute_zmp_from_com(current_com, com_acc)

        # Compute ZMP error and required torques
        zmp_error = zmp_ref - current_zmp[:2]

        # Use PID controllers to compute corrective torques
        torque_x = self.zmp_x_pid.update(zmp_ref[0], current_zmp[0], dt)
        torque_y = self.zmp_y_pid.update(zmp_ref[1], current_zmp[1], dt)

        # Map to joint torques (simplified mapping)
        joint_torques = self.map_zmp_torques_to_joints(torque_x, torque_y)

        # Check for step timing
        self.check_step_timing(current_com)

        return joint_torques

    def estimate_com_acceleration(self, current_com, current_com_vel, dt):
        """Estimate CoM acceleration from velocity"""
        # This would typically come from a state estimator
        # For now, we'll use a simple backward difference
        if not hasattr(self, 'prev_vel'):
            self.prev_vel = current_com_vel.copy()
            return np.zeros(3)

        com_acc = (current_com_vel - self.prev_vel) / dt
        self.prev_vel = current_com_vel.copy()
        return com_acc

    def compute_zmp_from_com(self, com_pos, com_acc):
        """Compute ZMP from CoM state"""
        zmp_x = com_pos[0] - (com_acc[0] * self.com_height / self.gravity)
        zmp_y = com_pos[1] - (com_acc[1] * self.com_height / self.gravity)
        return np.array([zmp_x, zmp_y, 0])

    def map_zmp_torques_to_joints(self, torque_x, torque_y):
        """Map ZMP torques to joint torques (simplified)"""
        # This is a simplified mapping - in reality, this would involve
        # whole-body control and inverse dynamics
        torques = np.zeros(6)  # 6-DOF torques at base

        # Map X torque to hip pitch
        torques[3] = torque_x * 50  # Scale factor

        # Map Y torque to hip roll
        torques[4] = -torque_y * 50  # Scale factor

        return torques

    def check_step_timing(self, current_com):
        """Check if it's time to take the next step"""
        cycle_time = self.walk_time % self.step_time

        # Take step when we reach the end of the step phase
        if (cycle_time < self.step_time / 4 or  # Beginning of new step phase
            (self.walk_time // self.step_time) > self.step_count):  # New step interval

            new_step_count = int(self.walk_time // self.step_time)
            if new_step_count > self.step_count:
                self.step_count = new_step_count
                # Place next foot
                if self.next_foot_placement == "left":
                    self.left_foot_pos = np.array([
                        self.step_count * self.step_length,
                        self.step_width,
                        0.0
                    ])
                    self.next_foot_placement = "right"
                else:
                    self.right_foot_pos = np.array([
                        self.step_count * self.step_length,
                        -self.step_width,
                        0.0
                    ])
                    self.next_foot_placement = "left"

    def stop_walking(self):
        """Stop walking motion"""
        self.is_walking = False
        # Implement stopping strategy (e.g., capture point approach)
```

## Simple Stepping Pattern

Creating basic stepping patterns for balance recovery:

```python
class SteppingPatternGenerator:
    """Generate stepping patterns for balance recovery"""
    def __init__(self, step_length=0.3, step_width=0.15):
        self.step_length = step_length
        self.step_width = step_width

    def compute_capture_point(self, com_pos, com_vel, com_height=0.8, gravity=9.81):
        """Compute capture point from CoM state"""
        omega = np.sqrt(gravity / com_height)
        capture_point = com_pos[:2] + com_vel[:2] / omega
        return capture_point

    def generate_recovery_step(self, current_support_pos, com_state, strategy="capture_point"):
        """Generate a recovery step based on current state"""
        com_pos = com_state['position']
        com_vel = com_state['velocity']
        com_height = com_state.get('height', 0.8)

        if strategy == "capture_point":
            # Place foot at or beyond capture point
            capture_point = self.compute_capture_point(com_pos, com_vel, com_height)

            # Ensure step is in a reasonable direction
            step_direction = capture_point - current_support_pos[:2]
            step_distance = np.linalg.norm(step_direction)

            # Limit step size
            max_step = self.step_length * 1.5  # 1.5x normal step
            if step_distance > max_step:
                step_direction = step_direction / step_distance * max_step

            step_pos = current_support_pos.copy()
            step_pos[:2] = current_support_pos[:2] + step_direction

            return step_pos

        elif strategy == "directional":
            # Step in the direction opposite to CoM offset
            com_offset = com_pos[:2] - current_support_pos[:2]
            step_direction = -com_offset
            step_distance = np.linalg.norm(step_direction)

            # Normalize and scale
            if step_distance > 0:
                step_direction = step_direction / step_distance
                step_size = min(step_distance * 1.2, self.step_length * 1.5)
                step_direction *= step_size

            step_pos = current_support_pos.copy()
            step_pos[:2] = current_support_pos[:2] + step_direction

            return step_pos

    def generate_omni_directional_step(self, direction, distance=None):
        """Generate a step in any direction"""
        if distance is None:
            distance = self.step_length

        step_x = distance * np.cos(direction)
        step_y = distance * np.sin(direction)

        return np.array([step_x, step_y, 0.0])

# Example: Recovery step generation
recovery_gen = SteppingPatternGenerator()
current_support = np.array([0.0, 0.1, 0.0])  # Left foot position
com_state = {
    'position': np.array([0.1, 0.15, 0.8]),
    'velocity': np.array([0.05, 0.02, 0.0]),
    'height': 0.8
}

recovery_step = recovery_gen.generate_recovery_step(current_support, com_state)
print(f"Recovery step position: {recovery_step}")
```

## Walk Pattern Generator

Creating complete walk patterns:

```python
class WalkPatternGenerator:
    """Generate complete walking patterns"""
    def __init__(self, step_length=0.3, step_width=0.15, step_time=0.8):
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.stepping_gen = SteppingPatternGenerator(step_length, step_width)

    def generate_walk_pattern(self, num_steps, start_pos=np.array([0, 0, 0.8])):
        """Generate a complete walk pattern"""
        pattern = {
            'com_trajectory': [],
            'left_foot_trajectory': [],
            'right_foot_trajectory': [],
            'zmp_trajectory': [],
            'timestamps': []
        }

        # Initial positions
        left_foot = start_pos + np.array([0, self.step_width, 0])
        right_foot = start_pos + np.array([0, -self.step_width, 0])

        com_pos = start_pos.copy()
        com_vel = np.array([0, 0, 0])

        dt = 0.005  # 200 Hz
        time = 0

        for step in range(num_steps):
            # Simulate one step cycle
            step_duration = self.step_time
            steps_in_cycle = int(step_duration / dt)

            for i in range(steps_in_cycle):
                t = i * dt

                # Simple forward progression
                com_pos[0] = start_pos[0] + (step + t/self.step_time) * self.step_length
                com_pos[1] = start_pos[1] + (-1)**step * 0.02 * np.sin(2*np.pi*t/self.step_time)  # Small lateral sway

                # Foot trajectories
                if step % 2 == 0:  # Left foot swings
                    left_x = step * self.step_length + t/self.step_time * self.step_length
                    left_y = self.step_width * (1 - t/self.step_time)  # Return to center
                    left_foot = np.array([left_x, left_y, 0])
                    right_foot = np.array([(step+1) * self.step_length, -self.step_width, 0])
                else:  # Right foot swings
                    right_x = step * self.step_length + t/self.step_time * self.step_length
                    right_y = -self.step_width * (1 - t/self.step_time)  # Return to center
                    right_foot = np.array([right_x, right_y, 0])
                    left_foot = np.array([step * self.step_length, self.step_width, 0])

                # Calculate ZMP (simplified)
                zmp = np.array([com_pos[0] - 0.02, com_pos[1]])  # Small offset

                # Store data
                pattern['com_trajectory'].append(com_pos.copy())
                pattern['left_foot_trajectory'].append(left_foot.copy())
                pattern['right_foot_trajectory'].append(right_foot.copy())
                pattern['zmp_trajectory'].append(zmp.copy())
                pattern['timestamps'].append(time)

                time += dt

        # Convert to numpy arrays
        for key in pattern:
            if key != 'timestamps':
                pattern[key] = np.array(pattern[key])

        return pattern

    def generate_turning_pattern(self, angle, turn_radius=0.5):
        """Generate a turning walking pattern"""
        # Calculate number of steps needed for turn
        arc_length = turn_radius * abs(angle)
        num_steps = int(arc_length / self.step_length) + 1

        pattern = {
            'com_trajectory': [],
            'left_foot_trajectory': [],
            'right_foot_trajectory': [],
            'zmp_trajectory': [],
            'timestamps': []
        }

        current_angle = 0
        current_pos = np.array([0, 0, 0.8])

        for step in range(num_steps):
            # Calculate turning angle for this step
            step_angle = angle * step / num_steps
            next_step_angle = angle * (step + 1) / num_steps

            # Calculate positions along arc
            avg_angle = (current_angle + step_angle) / 2
            dx = (next_step_angle - current_angle) * turn_radius * np.cos(avg_angle + np.pi/2)
            dy = (next_step_angle - current_angle) * turn_radius * np.sin(avg_angle + np.pi/2)

            current_pos[0] += dx
            current_pos[1] += dy

            # Foot positions during turn
            # Simplified turning model
            left_pos = current_pos + np.array([0, self.step_width, 0])
            right_pos = current_pos + np.array([0, -self.step_width, 0])

            pattern['com_trajectory'].append(current_pos.copy())
            pattern['left_foot_trajectory'].append(left_pos)
            pattern['right_foot_trajectory'].append(right_pos)
            pattern['zmp_trajectory'].append(current_pos[:2].copy())
            pattern['timestamps'].append(step * self.step_time)

            current_angle = step_angle

        # Convert to numpy arrays
        for key in pattern:
            if key != 'timestamps':
                pattern[key] = np.array(pattern[key])

        return pattern

# Example usage
pattern_gen = WalkPatternGenerator(step_length=0.3, step_width=0.15, step_time=0.8)
walk_pattern = pattern_gen.generate_walk_pattern(4)  # 4 steps forward

print(f"Generated walk pattern with {len(walk_pattern['com_trajectory'])} time steps")
print(f"Final CoM position: {walk_pattern['com_trajectory'][-1]}")
```

## Practical Walking Controller

A complete walking controller that combines all elements:

```python
class PracticalWalkingController:
    """Practical walking controller combining all elements"""
    def __init__(self, com_height=0.8):
        self.zmp_controller = ZMPWalkingController(com_height=com_height)
        self.pattern_generator = WalkPatternGenerator()
        self.stepping_generator = SteppingPatternGenerator()

        # Walking state
        self.current_com = np.array([0, 0, com_height])
        self.current_com_vel = np.array([0, 0, 0])
        self.walk_speed = 0.0  # Current walking speed
        self.walk_direction = 0.0  # Walking direction in radians

    def start_walking(self, speed=0.3, direction=0.0):
        """Start walking at specified speed and direction"""
        self.walk_speed = speed
        self.walk_direction = direction

        # Adjust step parameters based on speed
        step_time = max(0.6, 0.8 - speed * 0.2)  # Faster = shorter steps
        step_length = min(0.4, speed * step_time * 0.8)  # Adjust step length

        self.zmp_controller.start_walking(
            step_length=step_length,
            step_width=0.15,
            step_time=step_time
        )

    def update_walking(self, sensor_data, dt):
        """Main walking control update"""
        # Extract sensor data
        self.current_com = sensor_data.get('com_position', self.current_com)
        self.current_com_vel = sensor_data.get('com_velocity', self.current_com_vel)

        # Update ZMP-based walking controller
        torques = self.zmp_controller.update_walking(self.current_com, self.current_com_vel, dt)

        # Check for balance recovery needs
        com_offset = np.linalg.norm(self.current_com[:2])
        if com_offset > 0.15:  # Too far from center
            # Generate recovery step
            support_foot = self.get_current_support_foot()
            recovery_step = self.stepping_generator.generate_recovery_step(
                support_foot,
                {'position': self.current_com, 'velocity': self.current_com_vel}
            )
            print(f"Balance recovery step needed at: {recovery_step}")

        return torques

    def get_current_support_foot(self):
        """Get position of current support foot"""
        # This would typically come from contact sensors
        # For now, we'll use the controller's internal state
        if self.zmp_controller.next_foot_placement == "left":
            return self.zmp_controller.right_foot_pos
        else:
            return self.zmp_controller.left_foot_pos

    def turn_in_place(self, angle):
        """Execute turning in place"""
        # Generate turning pattern
        turn_pattern = self.pattern_generator.generate_turning_pattern(angle)
        return turn_pattern

    def stop_walking(self):
        """Stop walking with safe deceleration"""
        self.zmp_controller.stop_walking()
        self.walk_speed = 0.0

# Example usage
def example_walking():
    controller = PracticalWalkingController(com_height=0.85)

    # Start walking forward
    controller.start_walking(speed=0.4, direction=0.0)  # 0.4 m/s forward

    # Simulated sensor data
    sensor_data = {
        'com_position': np.array([0.1, 0.0, 0.85]),
        'com_velocity': np.array([0.3, 0.0, 0.0])
    }

    dt = 0.005  # 200 Hz

    for i in range(10):  # 10 control cycles
        torques = controller.update_walking(sensor_data, dt)
        print(f"Control cycle {i+1}: Torques = {torques[:3]}...")  # Show first 3 torques

        # Update sensor data for next cycle (simulated)
        sensor_data['com_position'][0] += sensor_data['com_velocity'][0] * dt
        sensor_data['com_position'][1] += 0.001 * np.sin(i)  # Small lateral movement

if __name__ == "__main__":
    example_walking()
```

## Best Practices

1. **Start with simple patterns**: Begin with basic stepping before complex gaits
2. **Prioritize stability**: Always ensure ZMP remains in support polygon
3. **Test incrementally**: Validate each component separately
4. **Consider terrain**: Adapt step patterns for different surfaces
5. **Implement safety margins**: Keep adequate stability margins
6. **Smooth transitions**: Ensure smooth transitions between steps

## Exercises

1. Implement a walking controller that can adjust step length based on walking speed.
2. Create a turning controller that maintains balance during turns.
3. Design a disturbance recovery system that automatically takes recovery steps.