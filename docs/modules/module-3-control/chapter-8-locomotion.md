---
sidebar_position: 2
---

# Chapter 8: Locomotion and Gait Control

## Summary

This chapter delves into the complex mechanics of bipedal locomotion, exploring the principles of walking patterns, gait generation, and the control systems that enable humanoid robots to move efficiently on two legs. We'll examine various walking patterns, gait control algorithms, and the integration of balance control with locomotion. Understanding locomotion is fundamental to creating humanoid robots that can navigate real-world environments.

## Learning Outcomes

By the end of this chapter, you will be able to:
- Understand the principles of bipedal locomotion and walking patterns
- Analyze different gait generation algorithms and their applications
- Implement Zero Moment Point (ZMP) and Capture Point control strategies
- Design controllers for dynamic balance during locomotion
- Evaluate locomotion performance and stability metrics

## Key Concepts

- **Bipedal Locomotion**: Two-legged walking motion
- **Gait Cycle**: Complete walking cycle including stance and swing phases
- **Zero Moment Point (ZMP)**: Point where the moment of ground reaction force equals zero
- **Capture Point**: Point where the robot must step to come to a complete stop
- **Limit Cycles**: Stable periodic solutions representing steady-state walking
- **Phase-Based Control**: Control strategies that depend on gait phase
- **Dynamic Balance**: Balance maintained through active control during motion

## Introduction to Bipedal Locomotion

Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control systems to maintain balance while moving. Unlike wheeled robots, bipedal robots have intermittent ground contact and must actively maintain stability throughout the walking cycle.

### Locomotion Fundamentals

**Stance Phase**: When the foot is in contact with the ground
**Swing Phase**: When the foot is off the ground moving forward
**Double Support Phase**: When both feet are in contact with the ground
**Single Support Phase**: When only one foot is in contact with the ground

### Walking Pattern Characteristics

**Step Length**: Distance between consecutive foot placements
**Step Width**: Lateral distance between feet
**Step Time**: Duration of a complete gait cycle
**Walking Speed**: Forward velocity of the robot
**Cadence**: Steps per unit time

## Gait Generation and Pattern Formation

### Inverted Pendulum Model

The simplest model for bipedal walking is the inverted pendulum:

```
ẍ = g/h * x
```

Where:
- x = horizontal position of center of mass
- h = height of center of mass
- g = gravitational acceleration

This model shows that the center of mass behaves like an unstable pendulum that must be actively controlled.

### Linear Inverted Pendulum Mode (LIPM)

A more tractable model linearizes the inverted pendulum:

```
ẍ = ω² * (x - z)
```

Where:
- ω² = g/h
- z = ZMP position

This model allows for analytical solutions and is widely used in gait planning.

### Predefined Gait Patterns

**Open-Loop Patterns**: Fixed sequences of joint angles
- Advantages: Simple, predictable, energy-efficient
- Disadvantages: Poor adaptability to disturbances

**Closed-Loop Patterns**: Adaptive patterns that respond to feedback
- Advantages: Better disturbance rejection, adaptability
- Disadvantages: More complex, potential stability issues

### Central Pattern Generators (CPGs)

CPGs are neural networks that generate rhythmic patterns:

```
dθ/dt = ω + ∑ w_ij * sin(θ_j - θ_i + φ_ij)
```

Where:
- θ_i = phase of oscillator i
- ω = natural frequency
- w_ij = coupling strength
- φ_ij = phase bias

## Zero Moment Point (ZMP) Control

### ZMP Definition

The Zero Moment Point is the point on the ground where the moment of the ground reaction force and gravity force equals zero:

```
ZMP_x = (M_x + F_z * h) / F_z
ZMP_y = (M_y + F_z * h) / F_z
```

Where:
- M_x, M_y = moments about x and y axes
- F_z = vertical ground reaction force
- h = height of center of mass

### ZMP Stability Criteria

For stable walking, the ZMP must remain within the support polygon (convex hull of ground contact points).

### ZMP Reference Trajectory Generation

**Preview Control**: Uses future reference trajectory to compute current control
```
u(k) = -Kx(k) - K_f * Σ(i=0 to N-1) G_f(i) * r(k+i)
```

Where:
- u = control input
- x = state
- r = reference trajectory
- N = preview horizon

### ZMP-Based Walking Pattern Generation

1. **Desired ZMP Trajectory**: Define stable ZMP pattern
2. **COM Trajectory**: Integrate inverted pendulum equation
3. **Foot Placement**: Determine foot positions based on COM trajectory
4. **Joint Trajectory**: Compute joint angles to achieve desired poses

## Capture Point Dynamics

### Capture Point Definition

The capture point indicates where the robot must step to come to a complete stop:

```
Capture Point = CoM Position + CoM Velocity / ω
```

Where ω = √(g/h), the natural frequency of the inverted pendulum.

### Capture Point Control Strategy

1. **Compute Capture Point**: From current CoM state
2. **Compare with Foot Location**: If capture point is outside support, step required
3. **Plan Step Location**: Place foot at or beyond capture point
4. **Adjust Control**: Modify walking pattern accordingly

### Capture Point vs. ZMP

**ZMP**: Used for trajectory planning and stability analysis
**Capture Point**: Used for step timing and placement decisions
**Relationship**: Capture point extends ZMP concept to include stopping capability

## Gait Control Algorithms

### Model Predictive Control (MPC) for Walking

MPC optimizes walking over a prediction horizon:

```
min Σ(k=0 to N-1) ||x(k) - x_ref(k)||²_Q + ||u(k)||²_R
s.t. x(k+1) = Ax(k) + Bu(k)
     ZMP constraints
     State constraints
```

### Feedback Linearization

Transform nonlinear dynamics into linear system:

```
τ = M(q)⁻¹ * (v_d - C(q, q̇)q̇ - G(q))
```

Where:
- τ = joint torques
- M = mass matrix
- C = Coriolis matrix
- G = gravity vector
- v_d = desired acceleration

### Hybrid Zero Dynamics (HZD)

Combine continuous dynamics with discrete events (foot impacts):

```
ẋ = f(x) + g(x)u,    if not in contact
x⁺ = Δ(x),          if in contact
```

## Dynamic Balance Control

### Balance Control Strategies

**Ankle Strategy**: Small balance adjustments using ankle torques
- Effective for small disturbances
- Low energy consumption
- Limited range of effectiveness

**Hip Strategy**: Larger adjustments using hip torques
- Effective for medium disturbances
- Higher energy consumption
- Greater range of effectiveness

**Stepping Strategy**: Taking a step to expand support base
- Effective for large disturbances
- High energy consumption
- Resets balance state

### Multi-Level Balance Control

**High Level**: Step planning and gait selection
**Mid Level**: ZMP/Capture Point tracking
**Low Level**: Joint-level control and actuator commands

### Disturbance Rejection

**Feedforward Compensation**: Predict and counteract known disturbances
**Feedback Control**: React to measured disturbances
**Adaptive Control**: Learn and adapt to disturbance patterns

## Technical Depth: Mathematical Models

### Walking Dynamics

The equation of motion for a bipedal robot:

```
M(q)q̈ + C(q, q̇)q̇ + G(q) = τ + JᵀF
```

Where:
- M(q) = mass/inertia matrix
- C(q, q̇) = Coriolis and centrifugal forces
- G(q) = gravitational forces
- τ = joint torques
- J = Jacobian matrix
- F = external forces

### Linear Inverted Pendulum Model

For ZMP-based control:

```
ẍ_com = g/h * (x_com - zmp)
ÿ_com = g/h * (y_com - zmp)
```

### Discrete State Model

At foot contact events:

```
x(k+1) = A_d * x(k) + B_d * u(k)
```

## Practical Applications

### Walking on Different Terrains

**Flat Ground**: Standard walking patterns
**Sloped Surfaces**: Adjusted body orientation and step parameters
**Stairs**: Specialized climbing/descending gaits
**Rough Terrain**: Adaptive foot placement and balance control
**Slippery Surfaces**: Reduced speed and modified gait patterns

### Speed Control

**Slow Walking**: Emphasize stability over speed
**Normal Walking**: Balance between stability and efficiency
**Fast Walking**: Increase step frequency and length
**Running**: Include flight phases (for capable robots)

### Turning and Maneuvering

**Pure Turning**: Rotate in place using differential stepping
**Curved Walking**: Combine forward and turning motions
**Sidestepping**: Lateral movement for obstacle avoidance
**Backward Walking**: Reverse locomotion for tight spaces

## Challenges

### Stability vs. Efficiency

Balancing stable walking with energy efficiency remains challenging.

### Terrain Adaptation

Adapting to unknown or changing terrain conditions in real-time.

### Disturbance Robustness

Maintaining stable walking despite external disturbances.

### Computational Complexity

Real-time computation of complex gait patterns and control laws.

## Figure List

1. **Figure 8.1**: Gait cycle phases and terminology
2. **Figure 8.2**: ZMP and Capture Point concepts
3. **Figure 8.3**: Inverted pendulum model visualization
4. **Figure 8.4**: Balance control strategy selection
5. **Figure 8.5**: Walking pattern generation flowchart

## Code Example: Gait Control Implementation

```python
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

@dataclass
class GaitPhase:
    """Represents a phase in the gait cycle"""
    name: str
    start_time: float
    end_time: float
    support_leg: str  # 'left', 'right', or 'both'
    duration: float

@dataclass
class WalkingState:
    """Current state of the walking robot"""
    com_pos: np.ndarray  # Center of mass position [x, y, z]
    com_vel: np.ndarray  # Center of mass velocity [vx, vy, vz]
    com_acc: np.ndarray  # Center of mass acceleration
    zmp_pos: np.ndarray  # Zero Moment Point [x, y]
    capture_point: np.ndarray  # Capture Point [x, y]
    support_foot_pos: np.ndarray  # Support foot position
    swing_foot_pos: np.ndarray    # Swing foot position
    gait_phase: str
    time_in_phase: float

class InvertedPendulumModel:
    """Simple inverted pendulum model for walking"""

    def __init__(self, height: float = 0.8, gravity: float = 9.81):
        self.height = height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / height)  # Natural frequency

    def compute_zmp(self, com_pos: np.ndarray, com_acc: np.ndarray) -> np.ndarray:
        """Compute ZMP from CoM position and acceleration"""
        # ZMP = CoM - (h/g) * CoM_acc
        zmp = com_pos[:2] - (self.height / self.gravity) * com_acc[:2]
        return zmp

    def compute_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """Compute capture point from CoM state"""
        # Capture Point = CoM + CoM_vel / omega
        capture_point = com_pos[:2] + com_vel[:2] / self.omega
        return capture_point

    def integrate_motion(self, zmp_ref: np.ndarray, dt: float,
                        current_com: np.ndarray, current_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate inverted pendulum dynamics"""
        # ẍ = ω² * (x - zmp)
        com_acc = self.omega**2 * (current_com[:2] - zmp_ref)

        # Update velocity and position
        new_vel = current_vel[:2] + com_acc * dt
        new_pos = current_com[:2] + new_vel * dt + 0.5 * com_acc * dt**2

        # Return new CoM position (with z unchanged) and velocity
        new_com = np.array([new_pos[0], new_pos[1], current_com[2]])
        new_vel_full = np.array([new_vel[0], new_vel[1], current_vel[2]])

        return new_com, new_vel_full, np.concatenate([com_acc, np.array([0])])

class GaitController:
    """Controller for generating and tracking walking gaits"""

    def __init__(self, step_length: float = 0.3, step_width: float = 0.2,
                 step_time: float = 0.8, height: float = 0.8):
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.inverted_pendulum = InvertedPendulumModel(height=height)

        # Walking parameters
        self.nominal_zmp_offset = 0.02  # Small offset for stability
        self.capture_point_threshold = 0.1  # When to step
        self.max_step_adjustment = 0.1     # Max adjustment to nominal step

        # Current walking state
        self.current_state = WalkingState(
            com_pos=np.array([0.0, 0.0, height]),
            com_vel=np.array([0.0, 0.0, 0.0]),
            com_acc=np.array([0.0, 0.0, 0.0]),
            zmp_pos=np.array([0.0, 0.0]),
            capture_point=np.array([0.0, 0.0]),
            support_foot_pos=np.array([0.0, -step_width/2, 0.0]),
            swing_foot_pos=np.array([0.0, step_width/2, 0.0]),
            gait_phase='double_support',
            time_in_phase=0.0
        )

        self.trajectory_queue = []
        self.is_walking = False

    def generate_zmp_trajectory(self, walking_speed: float, turn_rate: float = 0.0) -> np.ndarray:
        """Generate reference ZMP trajectory for walking"""
        # Create ZMP trajectory based on walking speed and turn rate
        num_steps = 10  # Plan 10 steps ahead
        dt = 0.01  # 100Hz control
        trajectory_length = int(self.step_time * num_steps / dt)

        zmp_trajectory = np.zeros((trajectory_length, 2))

        for i in range(trajectory_length):
            t = i * dt
            step_num = int(t / self.step_time)

            # Basic ZMP pattern: oscillate around nominal position
            nominal_x = walking_speed * t
            nominal_y = turn_rate * t * self.step_width  # Simplified turning

            # Add small oscillations for stability
            oscillation = 0.01 * np.sin(2 * np.pi * t / self.step_time)

            zmp_trajectory[i, 0] = nominal_x + oscillation
            zmp_trajectory[i, 1] = nominal_y + self.nominal_zmp_offset

        return zmp_trajectory

    def compute_foot_placement(self, current_support_foot: np.ndarray,
                             is_left_support: bool) -> np.ndarray:
        """Compute next foot placement based on current state"""
        # Calculate desired foot placement
        # This is a simplified version - in reality, this would consider balance state

        # Nominal step location
        if is_left_support:
            # Right foot should go forward and slightly inward
            dx = self.step_length
            dy = -self.step_width if self.current_state.gait_phase == 'left_swing' else self.step_width
        else:
            # Left foot should go forward and slightly inward
            dx = self.step_length
            dy = self.step_width if self.current_state.gait_phase == 'right_swing' else -self.step_width

        # Adjust for balance state (if capture point is far from support foot)
        capture_to_foot = self.current_state.capture_point - current_support_foot[:2]
        distance_to_capture = np.linalg.norm(capture_to_foot)

        if distance_to_capture > self.capture_point_threshold:
            # Step toward capture point
            direction = capture_to_foot / distance_to_capture
            adjustment = min(self.max_step_adjustment, distance_to_capture - self.capture_point_threshold)
            dx += direction[0] * adjustment
            dy += direction[1] * adjustment

        new_foot_pos = current_support_foot.copy()
        new_foot_pos[0] += dx
        new_foot_pos[1] += dy

        return new_foot_pos

    def generate_swing_foot_trajectory(self, start_pos: np.ndarray,
                                     end_pos: np.ndarray, phase: float) -> np.ndarray:
        """Generate smooth trajectory for swing foot"""
        # Use 3rd order polynomial for smooth lift and place
        # Phase from 0 (lift) to 1 (place)

        # Horizontal interpolation
        x = start_pos[0] + phase * (end_pos[0] - start_pos[0])
        y = start_pos[1] + phase * (end_pos[1] - start_pos[1])

        # Vertical trajectory (parabolic lift and place)
        # Maximum height at phase = 0.5
        vertical_phase = 4 * phase * (1 - phase)  # Parabolic profile
        z = start_pos[2] + 0.05 * vertical_phase  # 5cm maximum lift

        return np.array([x, y, z])

    def update_walking_state(self, dt: float, walking_speed: float = 0.1) -> WalkingState:
        """Update walking state based on control inputs"""
        # Generate reference ZMP
        zmp_ref = self.generate_zmp_trajectory(walking_speed)[int(self.current_state.time_in_phase/dt), :]

        # Integrate inverted pendulum dynamics
        new_com, new_vel, new_acc = self.inverted_pendulum.integrate_motion(
            zmp_ref, dt, self.current_state.com_pos, self.current_state.com_vel
        )

        # Update capture point
        capture_point = self.inverted_pendulum.compute_capture_point(new_com, new_vel)

        # Update ZMP (from actual CoM state)
        zmp = self.inverted_pendulum.compute_zmp(new_com, new_acc)

        # Update gait phase based on time
        self.current_state.time_in_phase += dt
        if self.current_state.time_in_phase >= self.step_time:
            self.current_state.time_in_phase = 0.0
            # Switch support leg
            if self.current_state.gait_phase == 'left_swing':
                self.current_state.gait_phase = 'right_swing'
            else:
                self.current_state.gait_phase = 'left_swing'

        # Update support and swing feet positions
        is_left_support = self.current_state.gait_phase == 'right_swing'

        # Compute next foot placement
        support_foot = self.current_state.support_foot_pos.copy()
        if is_left_support:
            # Right foot is swing foot, compute next right foot placement
            self.current_state.swing_foot_pos = self.compute_foot_placement(
                self.current_state.support_foot_pos, not is_left_support
            )
        else:
            # Left foot is swing foot, compute next left foot placement
            self.current_state.support_foot_pos = self.compute_foot_placement(
                self.current_state.swing_foot_pos, is_left_support
            )

        # Update current state
        self.current_state.com_pos = new_com
        self.current_state.com_vel = new_vel
        self.current_state.com_acc = new_acc
        self.current_state.zmp_pos = zmp
        self.current_state.capture_point = capture_point

        return self.current_state

    def start_walking(self, walking_speed: float = 0.1):
        """Initialize walking motion"""
        self.is_walking = True
        self.trajectory_queue = []
        # Initialize with basic walking parameters
        self.current_state.gait_phase = 'left_swing'

    def stop_walking(self):
        """Stop walking motion"""
        self.is_walking = False
        # Implement stopping strategy (e.g., capture point approach)
        self.slow_down()

class BalanceController:
    """Balance controller that works with gait controller"""

    def __init__(self, pendulum_model: InvertedPendulumModel):
        self.pendulum_model = pendulum_model
        self.k_p = 10.0  # Position gain
        self.k_d = 2.0   # Velocity gain

    def compute_balance_correction(self, desired_zmp: np.ndarray,
                                 actual_zmp: np.ndarray) -> np.ndarray:
        """Compute balance correction based on ZMP error"""
        zmp_error = desired_zmp - actual_zmp
        correction = self.k_p * zmp_error
        return correction

    def compute_ankle_torque(self, com_pos: np.ndarray, com_vel: np.ndarray,
                           zmp_ref: np.ndarray) -> np.ndarray:
        """Compute ankle torques for balance"""
        # Simple model: ankle torques proportional to ZMP error
        actual_zmp = self.pendulum_model.compute_zmp(com_pos, np.zeros(3))
        zmp_error = zmp_ref - actual_zmp

        # Map ZMP error to ankle torques
        ankle_torque = np.array([zmp_error[1] * 50, -zmp_error[0] * 50, 0])  # Simplified

        return ankle_torque

class WalkingPatternGenerator:
    """Generate complete walking patterns"""

    def __init__(self, gait_controller: GaitController):
        self.gait_controller = gait_controller

    def generate_omni_directional_walk(self, speed: float, direction: float,
                                     turn_rate: float) -> List[np.ndarray]:
        """Generate omni-directional walking pattern"""
        # direction in radians (0 = forward, π/2 = left, etc.)
        x_speed = speed * np.cos(direction)
        y_speed = speed * np.sin(direction)

        # Generate pattern for specified parameters
        steps = []
        dt = 0.01

        for i in range(int(5 / dt)):  # 5 seconds of walking
            state = self.gait_controller.update_walking_state(dt, x_speed)
            steps.append(state.com_pos.copy())

        return steps

    def generate_stair_climbing_pattern(self, step_height: float = 0.15) -> List[np.ndarray]:
        """Generate pattern for stair climbing"""
        # Simplified stair climbing pattern
        pattern = []

        # For each step, generate lift, place, and shift pattern
        for step in range(5):  # 5 steps up
            # Lift CoM
            for i in range(10):
                t = i / 10
                height = step * step_height + t * step_height
                pattern.append(np.array([step * 0.3, 0, height + 0.8]))

            # Shift weight
            for i in range(20):
                pattern.append(np.array([(step + 0.5) * 0.3, 0, (step + 1) * step_height + 0.8]))

        return pattern

def simulate_walking_trial():
    """Simulate a walking trial with the controllers"""
    print("Walking Simulation Trial")
    print("=" * 30)

    # Initialize controllers
    gait_controller = GaitController(step_length=0.3, step_width=0.2, step_time=0.8, height=0.8)
    balance_controller = BalanceController(gait_controller.inverted_pendulum)
    pattern_generator = WalkingPatternGenerator(gait_controller)

    print(f"1. Controller initialized with:")
    print(f"   - Step length: {gait_controller.step_length}m")
    print(f"   - Step width: {gait_controller.step_width}m")
    print(f"   - Step time: {gait_controller.step_time}s")
    print(f"   - CoM height: {gait_controller.inverted_pendulum.height}m")

    # Start walking
    gait_controller.start_walking(walking_speed=0.1)
    print(f"\n2. Walking started with speed: 0.1 m/s")

    # Simulate walking for a few seconds
    dt = 0.01  # 100Hz control
    simulation_time = 3.0  # 3 seconds
    steps = int(simulation_time / dt)

    com_positions = []
    zmp_positions = []
    capture_points = []

    for i in range(steps):
        state = gait_controller.update_walking_state(dt)

        # Store trajectory data
        com_positions.append(state.com_pos.copy())
        zmp_positions.append(state.zmp_pos.copy())
        capture_points.append(state.capture_point.copy())

    print(f"3. Simulated {steps} control cycles ({simulation_time}s)")
    print(f"   Final CoM position: [{state.com_pos[0]:.3f}, {state.com_pos[1]:.3f}, {state.com_pos[2]:.3f}]")
    print(f"   Final ZMP: [{state.zmp_pos[0]:.3f}, {state.zmp_pos[1]:.3f}]")
    print(f"   Final Capture Point: [{state.capture_point[0]:.3f}, {state.capture_point[1]:.3f}]")

    # Analyze stability
    com_path = np.array(com_positions)
    zmp_path = np.array(zmp_positions)

    # Calculate average CoM height maintenance
    height_variation = np.std(com_path[:, 2])
    print(f"4. Stability metrics:")
    print(f"   - CoM height std: {height_variation:.3f}m")
    print(f"   - Average forward progress: {com_path[-1, 0]:.3f}m")
    print(f"   - Average lateral deviation: {np.mean(np.abs(com_path[:, 1])):.3f}m")

    # Calculate ZMP tracking error
    zmp_error = np.mean(np.linalg.norm(zmp_path - com_path[:, :2] + np.array([0.02, 0]), axis=1))
    print(f"   - Average ZMP tracking error: {zmp_error:.3f}m")

    # Generate omni-directional walk
    print(f"\n5. Omni-directional walking:")
    directions = [0, np.pi/4, np.pi/2]  # Forward, diagonal, sideways
    for direction in directions:
        x_speed = 0.1 * np.cos(direction)
        y_speed = 0.1 * np.sin(direction)
        print(f"   - Direction {direction:.2f} rad: [{x_speed:.3f}, {y_speed:.3f}] m/s")

    # Test balance controller
    print(f"\n6. Balance control test:")
    test_zmp_ref = np.array([0.1, 0.05])
    balance_correction = balance_controller.compute_balance_correction(
        test_zmp_ref, state.zmp_pos
    )
    ankle_torque = balance_controller.compute_ankle_torque(
        state.com_pos, state.com_vel, test_zmp_ref
    )
    print(f"   - ZMP reference: [{test_zmp_ref[0]:.3f}, {test_zmp_ref[1]:.3f}]")
    print(f"   - Balance correction: [{balance_correction[0]:.3f}, {balance_correction[1]:.3f}]")
    print(f"   - Ankle torque: [{ankle_torque[0]:.3f}, {ankle_torque[1]:.3f}, {ankle_torque[2]:.3f}]")

    # Generate stair climbing pattern
    print(f"\n7. Stair climbing pattern:")
    stair_pattern = pattern_generator.generate_stair_climbing_pattern()
    print(f"   - Generated {len(stair_pattern)} steps for stair climbing")
    print(f"   - Final height: {stair_pattern[-1][2]:.3f}m")

    return com_positions, zmp_positions, capture_points

def analyze_locomotion_performance(com_positions: List[np.ndarray],
                                 zmp_positions: List[np.ndarray]) -> dict:
    """Analyze locomotion performance metrics"""
    com_path = np.array(com_positions)
    zmp_path = np.array(zmp_positions)

    metrics = {}

    # Stability metrics
    metrics['avg_height'] = np.mean(com_path[:, 2])
    metrics['height_std'] = np.std(com_path[:, 2])
    metrics['forward_progress'] = com_path[-1, 0] - com_path[0, 0]
    metrics['avg_speed'] = metrics['forward_progress'] / len(com_path) * 100  # Assuming 100Hz

    # Balance metrics
    # Calculate ZMP stability margin (distance from support polygon boundary)
    # For simplicity, assume square support polygon around foot
    support_size = 0.1  # 10cm support area
    zmp_stability = []
    for i, (com_pos, zmp_pos) in enumerate(zip(com_path, zmp_path)):
        # Distance from ZMP to CoM projection (simplified stability measure)
        stability_margin = support_size - np.linalg.norm(zmp_pos - com_pos[:2])
        zmp_stability.append(max(0, stability_margin))  # Only positive margins

    metrics['avg_stability_margin'] = np.mean(zmp_stability)
    metrics['min_stability_margin'] = min(zmp_stability) if zmp_stability else 0

    # Efficiency metrics
    metrics['path_efficiency'] = metrics['forward_progress'] / np.sum(
        np.linalg.norm(np.diff(com_path, axis=0), axis=1)
    ) if len(com_path) > 1 else 0

    # Smoothness metrics
    velocities = np.linalg.norm(np.diff(com_path, axis=0), axis=1)
    metrics['speed_variation'] = np.std(velocities) / (np.mean(velocities) + 1e-6)

    return metrics

if __name__ == "__main__":
    # Run the simulation
    com_positions, zmp_positions, capture_points = simulate_walking_trial()

    # Analyze performance
    performance_metrics = analyze_locomotion_performance(com_positions, zmp_positions)

    print(f"\n8. Performance Analysis:")
    for metric, value in performance_metrics.items():
        print(f"   - {metric}: {value:.3f}")

    print(f"\nLocomotion and Gait Control - Chapter 8 Complete!")
```

## Exercises

1. Implement a simple ZMP controller for a 2D inverted pendulum and simulate walking.

2. Design a gait pattern that allows a humanoid robot to walk up a ramp with a 15-degree incline.

3. Create a balance controller that uses both ankle and hip strategies based on disturbance magnitude.

## Summary

This chapter provided a comprehensive overview of locomotion and gait control for humanoid robots, covering the fundamental principles of bipedal walking, ZMP control, capture point dynamics, and practical implementation strategies. We explored mathematical models, control algorithms, and the integration of balance control with locomotion. The concepts and code examples presented will help in developing robust walking controllers for humanoid robots that can navigate real-world environments safely and efficiently.