---
sidebar_position: 3
---

# Chapter 9: Balance and Stability Control

## Summary

This chapter examines the critical systems that enable humanoid robots to maintain stability and balance during static and dynamic conditions. We'll explore feedback and feedforward control strategies, disturbance rejection techniques, and recovery mechanisms that allow robots to maintain equilibrium in the face of internal and external disturbances. Understanding balance control is fundamental to creating safe and capable humanoid robots.

## Learning Outcomes

By the end of this chapter, you will be able to:
- Understand the principles of static and dynamic balance control
- Analyze different feedback and feedforward control strategies
- Design controllers for disturbance rejection and recovery
- Implement multi-level control hierarchies for balance
- Evaluate balance performance and stability margins

## Key Concepts

- **Static Balance**: Stability without motion
- **Dynamic Balance**: Stability maintained through active control during motion
- **Feedback Control**: Control based on measured system state
- **Feedforward Control**: Control based on predicted or desired behavior
- **Disturbance Rejection**: Ability to maintain balance despite external forces
- **Recovery Strategies**: Methods to regain balance after disturbances
- **Stability Margins**: Quantitative measures of balance robustness
- **Control Hierarchies**: Multi-level control structures for balance

## Introduction to Balance Control

Balance control in humanoid robots is a complex task that involves maintaining the center of mass within the support polygon while executing tasks. Unlike static structures, humanoid robots must actively control their balance through coordinated joint movements and sophisticated control algorithms.

### Balance Fundamentals

**Support Polygon**: The convex hull of all contact points with the ground
**Center of Mass (CoM)**: The point where the total mass can be considered concentrated
**Zero Moment Point (ZMP)**: The point where the moment of ground reaction force equals zero
**Capture Point**: The point where the robot must step to come to a complete stop

### Balance Control Challenges

**High-Dimensional Systems**: Many degrees of freedom to coordinate
**Underactuation**: Fewer actuators than degrees of freedom
**Real-time Requirements**: Control must be computed and executed rapidly
**Uncertainty**: Modeling errors, sensor noise, and environmental changes
**Safety**: Maintaining balance is critical for preventing falls

## Feedback Control Systems

### Proportional-Integral-Derivative (PID) Control

PID controllers are fundamental in balance control:

```
u(t) = K_p * e(t) + K_i * ∫e(t)dt + K_d * de(t)/dt
```

Where:
- u(t) = control output
- e(t) = error (desired - actual)
- K_p, K_i, K_d = controller gains

**Position-based PID**: Controls joint positions relative to desired values
**Force-based PID**: Controls contact forces relative to desired values
**ZMP-based PID**: Controls ZMP position relative to desired values

### State Feedback Control

State feedback uses the full system state for control:

```
u = -K * x + r
```

Where:
- u = control input
- K = feedback gain matrix
- x = state vector
- r = reference input

**Linear Quadratic Regulator (LQR)**: Optimizes control for linear systems
```
K = R⁻¹ * Bᵀ * P
```

Where P is the solution to the algebraic Riccati equation.

### Adaptive Control

Adaptive controllers adjust parameters based on system behavior:

**Model Reference Adaptive Control (MRAC)**:
```
θ̇ = -Γ * φ * e
```

Where:
- θ = parameter vector
- Γ = adaptation gain
- φ = regressor vector
- e = tracking error

**Self-Tuning Regulators**: Adjust controller parameters based on system identification.

## Feedforward Control Strategies

### Preview Control

Preview control uses future reference trajectory to compute current control:

```
u(k) = -Kx(k) - K_f * Σ(i=0 to N-1) G_f(i) * r(k+i)
```

Where:
- N = preview horizon
- r = reference trajectory
- K_f, G_f = preview control gains

### Feedforward Compensation

**Gravity Compensation**: Counteract gravitational forces
```
τ_gravity = G(q)
```

**Coriolis Compensation**: Counteract velocity-dependent forces
```
τ_coriolis = C(q, q̇)q̇
```

**Disturbance Feedforward**: Predict and counteract known disturbances
```
τ_disturbance = F_external
```

### Trajectory-Based Feedforward

**Computed Torque Control**: Linearize system dynamics
```
τ = M(q) * (q̈_d + K_v * (q̇_d - q̇) + K_p * (q_d - q))
```

## Disturbance Rejection

### Types of Disturbances

**Impulsive Disturbances**: Sudden impacts or pushes
**Persistent Disturbances**: Continuous external forces
**Internal Disturbances**: Actuator failures, sensor noise
**Environmental Disturbances**: Wind, uneven terrain, slippery surfaces

### Disturbance Observer

Estimate disturbances and compensate for them:

```
d̂ = L(s) * (y - ŷ)
```

Where:
- d̂ = estimated disturbance
- L(s) = observer gain
- y = measured output
- ŷ = estimated output

### Robust Control

Design controllers that maintain performance despite uncertainties:

**H-infinity Control**: Minimize worst-case performance
```
||T_{wd}||_∞ < γ
```

Where T_{wd} is the transfer function from disturbance to output.

**Mu-Synthesis**: Handle structured uncertainties
```
μ_Δ(T) < 1
```

## Recovery from Perturbations

### Balance Recovery Strategies

**Ankle Strategy**: Use ankle torques for small disturbances
- Effective for disturbances within ankle torque limits
- Energy efficient
- Limited range of effectiveness

**Hip Strategy**: Use hip torques for medium disturbances
- Greater range of effectiveness
- Higher energy consumption
- Coordination with upper body required

**Stepping Strategy**: Take a step to expand support base
- Most effective for large disturbances
- High energy consumption
- Requires planning and coordination

**Suspension Strategy**: Move CoM to maintain balance
- For dynamic situations
- Requires sophisticated control
- Effective for specific scenarios

### Recovery Phase Classification

**Stability Boundary**: Threshold where different strategies become necessary
**Recovery Strategy Selection**: Choose strategy based on disturbance magnitude
**CoM State**: Position and velocity of center of mass
**Support State**: Current support polygon configuration

### Multi-Step Recovery

For large disturbances, multiple recovery actions may be needed:

1. **Initial Response**: Immediate reflexive action
2. **Strategy Selection**: Choose appropriate recovery strategy
3. **Execution**: Implement recovery action
4. **Verification**: Confirm balance recovery
5. **Return to Normal**: Resume normal behavior

## Multi-Level Control Hierarchy

### High-Level Balance Control

**Balance Planning**: Determine long-term balance strategy
- Step planning for stability
- CoM trajectory planning
- Gait selection

**Task Prioritization**: Balance competing objectives
- Balance vs. task execution
- Safety vs. efficiency
- Stability vs. performance

### Mid-Level Balance Control

**ZMP/Capture Point Control**: Track desired stability points
- Reference trajectory generation
- Feedback control design
- Stability margin maintenance

**Whole-Body Control**: Coordinate all degrees of freedom
- Inverse kinematics
- Optimization-based control
- Constraint handling

### Low-Level Balance Control

**Joint-Level Control**: Direct actuator commands
- PID control
- Current control
- Safety limits

**Sensor Processing**: Filter and interpret sensor data
- IMU processing
- Force sensing
- Vision feedback

## Technical Depth: Mathematical Foundations

### Linear Inverted Pendulum Model

For balance analysis:

```
ẍ_com = ω² * (x_com - x_zmp)
ÿ_com = ω² * (y_com - y_zmp)
```

Where ω² = g/h (natural frequency).

### State-Space Representation

Balance control as a state-space system:

```
ẋ = Ax + Bu + Bd
y = Cx + Du
```

Where:
- x = state vector [CoM_pos, CoM_vel]
- u = control input
- d = disturbance
- y = measured output

### Stability Analysis

**Lyapunov Stability**: V(x) > 0 and V̇(x) < 0 for stability
**BIBO Stability**: Bounded input produces bounded output
**Routh-Hurwitz Criterion**: Determine stability from characteristic polynomial

### Optimal Control Formulation

Minimize balance control cost:

```
J = ∫[xᵀQx + uᵀRu] dt
```

Subject to system dynamics:
```
ẋ = Ax + Bu
```

Solution: u = -R⁻¹BᵀPx, where P is Riccati solution.

## Practical Applications

### Standing Balance

**Quiet Stance**: Minimal control for stable standing
- Ankle strategy dominance
- Small sway patterns
- Sensorimotor integration

**Challenged Stance**: Balance under constraints
- Narrow support
- Unstable surfaces
- Visual deprivation

### Walking Balance

**Gait Integration**: Balance control during locomotion
- ZMP trajectory tracking
- Step timing adjustment
- Swing foot control

### Manipulation Balance

**Dual-Task Performance**: Balance while manipulating
- CoM compensation
- Anticipatory postural adjustments
- Load handling

### Recovery Scenarios

**Push Recovery**: Response to external forces
- Multi-strategy approach
- Real-time decision making
- Safety prioritization

**Trip Recovery**: Response to foot obstacles
- Swing leg adjustment
- CoM control
- Fall prevention

## Challenges

### Control Complexity

Balancing multiple objectives while maintaining stability.

### Real-time Performance

Computing complex control laws within tight timing constraints.

### Modeling Uncertainty

Handling errors in robot models and environmental assumptions.

### Safety vs. Performance

Maintaining safety margins without overly restricting capabilities.

## Figure List

1. **Figure 9.1**: Balance control hierarchy diagram
2. **Figure 9.2**: Feedback vs. feedforward control comparison
3. **Figure 9.3**: Disturbance rejection mechanisms
4. **Figure 9.4**: Recovery strategy selection tree
5. **Figure 9.5**: Stability margin visualization

## Code Example: Balance Control Implementation

```python
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve
import math

@dataclass
class BalanceState:
    """Current state for balance control"""
    com_pos: np.ndarray      # Center of mass position [x, y, z]
    com_vel: np.ndarray      # Center of mass velocity [vx, vy, vz]
    com_acc: np.ndarray      # Center of mass acceleration
    zmp_pos: np.ndarray      # Zero Moment Point [x, y]
    capture_point: np.ndarray # Capture Point [x, y]
    support_polygon: np.ndarray  # Support polygon vertices
    angular_momentum: np.ndarray  # Angular momentum [Lx, Ly, Lz]
    timestamp: float

@dataclass
class BalanceControlParams:
    """Parameters for balance control"""
    # PID gains
    kp_com: float = 10.0
    ki_com: float = 1.0
    kd_com: float = 2.0

    # ZMP control
    kp_zmp: float = 15.0
    kd_zmp: float = 5.0

    # Capture point
    capture_threshold: float = 0.15
    max_step_size: float = 0.3

    # Safety limits
    max_torque: float = 50.0
    max_angular_velocity: float = 1.0

class BalanceController:
    """Comprehensive balance controller for humanoid robots"""

    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # Initialize balance state
        self.state = BalanceState(
            com_pos=np.array([0.0, 0.0, com_height]),
            com_vel=np.array([0.0, 0.0, 0.0]),
            com_acc=np.array([0.0, 0.0, 0.0]),
            zmp_pos=np.array([0.0, 0.0]),
            capture_point=np.array([0.0, 0.0]),
            support_polygon=np.array([[0.1, 0.05], [0.1, -0.05], [-0.1, -0.05], [-0.1, 0.05]]),
            angular_momentum=np.array([0.0, 0.0, 0.0]),
            timestamp=0.0
        )

        # Control parameters
        self.params = BalanceControlParams()

        # PID control states
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)

        # Balance strategy state
        self.current_strategy = "ankle"
        self.recovery_active = False
        self.recovery_start_time = 0.0

        # Support polygon state
        self.left_foot_pos = np.array([0.0, 0.1, 0.0])
        self.right_foot_pos = np.array([0.0, -0.1, 0.0])
        self.support_foot = "both"  # "left", "right", or "both"

    def compute_zmp(self, com_pos: np.ndarray, com_acc: np.ndarray) -> np.ndarray:
        """Compute ZMP from CoM state"""
        # ZMP = CoM - (h/g) * CoM_acc
        zmp = com_pos[:2] - (self.com_height / self.gravity) * com_acc[:2]
        return zmp

    def compute_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """Compute capture point from CoM state"""
        # Capture Point = CoM + CoM_vel / omega
        capture_point = com_pos[:2] + com_vel[:2] / self.omega
        return capture_point

    def update_support_polygon(self):
        """Update support polygon based on foot positions"""
        if self.support_foot == "left":
            # Single support polygon for left foot
            self.state.support_polygon = np.array([
                self.left_foot_pos[:2] + [0.1, 0.05],
                self.left_foot_pos[:2] + [0.1, -0.05],
                self.left_foot_pos[:2] + [-0.1, -0.05],
                self.left_foot_pos[:2] + [-0.1, 0.05]
            ])
        elif self.support_foot == "right":
            # Single support polygon for right foot
            self.state.support_polygon = np.array([
                self.right_foot_pos[:2] + [0.1, 0.05],
                self.right_foot_pos[:2] + [0.1, -0.05],
                self.right_foot_pos[:2] + [-0.1, -0.05],
                self.right_foot_pos[:2] + [-0.1, 0.05]
            ])
        else:  # both feet
            # Convex hull of both feet
            left_poly = np.array([
                self.left_foot_pos[:2] + [0.1, 0.05],
                self.left_foot_pos[:2] + [0.1, -0.05],
                self.left_foot_pos[:2] + [-0.1, -0.05],
                self.left_foot_pos[:2] + [-0.1, 0.05]
            ])
            right_poly = np.array([
                self.right_foot_pos[:2] + [0.1, 0.05],
                self.right_foot_pos[:2] + [0.1, -0.05],
                self.right_foot_pos[:2] + [-0.1, -0.05],
                self.right_foot_pos[:2] + [-0.1, 0.05]
            ])
            # Simplified: create larger polygon encompassing both
            all_points = np.vstack([left_poly, right_poly])
            self.state.support_polygon = all_points

    def point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def select_balance_strategy(self) -> str:
        """Select appropriate balance strategy based on current state"""
        # Calculate distances to support polygon boundary
        zmp_distance = self.distance_to_polygon(self.state.zmp_pos, self.state.support_polygon)
        capture_distance = self.distance_to_polygon(self.state.capture_point, self.state.support_polygon)

        # Strategy selection logic
        if capture_distance > self.params.capture_threshold:
            return "stepping"
        elif capture_distance > 0.08:
            return "hip"
        elif zmp_distance > 0.05:
            return "ankle"
        else:
            return "ankle"  # Default for small disturbances

    def distance_to_polygon(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """Calculate minimum distance from point to polygon boundary"""
        min_dist = float('inf')

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            # Distance from point to line segment
            dist = self.distance_point_to_segment(point, p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist

    def distance_point_to_segment(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """Calculate distance from point to line segment"""
        # Vector from segment start to point
        v = point - seg_start
        # Vector along segment
        s = seg_end - seg_start

        # Project v onto s
        proj_len = np.dot(v, s) / np.dot(s, s)

        if proj_len < 0:
            # Closest point is segment start
            return np.linalg.norm(point - seg_start)
        elif proj_len > 1:
            # Closest point is segment end
            return np.linalg.norm(point - seg_end)
        else:
            # Closest point is along segment
            closest = seg_start + proj_len * s
            return np.linalg.norm(point - closest)

    def compute_ankle_strategy(self, dt: float) -> np.ndarray:
        """Compute control for ankle strategy"""
        # Calculate error between CoM and desired position
        desired_com = np.array([0.0, 0.0, self.com_height])
        pos_error = desired_com[:2] - self.state.com_pos[:2]

        # PID control for position
        self.integral_error += pos_error * dt
        derivative_error = (pos_error - self.previous_error) / dt if dt > 0 else np.zeros(2)

        control_output = (self.params.kp_com * pos_error +
                         self.params.ki_com * self.integral_error +
                         self.params.kd_com * derivative_error)

        self.previous_error = pos_error.copy()

        return control_output

    def compute_hip_strategy(self, dt: float) -> np.ndarray:
        """Compute control for hip strategy"""
        # Use ZMP error for hip control
        desired_zmp = np.array([0.0, 0.0])  # Center of support polygon
        zmp_error = desired_zmp - self.state.zmp_pos

        # PD control for ZMP
        derivative_zmp = (zmp_error - self.previous_error) / dt if dt > 0 else np.zeros(2)

        control_output = (self.params.kp_zmp * zmp_error +
                         self.params.kd_zmp * derivative_zmp)

        # Update previous error for ZMP
        self.previous_error = zmp_error.copy()

        return control_output

    def compute_stepping_strategy(self) -> np.ndarray:
        """Compute foot placement for stepping strategy"""
        # Place foot at or beyond capture point
        step_target = self.state.capture_point.copy()

        # Ensure step is within reasonable limits
        step_distance = np.linalg.norm(step_target)
        if step_distance > self.params.max_step_size:
            step_target = (step_target / step_distance) * self.params.max_step_size

        return step_target

    def update_balance_state(self, com_pos: np.ndarray, com_vel: np.ndarray,
                           com_acc: np.ndarray, dt: float) -> BalanceState:
        """Update balance state and compute control"""
        # Update CoM state
        self.state.com_pos = com_pos
        self.state.com_vel = com_vel
        self.state.com_acc = com_acc

        # Compute ZMP and capture point
        self.state.zmp_pos = self.compute_zmp(com_pos, com_acc)
        self.state.capture_point = self.compute_capture_point(com_pos, com_vel)

        # Update support polygon
        self.update_support_polygon()

        # Select balance strategy
        self.current_strategy = self.select_balance_strategy()

        # Compute control based on strategy
        if self.current_strategy == "stepping":
            control_output = self.compute_stepping_strategy()
            self.recovery_active = True
            self.recovery_start_time = self.state.timestamp
        elif self.current_strategy == "hip":
            control_output = self.compute_hip_strategy(dt)
        elif self.current_strategy == "ankle":
            control_output = self.compute_ankle_strategy(dt)
        else:
            control_output = np.zeros(2)  # No control if stable

        # Update timestamp
        self.state.timestamp += dt

        # Check if recovery is complete (simplified)
        if self.recovery_active:
            zmp_in_support = self.point_in_polygon(self.state.zmp_pos, self.state.support_polygon)
            if zmp_in_support and (self.state.timestamp - self.recovery_start_time) > 0.5:
                self.recovery_active = False

        return self.state

    def compute_joint_torques(self, control_output: np.ndarray) -> np.ndarray:
        """Convert balance control to joint torques (simplified mapping)"""
        # This would typically involve inverse dynamics and whole-body control
        # For this example, we'll create a simplified mapping

        # Map CoM control to hip and ankle torques
        torques = np.zeros(20)  # 20 joint torques for example

        # Ankle torques (simplified)
        torques[0] = control_output[0] * 20  # Ankle pitch
        torques[1] = control_output[1] * 10  # Ankle roll

        # Hip torques (simplified)
        torques[6] = -control_output[0] * 15  # Hip pitch
        torques[7] = -control_output[1] * 10  # Hip roll

        # Apply safety limits
        torques = np.clip(torques, -self.params.max_torque, self.params.max_torque)

        return torques

class DisturbanceObserver:
    """Observer for estimating external disturbances"""

    def __init__(self, model_uncertainty: float = 0.1):
        self.model_uncertainty = model_uncertainty
        self.estimated_disturbance = np.zeros(2)
        self.disturbance_error = np.zeros(2)
        self.gain = 1.0  # Observer gain

    def update(self, measured_output: np.ndarray, predicted_output: np.ndarray, dt: float) -> np.ndarray:
        """Update disturbance estimate"""
        # Calculate prediction error
        error = measured_output - predicted_output

        # Update disturbance estimate
        self.disturbance_error += error * dt
        self.estimated_disturbance = self.gain * error + 0.1 * self.disturbance_error

        # Limit disturbance estimate
        self.estimated_disturbance = np.clip(self.estimated_disturbance, -100, 100)

        return self.estimated_disturbance

class BalancePerformanceAnalyzer:
    """Analyze balance control performance"""

    def __init__(self):
        self.metrics = {
            'stability_margin': [],
            'recovery_time': [],
            'control_effort': [],
            'disturbance_rejection': []
        }

    def calculate_stability_margin(self, zmp_pos: np.ndarray, support_polygon: np.ndarray) -> float:
        """Calculate minimum distance from ZMP to support polygon boundary"""
        return min(np.linalg.norm(zmp_pos - vertex) for vertex in support_polygon)

    def calculate_control_effort(self, torques: np.ndarray) -> float:
        """Calculate control effort as sum of absolute torques"""
        return np.sum(np.abs(torques))

    def calculate_recovery_time(self, recovery_start: float, recovery_end: float) -> float:
        """Calculate time to recover from disturbance"""
        return recovery_end - recovery_start

    def add_sample(self, zmp_pos: np.ndarray, support_polygon: np.ndarray,
                   torques: np.ndarray, timestamp: float):
        """Add a sample for performance analysis"""
        stability_margin = self.calculate_stability_margin(zmp_pos, support_polygon)
        control_effort = self.calculate_control_effort(torques)

        self.metrics['stability_margin'].append(stability_margin)
        self.metrics['control_effort'].append(control_effort)

    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics"""
        if not self.metrics['stability_margin']:
            return {'error': 'No data collected'}

        return {
            'avg_stability_margin': np.mean(self.metrics['stability_margin']),
            'min_stability_margin': min(self.metrics['stability_margin']),
            'avg_control_effort': np.mean(self.metrics['control_effort']),
            'max_control_effort': max(self.metrics['control_effort']),
            'stability_std': np.std(self.metrics['stability_margin'])
        }

def simulate_balance_control():
    """Simulate balance control with various scenarios"""
    print("Balance Control Simulation")
    print("=" * 35)

    # Initialize controllers
    balance_controller = BalanceController(com_height=0.8)
    disturbance_observer = DisturbanceObserver()
    performance_analyzer = BalancePerformanceAnalyzer()

    print(f"1. Controllers initialized:")
    print(f"   - CoM height: {balance_controller.com_height}m")
    print(f"   - Gravity: {balance_controller.gravity} m/s²")
    print(f"   - Natural frequency: {balance_controller.omega:.3f} rad/s")

    # Scenario 1: Quiet standing
    print(f"\n2. Scenario 1: Quiet Standing")
    dt = 0.01  # 100Hz control
    simulation_time = 2.0  # 2 seconds
    steps = int(simulation_time / dt)

    for i in range(steps):
        # Nominal CoM state (with small variations)
        com_pos = np.array([0.01 * np.sin(i * dt * 2), 0.005 * np.cos(i * dt * 3), balance_controller.com_height])
        com_vel = np.array([0.01 * 2 * np.cos(i * dt * 2), -0.005 * 3 * np.sin(i * dt * 3), 0])
        com_acc = np.array([-0.01 * 4 * np.sin(i * dt * 2), -0.005 * 9 * np.cos(i * dt * 3), 0])

        # Update balance state
        state = balance_controller.update_balance_state(com_pos, com_vel, com_acc, dt)

        # Compute control
        if balance_controller.current_strategy == "stepping":
            control_output = balance_controller.compute_stepping_strategy()
        elif balance_controller.current_strategy == "hip":
            control_output = balance_controller.compute_hip_strategy(dt)
        else:
            control_output = balance_controller.compute_ankle_strategy(dt)

        # Convert to joint torques
        torques = balance_controller.compute_joint_torques(control_output)

        # Analyze performance
        performance_analyzer.add_sample(state.zmp_pos, state.support_polygon, torques, state.timestamp)

    print(f"   - Simulated {steps} control cycles")
    print(f"   - Final strategy: {balance_controller.current_strategy}")
    print(f"   - Final ZMP: [{state.zmp_pos[0]:.3f}, {state.zmp_pos[1]:.3f}]")
    print(f"   - Final capture point: [{state.capture_point[0]:.3f}, {state.capture_point[1]:.3f}]")

    # Scenario 2: Disturbance rejection
    print(f"\n3. Scenario 2: Disturbance Rejection")

    # Apply impulse disturbance
    impulse_time = 0.5  # Apply at 0.5s
    impulse_magnitude = np.array([5.0, 2.0])  # x and y impulse

    for i in range(steps):
        t = i * dt

        # Nominal state with disturbance
        com_pos = np.array([0.01 * np.sin(i * dt * 2), 0.005 * np.cos(i * dt * 3), balance_controller.com_height])

        # Add impulse effect
        if abs(t - impulse_time) < dt:
            com_vel = np.array([0.5, 0.2, 0])  # Impulse velocity
        else:
            com_vel = np.array([0.01 * 2 * np.cos(i * dt * 2), -0.005 * 3 * np.sin(i * dt * 3), 0])

        com_acc = np.array([-0.01 * 4 * np.sin(i * dt * 2), -0.005 * 9 * np.cos(i * dt * 3), 0])

        # Update balance state
        state = balance_controller.update_balance_state(com_pos, com_vel, com_acc, dt)

        # Compute control
        if balance_controller.current_strategy == "stepping":
            control_output = balance_controller.compute_stepping_strategy()
        elif balance_controller.current_strategy == "hip":
            control_output = balance_controller.compute_hip_strategy(dt)
        else:
            control_output = balance_controller.compute_ankle_strategy(dt)

        # Convert to joint torques
        torques = balance_controller.compute_joint_torques(control_output)

        # Analyze performance
        performance_analyzer.add_sample(state.zmp_pos, state.support_polygon, torques, state.timestamp)

    print(f"   - Applied impulse disturbance at t={impulse_time}s")
    print(f"   - Disturbance magnitude: [{impulse_magnitude[0]:.1f}, {impulse_magnitude[1]:.1f}] N")
    print(f"   - Recovery strategy: {balance_controller.current_strategy}")

    # Scenario 3: Step adjustment
    print(f"\n4. Scenario 3: Step Adjustment")

    # Simulate capture point outside support polygon
    balance_controller.state.capture_point = np.array([0.2, 0.0])  # Outside threshold

    # Compute step placement
    step_target = balance_controller.compute_stepping_strategy()
    print(f"   - Capture point: [{balance_controller.state.capture_point[0]:.3f}, {balance_controller.state.capture_point[1]:.3f}]")
    print(f"   - Step target: [{step_target[0]:.3f}, {step_target[1]:.3f}]")
    print(f"   - Required step size: {np.linalg.norm(step_target):.3f}m")

    # Analyze performance
    performance_summary = performance_analyzer.get_performance_summary()
    print(f"\n5. Performance Analysis:")
    for metric, value in performance_summary.items():
        if isinstance(value, float):
            print(f"   - {metric}: {value:.3f}")

    # Test different balance strategies
    print(f"\n6. Balance Strategy Testing:")

    # Test ankle strategy effectiveness
    zmp_near_center = np.array([0.01, 0.01])
    zmp_distance = balance_controller.distance_to_polygon(zmp_near_center, balance_controller.state.support_polygon)
    print(f"   - ZMP near center distance: {zmp_distance:.3f}m -> Strategy: Ankle")

    # Test hip strategy effectiveness
    zmp_far_center = np.array([0.08, 0.06])
    zmp_distance = balance_controller.distance_to_polygon(zmp_far_center, balance_controller.state.support_polygon)
    print(f"   - ZMP far center distance: {zmp_distance:.3f}m -> Strategy: Hip")

    # Test stepping strategy effectiveness
    capture_far = np.array([0.2, 0.1])
    capture_distance = balance_controller.distance_to_polygon(capture_far, balance_controller.state.support_polygon)
    print(f"   - Capture point distance: {capture_distance:.3f}m -> Strategy: Stepping")

    return balance_controller, performance_summary

def analyze_balance_stability(balance_controller: BalanceController) -> Dict:
    """Analyze balance stability margins and performance"""
    analysis = {
        'stability_margins': {},
        'recovery_capabilities': {},
        'control_performance': {}
    }

    # Test stability margins
    test_points = [
        np.array([0.0, 0.0]),      # Center
        np.array([0.05, 0.0]),     # Near boundary
        np.array([0.1, 0.05]),     # At boundary
        np.array([0.15, 0.05])     # Outside boundary
    ]

    distances = []
    for point in test_points:
        dist = balance_controller.distance_to_polygon(point, balance_controller.state.support_polygon)
        distances.append(dist)

    analysis['stability_margins'] = {
        'center_distance': distances[0],
        'near_boundary_distance': distances[1],
        'at_boundary_distance': distances[2],
        'outside_boundary_distance': distances[3]
    }

    # Test recovery capabilities
    analysis['recovery_capabilities'] = {
        'capture_threshold': balance_controller.params.capture_threshold,
        'max_step_size': balance_controller.params.max_step_size,
        'strategy_selection_range': 'ankle: <0.05m, hip: 0.05-0.08m, stepping: >0.08m'
    }

    # Test control performance
    analysis['control_performance'] = {
        'max_torque_limit': balance_controller.params.max_torque,
        'current_strategy': balance_controller.current_strategy,
        'recovery_active': balance_controller.recovery_active
    }

    return analysis

if __name__ == "__main__":
    # Run the simulation
    balance_controller, performance_summary = simulate_balance_control()

    # Perform detailed stability analysis
    stability_analysis = analyze_balance_stability(balance_controller)

    print(f"\n7. Detailed Stability Analysis:")
    for category, metrics in stability_analysis.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"     - {metric.replace('_', ' ')}: {value}")

    print(f"\n8. Control Parameters:")
    params = balance_controller.params
    print(f"   - Ankle strategy gains: Kp={params.kp_com}, Ki={params.ki_com}, Kd={params.kd_com}")
    print(f"   - ZMP control gains: Kp={params.kp_zmp}, Kd={params.kd_zmp}")
    print(f"   - Capture threshold: {params.capture_threshold}m")
    print(f"   - Max step size: {params.max_step_size}m")

    print(f"\nBalance and Stability Control - Chapter 9 Complete!")
```

## Exercises

1. Implement a balance controller that switches between ankle and hip strategies based on ZMP position.

2. Design a disturbance observer that can estimate external forces applied to a humanoid robot.

3. Create a multi-level balance control system with high-level planning and low-level joint control.

## Summary

This chapter provided a comprehensive overview of balance and stability control for humanoid robots, covering feedback and feedforward control strategies, disturbance rejection techniques, and recovery mechanisms. We explored mathematical models, control algorithms, and multi-level control hierarchies essential for maintaining robot stability. The concepts and implementations presented will help in developing robust balance control systems for humanoid robots that can maintain equilibrium in various conditions and recover from disturbances safely.