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

- **Support Polygon**: The convex hull of all contact points with the ground  
- **Center of Mass (CoM)**: The point where the total mass can be considered concentrated  
- **Zero Moment Point (ZMP)**: The point where the moment of ground reaction force equals zero  
- **Capture Point**: The point where the robot must step to stop  

### Balance Control Challenges

- **High-Dimensional Systems**  
- **Underactuation**  
- **Real-time Requirements**  
- **Uncertainty**  
- **Safety**  

## Feedback Control Systems

### Proportional-Integral-Derivative (PID) Control

PID controllers are fundamental in balance control:

u(t) = K_p * e(t) + K_i * ∫e(t)dt + K_d * de(t)/dt

markdown
Copy code

Where:
- u(t) = control output
- e(t) = error (desired - actual)
- K_p, K_i, K_d = controller gains

**Position-based PID**: Controls joint positions relative to desired values  
**Force-based PID**: Controls contact forces relative to desired values  
**ZMP-based PID**: Controls ZMP position relative to desired values  

### State Feedback Control

State feedback uses the full system state for control:

u = -K * x + r

markdown
Copy code

Where:
- u = control input
- K = feedback gain matrix
- x = state vector
- r = reference input

**Linear Quadratic Regulator (LQR)**: Optimizes control for linear systems
K = R⁻¹ * Bᵀ * P

vbnet
Copy code

Where P is the solution to the algebraic Riccati equation.

### Adaptive Control

Adaptive controllers adjust parameters based on system behavior:

**Model Reference Adaptive Control (MRAC)**:
θ̇ = -Γ * φ * e

markdown
Copy code

Where:
- θ = parameter vector
- Γ = adaptation gain
- φ = regressor vector
- e = tracking error

**Self-Tuning Regulators**: Adjust controller parameters based on system identification.

## Feedforward Control Strategies

### Preview Control

Preview control uses future reference trajectory to compute current control:

u(k) = -Kx(k) - K_f * Σ(i=0 to N-1) G_f(i) * r(k+i)

markdown
Copy code

Where:
- N = preview horizon
- r = reference trajectory
- K_f, G_f = preview control gains

### Feedforward Compensation

**Gravity Compensation**: Counteract gravitational forces
τ_gravity = G(q)

markdown
Copy code

**Coriolis Compensation**: Counteract velocity-dependent forces
τ_coriolis = C(q, q̇)q̇

markdown
Copy code

**Disturbance Feedforward**: Predict and counteract known disturbances
τ_disturbance = F_external

markdown
Copy code

### Trajectory-Based Feedforward

**Computed Torque Control**: Linearize system dynamics
τ = M(q) * (q̈_d + K_v * (q̇_d - q̇) + K_p * (q_d - q))

markdown
Copy code

## Disturbance Rejection

### Types of Disturbances

- **Impulsive Disturbances**: Sudden impacts or pushes  
- **Persistent Disturbances**: Continuous external forces  
- **Internal Disturbances**: Actuator failures, sensor noise  
- **Environmental Disturbances**: Wind, uneven terrain, slippery surfaces  

### Disturbance Observer

Estimate disturbances and compensate for them:

d̂ = L(s) * (y - ŷ)

markdown
Copy code

Where:
- d̂ = estimated disturbance
- L(s) = observer gain
- y = measured output
- ŷ = estimated output

### Robust Control

Design controllers that maintain performance despite uncertainties:

**H-infinity Control**: Minimize worst-case performance
||T_{wd}||_∞ < γ

pgsql
Copy code

Where T_{wd} is the transfer function from disturbance to output.

**Mu-Synthesis**: Handle structured uncertainties
μ_Δ(T) < 1

markdown
Copy code

## Recovery from Perturbations

### Balance Recovery Strategies

**Ankle Strategy**: Use ankle torques for small disturbances  
**Hip Strategy**: Use hip torques for medium disturbances  
**Stepping Strategy**: Take a step to expand support base  
**Suspension Strategy**: Move CoM to maintain balance  

### Recovery Phase Classification

- **Stability Boundary**: Threshold where different strategies become necessary  
- **Recovery Strategy Selection**: Choose strategy based on disturbance magnitude  
- **CoM State**: Position and velocity of center of mass  
- **Support State**: Current support polygon configuration  

### Multi-Step Recovery

1. **Initial Response**: Immediate reflexive action  
2. **Strategy Selection**: Choose appropriate recovery strategy  
3. **Execution**: Implement recovery action  
4. **Verification**: Confirm balance recovery  
5. **Return to Normal**: Resume normal behavior  

## Multi-Level Control Hierarchy

### High-Level Balance Control

- **Balance Planning**: Step planning, CoM trajectory, gait selection  
- **Task Prioritization**: Balance vs task execution, safety vs efficiency  

### Mid-Level Balance Control

- **ZMP/Capture Point Control**: Reference trajectory tracking  
- **Whole-Body Control**: Coordinate all degrees of freedom  

### Low-Level Balance Control

- **Joint-Level Control**: PID, current control, safety limits  
- **Sensor Processing**: IMU, force, and vision feedback  

## Technical Depth: Mathematical Foundations

### Linear Inverted Pendulum Model

ẍ_com = ω² * (x_com - x_zmp)
ÿ_com = ω² * (y_com - y_zmp)

powershell
Copy code

Where ω² = g/h (natural frequency).

### State-Space Representation

ẋ = Ax + Bu + Bd
y = Cx + Du

markdown
Copy code

### Stability Analysis

- **Lyapunov Stability**  
- **BIBO Stability**  
- **Routh-Hurwitz Criterion**  

### Optimal Control Formulation

J = ∫[xᵀQx + uᵀRu] dt
ẋ = Ax + Bu
u = -R⁻¹BᵀPx

markdown
Copy code

## Practical Applications

- **Standing Balance**: Quiet vs challenged stance  
- **Walking Balance**: Gait integration, ZMP trajectory  
- **Manipulation Balance**: Dual-task performance  
- **Recovery Scenarios**: Push and trip recovery  

## Challenges

- **Control Complexity**  
- **Real-time Performance**  
- **Modeling Uncertainty**  
- **Safety vs Performance**  

## Figure List

1. Balance control hierarchy diagram  
2. Feedback vs feedforward control comparison  
3. Disturbance rejection mechanisms  
4. Recovery strategy selection tree  
5. Stability margin visualization  

## Code Example: Balance Control Implementation

```python
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve

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
    kp_com: float = 10.0
    ki_com: float = 1.0
    kd_com: float = 2.0
    kp_zmp: float = 15.0
    kd_zmp: float = 5.0
    capture_threshold: float = 0.15
    max_step_size: float = 0.3
    max_torque: float = 50.0
    max_angular_velocity: float = 1.0

class BalanceController:
    def __init__(self, com_height: float = 0.8):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.com_height)
        self.state = BalanceState(
            com_pos=np.zeros(3),
            com_vel=np.zeros(3),
            com_acc=np.zeros(3),
            zmp_pos=np.zeros(2),
            capture_point=np.zeros(2),
            support_polygon=np.array([[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1]]),
            angular_momentum=np.zeros(3),
            timestamp=0.0
        )
        self.params = BalanceControlParams()
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.current_strategy = "ankle"
        self.recovery_active = False
        self.recovery_start_time = 0.0

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

    def distance_point_to_segment(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """Distance from point to line segment"""
        v = point - seg_start
        s = seg_end - seg_start
        proj_len = np.dot(v, s) / np.dot(s, s)
        if proj_len < 0:
            return np.linalg.norm(point - seg_start)
        elif proj_len > 1:
            return np.linalg.norm(point - seg_end)
        else:
            closest = seg_start + proj_len * s
            return np.linalg.norm(point - closest)

    def distance_to_polygon(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """Minimum distance from point to polygon boundary"""
        return min(self.distance_point_to_segment(point, polygon[i], polygon[(i+1)%len(polygon)]) for i in range(len(polygon)))

    def select_balance_strategy(self) -> str:
        """Select strategy based on CoM/ZMP state"""
        zmp_distance = self.distance_to_polygon(self.state.zmp_pos, self.state.support_polygon)
        capture_distance = self.distance_to_polygon(self.state.capture_point, self.state.support_polygon)
        if capture_distance > self.params.capture_threshold:
            return "stepping"
        elif capture_distance > 0.08:
            return "hip"
        elif zmp_distance > 0.05:
            return "ankle"
        else:
            return "ankle"

    def compute_ankle_strategy(self, dt: float) -> np.ndarray:
        """Compute ankle strategy control"""
        desired_com = np.array([0.0, 0.0, self.com_height])
        pos_error = desired_com[:2] - self.state.com_pos[:2]
        self.integral_error += pos_error * dt
        derivative_error = (pos_error - self.previous_error)/dt if dt>0 else np.zeros(2)
        control_output = self.params.kp_com*pos_error + self.params.ki_com*self.integral_error + self.params.kd_com*derivative_error
        self.previous_error = pos_error.copy()
        return control_output

    def compute_hip_strategy(self, dt: float) -> np.ndarray:
        """Compute hip strategy control"""
        desired_zmp = np.array([0.0, 0.0])
        zmp_error = desired_zmp - self.state.zmp_pos
        derivative_zmp = (zmp_error - self.previous_error)/dt if dt>0 else np.zeros(2)
        control_output = self.params.kp_zmp*zmp_error + self.params.kd_zmp*derivative_zmp
        self.previous_error = zmp_error.copy()
        return control_output

    def compute_stepping_strategy(self) -> np.ndarray:
        """Compute step target"""
        step_target = self.state.capture_point.copy()
        step_distance = np.linalg.norm(step_target)
        if step_distance > self.params.max_step_size:
            step_target = (step_target/step_distance)*self.params.max_step_size
        return step_target

    def compute_zmp(self, com_pos: np.ndarray, com_acc: np.ndarray) -> np.ndarray:
        """Simplified ZMP computation"""
        return com_pos[:2] - com_acc[:2]/self.omega**2

    def compute_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """Simplified capture point computation"""
        return com_pos[:2] + com_vel[:2]/self.omega

    def update_support_polygon(self):
        """Placeholder: normally would update based on foot contacts"""
        pass

    def update_balance_state(self, com_pos: np.ndarray, com_vel: np.ndarray, com_acc: np.ndarray, dt: float) -> BalanceState:
        """Update balance state and compute control"""
        self.state.com_pos = com_pos
        self.state.com_vel = com_vel
        self.state.com_acc = com_acc
        self.state.zmp_pos = self.compute_zmp(com_pos, com_acc)
        self.state.capture_point = self.compute_capture_point(com_pos, com_vel)
        self.update_support_polygon()
        self.current_strategy = self.select_balance_strategy()
        self.state.timestamp += dt
        return self.state

    def compute_joint_torques(self, control_output: np.ndarray) -> np.ndarray:
        """Map control to joint torques"""
        torques = np.zeros(20)
        torques[0] = control_output[0]*20
        torques[1] = control_output[1]*10
        torques[6] = -control_output[0]*15
        torques[7] = -control_output[1]*10
        torques = np.clip(torques, -self.params.max_torque, self.params.max_torque)
        return torques

class DisturbanceObserver:
    """Estimate external disturbances"""
    def __init__(self, model_uncertainty: float = 0.1):
        self.model_uncertainty = model_uncertainty
        self.estimated_disturbance = np.zeros(2)
        self.disturbance_error = np.zeros(2)
        self.gain = 1.0

    def update(self, measured_output: np.ndarray, predicted_output: np.ndarray, dt: float) -> np.ndarray:
        error = measured_output - predicted_output
        self.disturbance_error += error*dt
        self.estimated_disturbance = self.gain*error + 0.1*self.disturbance_error
        self.estimated_disturbance = np.clip(self.estimated_disturbance, -100, 100)
        return self.estimated_disturbance

class BalancePerformanceAnalyzer:
    """Analyze performance"""
    def __init__(self):
        self.metrics = {'stability_margin': [], 'control_effort': []}

    def calculate_stability_margin(self, zmp_pos: np.ndarray, support_polygon: np.ndarray) -> float:
        return min(np.linalg.norm(zmp_pos - vertex) for vertex in support_polygon)

    def calculate_control_effort(self, torques: np.ndarray) -> float:
        return np.sum(np.abs(torques))

    def add_sample(self, zmp_pos: np.ndarray, support_polygon: np.ndarray, torques: np.ndarray, timestamp: float):
        self.metrics['stability_margin'].append(self.calculate_stability_margin(zmp_pos, support_polygon))
        self.metrics['control_effort'].append(self.calculate_control_effort(torques))

    def get_performance_summary(self) -> Dict:
        return {
            'avg_stability_margin': np.mean(self.metrics['stability_margin']),
            'min_stability_margin': min(self.metrics['stability_margin']),
            'avg_control_effort': np.mean(self.metrics['control_effort']),
            'max_control_effort': max(self.metrics['control_effort']),
            'stability_std': np.std(self.metrics['stability_margin'])
        }


Exercises
Implement a balance controller that switches between ankle and hip strategies based on ZMP position.

Design a disturbance observer that can estimate external forces applied to a humanoid robot.

Create a multi-level balance control system with high-level planning and low-level joint control.

Summary
This chapter provided a comprehensive overview of balance and stability control for humanoid robots, covering feedback and feedforward control strategies, disturbance rejection techniques, and recovery mechanisms. We explored mathematical models, control algorithms, and multi-level control hierarchies essential for maintaining robot stability. The concepts and implementations presented will help in developing robust balance control systems for humanoid robots that can maintain equilibrium in various conditions and recover from disturbances safely.


