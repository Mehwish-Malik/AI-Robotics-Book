---
sidebar_position: 1
---

# Chapter 7: Motion Planning and Locomotion

## Summary

This chapter explores the fundamental concepts of motion planning for humanoid robots, focusing on how these complex systems navigate space and execute purposeful movements. We'll examine various motion planning algorithms, trajectory generation techniques, and the unique challenges of planning for bipedal locomotion. Understanding motion planning is crucial for developing robots that can move efficiently and safely in complex environments.

## Learning Outcomes

By the end of this chapter, you will be able to:
- Understand the principles of motion planning for humanoid robots
- Analyze different motion planning algorithms and their applications
- Generate trajectories for complex humanoid movements
- Implement obstacle avoidance strategies
- Evaluate motion planning performance in dynamic environments

## Key Concepts

- **Configuration Space (C-space)**: The space of all possible robot configurations
- **Motion Planning**: Finding a collision-free path from start to goal
- **Trajectory Generation**: Creating time-parameterized paths with velocity and acceleration profiles
- **Kinodynamic Planning**: Planning that considers both kinematic and dynamic constraints
- **Sampling-based Algorithms**: Algorithms that sample the configuration space
- **Optimization-based Planning**: Formulating planning as an optimization problem
- **Anytime Algorithms**: Algorithms that can return valid solutions at any time

## Introduction to Motion Planning

Motion planning for humanoid robots is significantly more complex than for simpler robots due to their high degrees of freedom and the need to maintain balance during movement. Unlike wheeled robots that operate in 2D space, humanoid robots must plan in high-dimensional configuration spaces while considering balance, obstacle avoidance, and dynamic constraints.

### Motion Planning Challenges

**High Dimensionality**: Humanoid robots typically have 30+ degrees of freedom, creating enormous configuration spaces.

**Balance Constraints**: The robot must maintain stability throughout the motion, limiting feasible configurations.

**Dynamic Constraints**: Acceleration and velocity limits must be respected to prevent damage and maintain stability.

**Real-time Requirements**: Many applications require motion planning to be computed in real-time.

**Environmental Complexity**: Real-world environments are dynamic and partially observable.

### Motion Planning Framework

The motion planning process typically involves:

1. **Problem Formulation**: Define start state, goal state, and constraints
2. **Space Representation**: Represent the configuration space and obstacles
3. **Planning Algorithm**: Apply algorithm to find a feasible path
4. **Trajectory Generation**: Convert path to time-parameterized trajectory
5. **Execution and Monitoring**: Execute trajectory and handle deviations

## Configuration Space and Representation

### Configuration Space (C-space)

For a humanoid robot, the configuration space is the set of all possible joint angles:

```
q = [q₁, q₂, ..., qₙ]
```

Where n is the number of degrees of freedom. The dimensionality of C-space makes planning computationally intensive.

### Free Space vs. Obstacle Space

**Free Space (C_free)**: Configurations where the robot does not collide with obstacles
**Obstacle Space (C_obs)**: Configurations where the robot collides with obstacles
**C_free = C_total - C_obs**

### Kinematic Constraints

Humanoid robots have various kinematic constraints:

**Joint Limits**: Each joint has physical limits
```
q_min ≤ q ≤ q_max
```

**Holonomic Constraints**: Constraints that can be expressed as functions of position
**Non-holonomic Constraints**: Constraints that depend on velocity and cannot be integrated to position constraints

## Motion Planning Algorithms

### Sampling-based Algorithms

Sampling-based algorithms explore the configuration space by randomly sampling configurations and connecting them to form a graph or tree.

**Probabilistic Roadmap (PRM)**:
- Pre-compute roadmap of the environment
- Query-specific planning using the roadmap
- Good for multiple queries in static environments

**Rapidly-exploring Random Trees (RRT)**:
- Grow tree from start configuration
- Random sampling guides exploration
- Good for single-query problems

**RRT*** (RRT-star):
- Asymptotically optimal version of RRT
- Rewires tree to improve solution quality
- Balances exploration and optimization

### Grid-based Algorithms

Grid-based methods discretize the configuration space into a grid:

**A* Algorithm**:
- Heuristic search algorithm
- Guarantees optimal solution
- Memory usage grows exponentially with dimensionality

**Dijkstra's Algorithm**:
- Explores outward from start
- Guarantees optimal solution
- No heuristic guidance

### Optimization-based Algorithms

Formulate planning as an optimization problem:

**CHOMP (Covariant Hamiltonian Optimization for Motion Planning)**:
- Trajectory optimization approach
- Uses covariant gradient to handle constraints
- Good for high-DOF systems

**STOMP (Stochastic Trajectory Optimization)**:
- Random sampling and optimization
- Handles complex cost functions
- Robust to local minima

## Trajectory Generation and Optimization

### Path vs. Trajectory

**Path**: Geometric route through configuration space
**Trajectory**: Time-parameterized path with velocity and acceleration profiles

### Trajectory Representation

Trajectories can be represented as:

**Polynomial Trajectories**:
```
q(t) = a₀ + a₁t + a₂t² + a₃t³ + ...
```

**Spline Trajectories**: Piecewise polynomial curves
**Bézier Curves**: Parametric curves defined by control points

### Time-Optimal Trajectory Generation

Minimize execution time while respecting constraints:

```
min ∫₀ᵀ dt
s.t. q(t) ∈ C_free ∀t ∈ [0,T]
     |q̇(t)| ≤ v_max
     |q̈(t)| ≤ a_max
```

### Velocity and Acceleration Profiles

**Trapezoidal Profile**: Constant acceleration, constant velocity, constant deceleration
**S-curve Profile**: Smooth acceleration/deceleration profiles for reduced jerk
**Minimum Jerk Profile**: Minimize jerk (third derivative) for smooth motion

## Obstacle Avoidance

### Static Obstacle Avoidance

**Configuration Space Obstacles**: Transform workspace obstacles to C-space
**Distance Fields**: Pre-computed distance to nearest obstacle
**Collision Detection**: Real-time collision checking algorithms

### Dynamic Obstacle Avoidance

**Velocity Obstacles**: Regions in velocity space that lead to collision
**Reciprocal Velocity Obstacles (RVO)**: Consider other agents' responses
**Optimal Reciprocal Collision Avoidance (ORCA)**: Linear constraints for collision avoidance

### Local Replanning

**D* Algorithm**: Incremental path replanning
**D* Lite**: Simpler version of D*
**ARA* (Anytime Repairing A*)**: Anytime algorithm with cost bounds

## Bipedal Locomotion Planning

### Walking Pattern Generation

**Predefined Gaits**: Fixed walking patterns for stable locomotion
**Online Gait Generation**: Real-time adaptation to terrain and conditions
**Footstep Planning**: Determine where to place feet

### Zero Moment Point (ZMP) Planning

ZMP-based planning ensures dynamic stability:

```
ZMP_x = (M_x + F_z * h) / F_z
ZMP_y = (M_y + F_z * h) / F_z
```

Where M_x, M_y are moments, F_z is vertical force, and h is height.

### Capture Point Planning

The capture point indicates where the robot must step to come to a stop:

```
Capture Point = CoM Position + CoM Velocity * √(Height / Gravity)
```

## Technical Depth: Mathematical Foundations

### Configuration Space Formulation

For a humanoid robot with n joints:

```
C = SE(3) × S¹ × S¹ × ... × S¹  (for n joints)
```

Where SE(3) represents the special Euclidean group (position and orientation) and S¹ represents joint angles.

### Collision Detection

**Bounding Volume Hierarchies (BVH)**: Hierarchical bounding volumes for efficient collision detection
**Separating Axis Theorem**: For convex object collision detection
**GJK Algorithm**: Gilbert-Johnson-Keerthi for distance computation

### Optimization Formulation

Trajectory optimization as a constrained optimization problem:

```
min ∫₀ᵀ [q̈(t)ᵀH(t)q̈(t) + q̇(t)ᵀR(t)q̇(t) + q(t)ᵀQ(t)q(t)] dt
s.t. f(q, q̇, q̈, t) = 0  (system dynamics)
     g(q, q̇, q̈, t) ≤ 0  (inequality constraints)
     q(0) = q_start, q(T) = q_goal  (boundary conditions)
```

## Practical Applications

### Navigation in Human Environments

Humanoid robots must navigate spaces designed for humans:
- Doorways and corridors
- Stairs and ramps
- Furniture and obstacles
- Moving humans and objects

### Manipulation Motion Planning

Coordinating arm movements with balance:
- Reaching without losing stability
- Avoiding self-collisions
- Coordinated multi-limb motion

### Multi-modal Locomotion

Planning for different types of movement:
- Walking
- Crawling
- Climbing
- Assisted movement

## Challenges

### Computational Complexity

High-dimensional configuration spaces require significant computational resources.

### Real-time Performance

Planning algorithms must run in real-time for responsive behavior.

### Uncertainty Handling

Sensing and modeling uncertainties affect planning reliability.

### Dynamic Environments

Moving obstacles and changing environments require continuous replanning.

## Figure List

1. **Figure 7.1**: Configuration space representation for humanoid robot
2. **Figure 7.2**: RRT algorithm visualization
3. **Figure 7.3**: Trajectory generation profiles
4. **Figure 7.4**: ZMP and Capture Point concepts
5. **Figure 7.5**: Obstacle avoidance strategies

## Code Example: Motion Planning Implementation

```python
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import random

@dataclass
class Node:
    """Node in the motion planning tree"""
    config: np.ndarray  # Configuration (position + orientation)
    parent: Optional['Node'] = None
    cost: float = 0.0
    heuristic: float = 0.0

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

class RRTPlanner:
    """Rapidly-exploring Random Tree planner for motion planning"""

    def __init__(self,
                 start: np.ndarray,
                 goal: np.ndarray,
                 bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 step_size: float = 0.1,
                 max_iterations: int = 10000):
        self.start = Node(start.copy())
        self.goal = Node(goal.copy())
        self.bounds = bounds  # ((x_min, x_max), (y_min, y_max))
        self.step_size = step_size
        self.max_iterations = max_iterations

        # Tree structure
        self.nodes = [self.start]
        self.goal_found = False

    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """Calculate distance between two configurations"""
        return np.linalg.norm(config1 - config2)

    def random_config(self) -> np.ndarray:
        """Generate random configuration within bounds"""
        x = random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = random.uniform(self.bounds[1][0], self.bounds[1][1])
        return np.array([x, y])

    def nearest_node(self, config: np.ndarray) -> Node:
        """Find nearest node in tree to given configuration"""
        distances = [self.distance(node.config, config) for node in self.nodes]
        nearest_idx = np.argmin(distances)
        return self.nodes[nearest_idx]

    def steer(self, from_node: Node, to_config: np.ndarray) -> np.ndarray:
        """Steer from one configuration towards another"""
        direction = to_config - from_node.config
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_config
        else:
            return from_node.config + (direction / distance) * self.step_size

    def is_collision_free(self, config: np.ndarray) -> bool:
        """Check if configuration is collision-free (simplified)"""
        # In a real implementation, this would check against obstacles
        # For this example, we'll assume no obstacles
        return True

    def plan(self) -> Optional[List[np.ndarray]]:
        """Plan path using RRT algorithm"""
        for i in range(self.max_iterations):
            # Sample random configuration
            rand_config = self.random_config()

            # Find nearest node
            nearest = self.nearest_node(rand_config)

            # Steer towards random configuration
            new_config = self.steer(nearest, rand_config)

            # Check collision
            if self.is_collision_free(new_config):
                # Create new node
                new_node = Node(new_config.copy())
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest.config, new_config)

                # Add to tree
                self.nodes.append(new_node)

                # Check if goal reached
                if self.distance(new_config, self.goal.config) < self.step_size:
                    self.goal_found = True
                    return self.extract_path(new_node)

        return None  # No path found

    def extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """Extract path from goal node back to start"""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.config.copy())
            current = current.parent
        return path[::-1]  # Reverse to get start-to-goal path

class TrajectoryGenerator:
    """Generate smooth trajectories from path waypoints"""

    def __init__(self, max_velocity: float = 1.0, max_acceleration: float = 2.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_polynomial_trajectory(self, waypoints: List[np.ndarray],
                                    times: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate cubic polynomial trajectory between waypoints"""
        if len(waypoints) < 2:
            return np.array([]), np.array([]), np.array([])

        # Time intervals between waypoints
        dt = np.diff(times)

        # Positions, velocities, and accelerations
        positions = []
        velocities = []
        accelerations = []

        # For each segment
        for i in range(len(waypoints) - 1):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]
            segment_time = dt[i]

            # Cubic polynomial coefficients for smooth transition
            # q(t) = a0 + a1*t + a2*t^2 + a3*t^3
            a0 = start_pos
            a1 = 0  # Assuming zero initial velocity
            a2 = 3 * (end_pos - start_pos) / (segment_time ** 2)
            a3 = -2 * (end_pos - start_pos) / (segment_time ** 3)

            # Generate points along the trajectory
            t_segment = np.linspace(0, segment_time, int(segment_time * 100) + 1)

            for t in t_segment:
                pos = a0 + a1 * t + a2 * t**2 + a3 * t**3
                vel = a1 + 2 * a2 * t + 3 * a3 * t**2
                acc = 2 * a2 + 6 * a3 * t

                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)

        return np.array(positions), np.array(velocities), np.array(accelerations)

    def velocity_smoothing(self, path: List[np.ndarray],
                          max_vel: float = 1.0,
                          max_acc: float = 2.0) -> List[Tuple[np.ndarray, float]]:
        """Apply velocity smoothing to path"""
        if len(path) < 2:
            return [(path[0], 0.0)] if path else []

        smoothed_path = []

        for i in range(len(path)):
            if i == 0:
                # First point
                smoothed_path.append((path[i], 0.0))
            elif i == len(path) - 1:
                # Last point - zero velocity
                smoothed_path.append((path[i], 0.0))
            else:
                # Intermediate points
                prev_pos = path[i-1]
                curr_pos = path[i]
                next_pos = path[i+1]

                # Calculate direction and distance
                dist_to_prev = np.linalg.norm(curr_pos - prev_pos)
                dist_to_next = np.linalg.norm(next_pos - curr_pos)

                # Determine appropriate velocity based on clearance
                velocity = min(max_vel, max(0.1, max(dist_to_prev, dist_to_next) * max_vel / 2))

                smoothed_path.append((curr_pos, velocity))

        return smoothed_path

class HumanoidMotionPlanner:
    """Motion planner specifically designed for humanoid robots"""

    def __init__(self, num_joints: int = 20):
        self.num_joints = num_joints
        self.balance_constraint = True  # Whether to consider balance
        self.max_joint_velocity = 5.0  # rad/s
        self.max_joint_acceleration = 10.0  # rad/s²

    def plan_reaching_motion(self, start_config: np.ndarray,
                           target_pos: np.ndarray,
                           target_orientation: Optional[np.ndarray] = None) -> Optional[List[np.ndarray]]:
        """Plan motion for reaching a target position"""
        if len(start_config) != self.num_joints:
            raise ValueError(f"Start configuration must have {self.num_joints} joints")

        # Simplified reaching motion planning
        # In reality, this would involve inverse kinematics and full motion planning

        # For this example, we'll create a simple linear interpolation
        # in joint space (not ideal, but demonstrates the concept)

        # This is a simplified example - real reaching would use:
        # 1. Inverse kinematics to find target configuration
        # 2. Motion planning to avoid obstacles
        # 3. Balance constraint checking

        target_config = start_config.copy()
        # Modify only the arm joints for reaching (simplified)
        for i in range(6, 12):  # Assuming arm joints are indices 6-11
            if i < len(target_config):
                target_config[i] += random.uniform(-0.5, 0.5)  # Random adjustment

        # Create trajectory by interpolating between start and target
        steps = 50
        trajectory = []
        for step in range(steps + 1):
            ratio = step / steps
            config = start_config + ratio * (target_config - start_config)
            trajectory.append(config)

        return trajectory

    def plan_walking_trajectory(self, start_pos: np.ndarray,
                              goal_pos: np.ndarray,
                              step_length: float = 0.3) -> List[np.ndarray]:
        """Plan a simple walking trajectory"""
        # Calculate number of steps needed
        distance = np.linalg.norm(goal_pos - start_pos)
        num_steps = int(distance / step_length)

        if num_steps == 0:
            return [start_pos]

        # Generate intermediate waypoints
        trajectory = []
        for i in range(num_steps + 1):
            ratio = i / num_steps
            waypoint = start_pos + ratio * (goal_pos - start_pos)
            trajectory.append(waypoint)

        return trajectory

    def check_balance_constraint(self, config: np.ndarray) -> bool:
        """Check if configuration maintains balance"""
        # Simplified balance check
        # In reality, this would compute center of mass and check against support polygon
        return True  # Placeholder

    def optimize_trajectory(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize trajectory for smoothness and efficiency"""
        if len(trajectory) < 3:
            return trajectory

        # Apply smoothing using a simple averaging filter
        optimized = [trajectory[0]]  # Keep first point

        for i in range(1, len(trajectory) - 1):
            # Average with neighbors for smoothing
            smoothed_point = (trajectory[i-1] + 2*trajectory[i] + trajectory[i+1]) / 4
            optimized.append(smoothed_point)

        optimized.append(trajectory[-1])  # Keep last point
        return optimized

def demonstrate_motion_planning():
    """Demonstrate motion planning concepts"""
    print("Motion Planning Demonstration")
    print("=" * 40)

    # Example 1: RRT Planning in 2D
    print("\n1. RRT Path Planning Example:")
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    bounds = ((-1.0, 6.0), (-1.0, 6.0))

    rrt = RRTPlanner(start, goal, bounds, step_size=0.2, max_iterations=1000)
    path = rrt.plan()

    if path:
        print(f"  Found path with {len(path)} waypoints")
        print(f"  Path length: {sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1)):.2f}")

        # Generate smooth trajectory
        times = [i * 0.1 for i in range(len(path))]
        traj_gen = TrajectoryGenerator(max_velocity=1.0, max_acceleration=2.0)
        positions, velocities, accelerations = traj_gen.generate_polynomial_trajectory(path, times)

        print(f"  Generated trajectory with {len(positions)} points")
    else:
        print("  No path found")

    # Example 2: Humanoid-specific planning
    print("\n2. Humanoid Motion Planning Example:")
    humanoid_planner = HumanoidMotionPlanner(num_joints=20)

    start_config = np.zeros(20)  # Starting configuration
    target_pos = np.array([1.0, 0.5, 0.8])  # Target position in workspace

    reaching_trajectory = humanoid_planner.plan_reaching_motion(start_config, target_pos)
    if reaching_trajectory:
        print(f"  Generated reaching trajectory with {len(reaching_trajectory)} configurations")

        # Optimize trajectory
        optimized = humanoid_planner.optimize_trajectory(reaching_trajectory)
        print(f"  Optimized trajectory with {len(optimized)} configurations")

    # Example 3: Walking trajectory
    print("\n3. Walking Trajectory Example:")
    start_pos = np.array([0.0, 0.0])
    goal_pos = np.array([3.0, 2.0])

    walking_traj = humanoid_planner.plan_walking_trajectory(start_pos, goal_pos)
    print(f"  Walking trajectory with {len(walking_traj)} steps")

    # Calculate step statistics
    if len(walking_traj) > 1:
        total_distance = sum(np.linalg.norm(walking_traj[i+1]-walking_traj[i])
                           for i in range(len(walking_traj)-1))
        print(f"  Total walking distance: {total_distance:.2f}m")
        print(f"  Average step length: {total_distance/max(1, len(walking_traj)-1):.2f}m")

    # Example 4: Trajectory smoothing
    print("\n4. Trajectory Smoothing Example:")
    # Create a zigzag path
    zigzag_path = []
    for i in range(10):
        x = i * 0.5
        y = 2.0 + (1.0 if i % 2 == 0 else -1.0) * 0.5
        zigzag_path.append(np.array([x, y]))

    smoothed = TrajectoryGenerator().velocity_smoothing(zigzag_path, max_vel=0.5)
    print(f"  Original path: {len(zigzag_path)} points")
    print(f"  Smoothed path: {len(smoothed)} points with velocity information")

def analyze_planning_performance():
    """Analyze motion planning performance metrics"""
    metrics = {
        'planning_time': [],
        'path_length': [],
        'smoothness': [],
        'success_rate': 0
    }

    # Simulate multiple planning scenarios
    successful_plans = 0
    total_plans = 10

    for i in range(total_plans):
        # Simulate planning time (in seconds)
        planning_time = random.uniform(0.01, 0.5)
        metrics['planning_time'].append(planning_time)

        # Simulate path length
        path_length = random.uniform(1.0, 10.0)
        metrics['path_length'].append(path_length)

        # Simulate smoothness (lower is smoother)
        smoothness = random.uniform(0.1, 2.0)
        metrics['smoothness'].append(smoothness)

        # Random success/failure
        if random.random() > 0.1:  # 90% success rate
            successful_plans += 1

    metrics['success_rate'] = successful_plans / total_plans

    print("\n5. Planning Performance Analysis:")
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Average Planning Time: {np.mean(metrics['planning_time']):.3f}s")
    print(f"  Average Path Length: {np.mean(metrics['path_length']):.2f}m")
    print(f"  Average Smoothness: {np.mean(metrics['smoothness']):.2f}")
    print(f"  Planning Time Std: {np.std(metrics['planning_time']):.3f}s")

if __name__ == "__main__":
    demonstrate_motion_planning()
    analyze_planning_performance()

    print("\nMotion Planning and Locomotion - Chapter 7 Complete!")
```

## Exercises

1. Implement a simple A* algorithm for path planning in a 2D grid environment.

2. Design a trajectory generator that creates minimum-jerk trajectories for a 6-DOF arm.

3. Create a motion planner that considers both collision avoidance and balance constraints for a simple humanoid model.

## Summary

This chapter provided a comprehensive overview of motion planning for humanoid robots, covering fundamental algorithms, trajectory generation techniques, and the unique challenges of planning for bipedal locomotion. We explored sampling-based algorithms like RRT, optimization-based approaches, and trajectory generation methods. The mathematical foundations and practical examples presented will help in developing motion planning systems for specific humanoid robot applications. Understanding these concepts is essential for creating robots that can navigate complex environments safely and efficiently.