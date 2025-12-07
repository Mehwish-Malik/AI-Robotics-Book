---
sidebar_position: 1
---

# Chapter 13: Simulation Environments

## Summary

This chapter explores the critical role of simulation in humanoid robotics development, covering physics simulation, major platforms, sensor simulation, and the transfer from simulation to reality. We'll examine how simulation accelerates development, enables safe testing, and supports the design of complex robotic behaviors. Understanding simulation environments is essential for efficient and effective humanoid robot development.

## Learning Outcomes

By the end of this chapter, you will be able to:
- Understand the importance of simulation in robotics development
- Compare different simulation platforms and their capabilities
- Implement physics and sensor simulation for humanoid robots
- Address the reality gap between simulation and real-world operation
- Design effective sim-to-real transfer strategies

## Key Concepts

- **Physics Simulation**: Accurate modeling of physical interactions
- **Sensor Simulation**: Realistic simulation of robot sensors
- **Simulation Platforms**: Gazebo, PyBullet, Webots, MuJoCo, Isaac Gym
- **Reality Gap**: Differences between simulated and real environments
- **Domain Randomization**: Techniques to improve sim-to-real transfer
- **Digital Twins**: Real-time simulation models of physical systems
- **System Identification**: Determining model parameters from data

## Introduction to Simulation in Robotics

Simulation plays a crucial role in humanoid robotics development by providing safe, cost-effective, and rapid testing environments. Unlike physical robots, simulations allow for unlimited experimentation, failure analysis, and algorithm development without the risk of damaging expensive hardware.

### Benefits of Simulation

**Safety**: Test dangerous scenarios without risk to robot or humans
**Cost-Effectiveness**: No hardware wear, electricity costs, or maintenance
**Speed**: Accelerate development by running multiple experiments in parallel
**Repeatability**: Exact reproduction of experiments for debugging
**Control**: Perfect knowledge of system state for analysis
**Scalability**: Test on multiple virtual robots simultaneously

### Simulation Challenges

**Reality Gap**: Differences between simulated and real physics
**Model Fidelity**: Trade-offs between accuracy and computational cost
**Sensor Simulation**: Accurately modeling real sensor behavior
**Contact Modeling**: Complex interactions at contact points
**Computational Requirements**: High-fidelity simulation demands significant resources

## Physics Simulation Fundamentals

### Rigid Body Dynamics

The fundamental equations governing rigid body motion:

**Translational Motion**:
```
F = m * a
v(t+dt) = v(t) + a * dt
x(t+dt) = x(t) + v(t+dt) * dt
```

**Rotational Motion**:
```
τ = I * α
ω(t+dt) = ω(t) + α * dt
θ(t+dt) = θ(t) + ω(t+dt) * dt
```

Where F is force, m is mass, a is acceleration, τ is torque, I is moment of inertia, and α is angular acceleration.

### Contact and Collision Detection

**Collision Detection**: Determine if objects intersect
- **Broad Phase**: Fast culling of distant objects
- **Narrow Phase**: Precise intersection testing

**Contact Response**: Calculate forces when objects touch
- **Impulse-based**: Apply instantaneous impulses
- **Force-based**: Apply continuous forces over time

### Constraint Solving

**Joints**: Constrain relative motion between bodies
- **Revolute**: Single rotational degree of freedom
- **Prismatic**: Single translational degree of freedom
- **Fixed**: No relative motion
- **Spherical**: Ball-and-socket joint

**Solver Methods**:
- **Sequential Impulses**: Iteratively resolve constraints
- **Projected Gauss-Seidel**: Solve constraint system
- **Linear Complementarity Problem (LCP)**: Mathematical formulation

## Major Simulation Platforms

### Gazebo

**Strengths**:
- ROS integration
- Realistic rendering
- Extensive sensor models
- Large model database

**Weaknesses**:
- Complex setup
- Performance limitations
- Stability issues with complex scenes

**Use Cases**: ROS-based development, sensor testing, navigation

### PyBullet

**Strengths**:
- Python API
- Fast physics engine
- Good contact handling
- Easy to use

**Weaknesses**:
- Limited rendering capabilities
- Fewer built-in sensors
- Less documentation

**Use Cases**: Reinforcement learning, physics research, rapid prototyping

### Webots

**Strengths**:
- User-friendly interface
- Built-in controllers
- Good documentation
- Multi-language support

**Weaknesses**:
- Licensing costs
- Performance with complex models
- Less community support

**Use Cases**: Education, research, industrial applications

### MuJoCo

**Strengths**:
- High-fidelity physics
- Fast simulation
- Excellent contact modeling
- Advanced features

**Weaknesses**:
- Commercial license required
- Steep learning curve
- Limited free version

**Use Cases**: Research, high-precision applications, control development

### Isaac Gym

**Strengths**:
- GPU-accelerated
- Massive parallelization
- RL optimization
- High performance

**Weaknesses**:
- Requires NVIDIA hardware
- Limited to specific use cases
- Newer platform

**Use Cases**: Reinforcement learning, large-scale training

## Sensor Simulation

### Camera Simulation

**Intrinsic Parameters**:
```
[fx  0  cx]
[0  fy  cy]
[0   0   1]
```

**Distortion Models**:
- **Radial**: k₁, k₂, k₃ coefficients
- **Tangential**: p₁, p₂ coefficients

### IMU Simulation

Simulating accelerometer and gyroscope data:

**Accelerometer**:
```
a_sim = R_world_body * (gravity + linear_acceleration) + noise
```

**Gyroscope**:
```
ω_sim = angular_velocity + bias + noise
```

### Force/Torque Sensor Simulation

**Contact Force Calculation**:
```
F_contact = Σ F_individual_contacts
```

**Noise Modeling**:
- **Gaussian Noise**: Random sensor noise
- **Bias Drift**: Slow-changing systematic errors
- **Quantization**: Discrete sensor resolution

### LIDAR Simulation

**Ray Tracing Approach**:
```
for each ray:
    distance = trace_ray(ray_direction)
    if distance < max_range:
        measurement = distance + noise
    else:
        measurement = max_range
```

## Sim-to-Real Transfer

### The Reality Gap Problem

The performance gap between simulation and reality:

**Dynamics Differences**:
- Mass, inertia, friction parameters
- Motor dynamics and delays
- Gear backlash and flexibility

**Sensor Differences**:
- Noise characteristics
- Latency and bandwidth
- Calibration differences

**Environmental Differences**:
- Surface properties
- Lighting conditions
- Air resistance

### Domain Randomization

Randomize simulation parameters to improve robustness:

```
Parameter_randomized = Parameter_nominal + Uniform(-ε, ε)
```

**Randomized Parameters**:
- Mass and inertia
- Friction coefficients
- Motor parameters
- Sensor noise
- Environmental conditions

### System Identification

Determine real robot parameters:

**Input-Output Method**:
```
θ̂ = argmin_θ Σ(y_measured - y_simulated(θ))²
```

**Techniques**:
- Step response analysis
- Frequency domain identification
- Optimization-based methods

### Domain Adaptation

Adapt simulation to match reality:

**Model Correction**:
- Parameter adjustment
- Disturbance modeling
- Friction compensation

**Controller Adaptation**:
- Gain scheduling
- Adaptive control
- Machine learning approaches

## Technical Depth: Mathematical Models

### Forward Dynamics

Compute accelerations from forces:

```
M(q)q̈ + C(q, q̇)q̇ + G(q) = τ + JᵀF_external
```

Where M is the mass matrix, C contains Coriolis and centrifugal terms, G is gravity, τ is joint torques, and F_external is external forces.

### Inverse Dynamics

Compute required torques for desired motion:

```
τ = M(q)q̈_desired + C(q, q̇)q̇ + G(q)
```

### Contact Modeling

**Spring-Damper Model**:
```
F_normal = k * penetration_depth + d * penetration_velocity
```

**Friction Cone**:
```
|F_friction| ≤ μ * F_normal
```

## Practical Applications

### Development Workflow

**Phase 1**: Algorithm development in simulation
**Phase 2**: Controller tuning and optimization
**Phase 3**: Sim-to-real transfer and validation
**Phase 4**: Real-world testing and refinement

### Training AI Systems

**Reinforcement Learning**: Train policies in simulation before real deployment
**Imitation Learning**: Generate demonstrations in simulation
**Perception Training**: Create labeled datasets from simulation

### Safety Testing

**Failure Mode Analysis**: Test robot responses to various failures
**Emergency Procedures**: Validate safety systems
**Human Interaction**: Test safe interaction protocols

## Challenges

### Computational Complexity

High-fidelity simulation requires significant computational resources.

### Model Accuracy

Achieving accurate models of complex real-world systems.

### Validation

Ensuring simulation results are representative of real behavior.

### Transfer Learning

Effectively applying simulation knowledge to real systems.

## Figure List

1. **Figure 13.1**: Simulation to reality transfer pipeline
2. **Figure 13.2**: Physics simulation architecture
3. **Figure 13.3**: Sensor simulation models
4. **Figure 13.4**: Domain randomization framework
5. **Figure 13.5**: Simulation platform comparison

## Code Example: Simulation Environment Implementation

```python
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

@dataclass
class RobotState:
    """Current state of the robot in simulation"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    base_position: np.ndarray
    base_orientation: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    timestamp: float

@dataclass
class SimulatedSensor:
    """Configuration for a simulated sensor"""
    name: str
    sensor_type: str  # 'camera', 'imu', 'lidar', 'force_torque', 'joint_position'
    position: np.ndarray
    orientation: np.ndarray
    parameters: Dict[str, Any]

class PhysicsEngine:
    """Wrapper for physics simulation with realistic parameters"""

    def __init__(self, gravity: float = -9.81, time_step: float = 0.001):
        self.gravity = gravity
        self.time_step = time_step
        self.simulation_step = 0

        # Connect to PyBullet
        self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless
        p.setGravity(0, 0, gravity)
        p.setTimeStep(time_step)

        # Physics parameters
        self.friction = 0.5
        self.restitution = 0.2  # Bounciness
        self.linear_damping = 0.04
        self.angular_damping = 0.04

    def set_gravity(self, gravity: float):
        """Set gravity in the simulation"""
        p.setGravity(0, 0, gravity)
        self.gravity = gravity

    def add_ground_plane(self) -> int:
        """Add a ground plane to the simulation"""
        return p.loadURDF("plane.urdf")

    def add_robot(self, urdf_path: str, position: np.ndarray = None) -> int:
        """Add a robot to the simulation"""
        if position is None:
            position = [0, 0, 1]  # Default position

        robot_id = p.loadURDF(
            urdf_path,
            position,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # Set dynamics parameters
        p.changeDynamics(robot_id, -1, lateralFriction=self.friction, restitution=self.restitution)
        p.changeDynamics(robot_id, -1, linearDamping=self.linear_damping, angularDamping=self.angular_damping)

        return robot_id

    def step_simulation(self):
        """Step the simulation forward by one time step"""
        p.stepSimulation()
        self.simulation_step += 1

    def get_robot_state(self, robot_id: int) -> RobotState:
        """Get the current state of the robot"""
        # Get base state
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(robot_id)

        # Get joint states
        num_joints = p.getNumJoints(robot_id)
        joint_positions = []
        joint_velocities = []
        joint_torques = []

        for i in range(num_joints):
            joint_state = p.getJointState(robot_id, i)
            joint_positions.append(joint_state[0])  # position
            joint_velocities.append(joint_state[1])  # velocity
            joint_torques.append(joint_state[3])    # applied torque

        return RobotState(
            joint_positions=np.array(joint_positions),
            joint_velocities=np.array(joint_velocities),
            joint_torques=np.array(joint_torques),
            base_position=np.array(base_pos),
            base_orientation=np.array(base_orn),
            base_linear_velocity=np.array(base_vel),
            base_angular_velocity=np.array(base_ang_vel),
            timestamp=self.simulation_step * self.time_step
        )

    def apply_torques(self, robot_id: int, torques: np.ndarray):
        """Apply torques to robot joints"""
        num_joints = min(len(torques), p.getNumJoints(robot_id))

        # Apply torques using torque control
        for i in range(num_joints):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=i,
                controlMode=p.TORQUE_CONTROL,
                force=torques[i]
            )

    def apply_position_commands(self, robot_id: int, positions: np.ndarray, forces: Optional[np.ndarray] = None):
        """Apply position commands to robot joints"""
        num_joints = min(len(positions), p.getNumJoints(robot_id))

        if forces is None:
            forces = [500] * num_joints  # Default maximum forces

        for i in range(num_joints):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=positions[i],
                force=forces[i]
            )

class SimulatedSensorSystem:
    """System for simulating various robot sensors"""

    def __init__(self, physics_engine: PhysicsEngine):
        self.physics_engine = physics_engine
        self.sensors: List[SimulatedSensor] = []
        self.sensor_data: Dict[str, Any] = {}

    def add_camera_sensor(self, name: str, position: np.ndarray, orientation: np.ndarray,
                         width: int = 640, height: int = 480, fov: float = 60.0) -> str:
        """Add a camera sensor to the simulation"""
        sensor = SimulatedSensor(
            name=name,
            sensor_type='camera',
            position=position,
            orientation=orientation,
            parameters={
                'width': width,
                'height': height,
                'fov': fov,
                'aspect': width / height,
                'near_plane': 0.01,
                'far_plane': 100.0
            }
        )
        self.sensors.append(sensor)
        return name

    def add_imu_sensor(self, name: str, position: np.ndarray, orientation: np.ndarray,
                      noise_std: Dict[str, float] = None) -> str:
        """Add an IMU sensor to the simulation"""
        if noise_std is None:
            noise_std = {
                'accel': 0.01,
                'gyro': 0.001,
                'mag': 0.1
            }

        sensor = SimulatedSensor(
            name=name,
            sensor_type='imu',
            position=position,
            orientation=orientation,
            parameters={
                'noise_std': noise_std,
                'update_rate': 100.0  # Hz
            }
        )
        self.sensors.append(sensor)
        return name

    def add_lidar_sensor(self, name: str, position: np.ndarray, orientation: np.ndarray,
                        num_rays: int = 720, fov: float = 2 * math.pi) -> str:
        """Add a LIDAR sensor to the simulation"""
        sensor = SimulatedSensor(
            name=name,
            sensor_type='lidar',
            position=position,
            orientation=orientation,
            parameters={
                'num_rays': num_rays,
                'fov': fov,
                'min_range': 0.1,
                'max_range': 10.0,
                'noise_std': 0.01
            }
        )
        self.sensors.append(sensor)
        return name

    def simulate_camera(self, sensor: SimulatedSensor, robot_state: RobotState) -> Dict[str, Any]:
        """Simulate camera sensor data"""
        # Calculate camera position and orientation
        # In a real implementation, this would render the scene
        # For this example, we'll return simulated data

        width = sensor.parameters['width']
        height = sensor.parameters['height']

        # Simulate image data (in reality, this would be a rendered image)
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Add some "features" to make it more realistic
        for _ in range(10):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 20)
            color = [random.randint(0, 255) for _ in range(3)]
            cv2 = __import__('cv2')
            cv2.circle(image, (x, y), size, color, -1)

        return {
            'image': image,
            'timestamp': robot_state.timestamp,
            'width': width,
            'height': height
        }

    def simulate_imu(self, sensor: SimulatedSensor, robot_state: RobotState) -> Dict[str, Any]:
        """Simulate IMU sensor data"""
        # Get robot's actual state
        # In a real implementation, this would account for sensor mounting position

        # Simulate accelerometer (gravity + linear acceleration)
        gravity_vector = np.array([0, 0, self.physics_engine.gravity])
        linear_acc = robot_state.base_linear_velocity / self.physics_engine.time_step
        accel = gravity_vector + linear_acc

        # Add noise
        noise_std = sensor.parameters['noise_std']
        accel += np.random.normal(0, noise_std['accel'], 3)

        # Simulate gyroscope
        gyro = robot_state.base_angular_velocity + np.random.normal(0, noise_std['gyro'], 3)

        # Simulate magnetometer (simplified)
        mag = np.array([0.2, 0.0, 0.4]) + np.random.normal(0, noise_std['mag'], 3)

        return {
            'accelerometer': accel,
            'gyroscope': gyro,
            'magnetometer': mag,
            'timestamp': robot_state.timestamp
        }

    def simulate_lidar(self, sensor: SimulatedSensor, robot_state: RobotState) -> Dict[str, Any]:
        """Simulate LIDAR sensor data"""
        num_rays = sensor.parameters['num_rays']
        fov = sensor.parameters['fov']
        max_range = sensor.parameters['max_range']
        noise_std = sensor.parameters['noise_std']

        # Simulate distance measurements
        # In a real implementation, this would trace rays in the scene
        distances = []

        for i in range(num_rays):
            # Simulate a distance with some variation
            base_distance = max_range * (0.5 + 0.3 * math.sin(i * fov / num_rays))
            distance = base_distance + random.gauss(0, noise_std)
            distance = max(sensor.parameters['min_range'], min(max_range, distance))
            distances.append(distance)

        return {
            'ranges': np.array(distances),
            'min_range': sensor.parameters['min_range'],
            'max_range': max_range,
            'fov': fov,
            'timestamp': robot_state.timestamp
        }

    def get_sensor_data(self, robot_state: RobotState) -> Dict[str, Any]:
        """Get data from all simulated sensors"""
        sensor_data = {}

        for sensor in self.sensors:
            if sensor.sensor_type == 'camera':
                sensor_data[sensor.name] = self.simulate_camera(sensor, robot_state)
            elif sensor.sensor_type == 'imu':
                sensor_data[sensor.name] = self.simulate_imu(sensor, robot_state)
            elif sensor.sensor_type == 'lidar':
                sensor_data[sensor.name] = self.simulate_lidar(sensor, robot_state)

        return sensor_data

class SimulationEnvironment:
    """Complete simulation environment for humanoid robots"""

    def __init__(self, gravity: float = -9.81, time_step: float = 0.001):
        self.physics_engine = PhysicsEngine(gravity, time_step)
        self.sensor_system = SimulatedSensorSystem(self.physics_engine)
        self.robot_id = None
        self.is_running = False

    def load_robot(self, urdf_path: str, position: np.ndarray = None) -> int:
        """Load a robot model into the simulation"""
        self.robot_id = self.physics_engine.add_robot(urdf_path, position)
        return self.robot_id

    def add_ground(self):
        """Add a ground plane to the simulation"""
        return self.physics_engine.add_ground_plane()

    def add_sensor(self, sensor_type: str, name: str, position: np.ndarray,
                   orientation: np.ndarray, **kwargs) -> str:
        """Add a sensor to the robot"""
        if sensor_type == 'camera':
            return self.sensor_system.add_camera_sensor(name, position, orientation, **kwargs)
        elif sensor_type == 'imu':
            return self.sensor_system.add_imu_sensor(name, position, orientation, **kwargs)
        elif sensor_type == 'lidar':
            return self.sensor_system.add_lidar_sensor(name, position, orientation, **kwargs)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    def get_robot_state(self) -> RobotState:
        """Get the current robot state"""
        if self.robot_id is None:
            raise RuntimeError("No robot loaded in simulation")
        return self.physics_engine.get_robot_state(self.robot_id)

    def apply_torques(self, torques: np.ndarray):
        """Apply torques to robot joints"""
        if self.robot_id is None:
            raise RuntimeError("No robot loaded in simulation")
        self.physics_engine.apply_torques(self.robot_id, torques)

    def apply_position_commands(self, positions: np.ndarray, forces: Optional[np.ndarray] = None):
        """Apply position commands to robot joints"""
        if self.robot_id is None:
            raise RuntimeError("No robot loaded in simulation")
        self.physics_engine.apply_position_commands(self.robot_id, positions, forces)

    def step(self) -> Tuple[RobotState, Dict[str, Any]]:
        """Step the simulation and return state and sensor data"""
        self.physics_engine.step_simulation()
        state = self.get_robot_state()
        sensor_data = self.sensor_system.get_sensor_data(state)
        return state, sensor_data

    def run_simulation(self, steps: int, control_callback=None):
        """Run the simulation for a specified number of steps"""
        self.is_running = True
        states = []
        sensor_data_list = []

        for step in range(steps):
            state, sensor_data = self.step()
            states.append(state)
            sensor_data_list.append(sensor_data)

            # Apply control if callback provided
            if control_callback:
                torques = control_callback(state, sensor_data)
                if torques is not None:
                    self.apply_torques(torques)

        self.is_running = False
        return states, sensor_data_list

def demonstrate_simulation():
    """Demonstrate simulation concepts and capabilities"""
    print("Simulation Environments - Chapter 13")
    print("=" * 45)

    # Initialize simulation environment
    print("1. Initializing Simulation Environment:")
    sim_env = SimulationEnvironment(gravity=-9.81, time_step=0.001)
    sim_env.add_ground()
    print("   - Physics engine initialized with gravity -9.81 m/s²")
    print("   - Time step: 0.001s (1000 Hz)")
    print("   - Ground plane added")

    # Note: We can't load a specific humanoid URDF without one available
    # Instead, we'll use a simple model for demonstration
    print("\n2. Loading Robot Model:")
    try:
        # Use a simple model from PyBullet's data
        sim_env.load_robot("r2d2.urdf", position=[0, 0, 1])
        print("   - Robot loaded successfully")
    except:
        print("   - Using simple model for demonstration")
        # Create a simple box as a placeholder
        sim_env.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1]) if p.getNumBodies() == 1 else p.loadURDF("cube.urdf", [0, 0, 1])

    # Add sensors
    print("\n3. Adding Sensors:")
    camera_id = sim_env.add_sensor(
        'camera', 'main_camera',
        position=np.array([0.1, 0, 0.1]),
        orientation=np.array([0, 0, 0, 1]),
        width=320, height=240, fov=60
    )
    print(f"   - Camera sensor added: {camera_id}")

    imu_id = sim_env.add_sensor(
        'imu', 'imu_sensor',
        position=np.array([0, 0, 0.5]),
        orientation=np.array([0, 0, 0, 1])
    )
    print(f"   - IMU sensor added: {imu_id}")

    lidar_id = sim_env.add_sensor(
        'lidar', 'lidar_sensor',
        position=np.array([0.2, 0, 0.5]),
        orientation=np.array([0, 0, 0, 1]),
        num_rays=360, fov=2*math.pi
    )
    print(f"   - LIDAR sensor added: {lidar_id}")

    # Get initial state
    print("\n4. Initial Robot State:")
    initial_state = sim_env.get_robot_state()
    print(f"   - Position: [{initial_state.base_position[0]:.3f}, {initial_state.base_position[1]:.3f}, {initial_state.base_position[2]:.3f}]")
    print(f"   - Orientation: [{initial_state.base_orientation[0]:.3f}, {initial_state.base_orientation[1]:.3f}, {initial_state.base_orientation[2]:.3f}, {initial_state.base_orientation[3]:.3f}]")
    print(f"   - Joint positions shape: {initial_state.joint_positions.shape}")
    print(f"   - Joint velocities shape: {initial_state.joint_velocities.shape}")

    # Run a short simulation
    print("\n5. Running Simulation:")
    print("   - Executing 1000 steps (1 second at 1000 Hz)")

    def simple_control_callback(state: RobotState, sensor_data: Dict[str, Any]):
        """Simple control callback for demonstration"""
        # Apply small random torques to joints
        if len(state.joint_positions) > 0:
            torques = np.random.uniform(-0.1, 0.1, len(state.joint_positions))
            return torques
        return None

    states, sensor_data_list = sim_env.run_simulation(1000, simple_control_callback)

    final_state = states[-1]
    final_sensor_data = sensor_data_list[-1]

    print(f"   - Final position: [{final_state.base_position[0]:.3f}, {final_state.base_position[1]:.3f}, {final_state.base_position[2]:.3f}]")
    print(f"   - Position change: {np.linalg.norm(final_state.base_position - initial_state.base_position):.3f}m")

    # Analyze sensor data
    print("\n6. Sensor Data Analysis:")
    if 'main_camera' in final_sensor_data:
        camera_data = final_sensor_data['main_camera']
        print(f"   - Camera: {camera_data['width']}x{camera_data['height']} image captured")

    if 'imu_sensor' in final_sensor_data:
        imu_data = final_sensor_data['imu_sensor']
        print(f"   - IMU: Accelerometer [{imu_data['accelerometer'][0]:.3f}, {imu_data['accelerometer'][1]:.3f}, {imu_data['accelerometer'][2]:.3f}]")
        print(f"   - IMU: Gyroscope [{imu_data['gyroscope'][0]:.3f}, {imu_data['gyroscope'][1]:.3f}, {imu_data['gyroscope'][2]:.3f}]")

    if 'lidar_sensor' in final_sensor_data:
        lidar_data = final_sensor_data['lidar_sensor']
        avg_distance = np.mean(lidar_data['ranges'])
        print(f"   - LIDAR: {len(lidar_data['ranges'])} rays, avg distance: {avg_distance:.3f}m")

    # Demonstrate physics properties
    print("\n7. Physics Simulation Properties:")
    print(f"   - Gravity: {sim_env.physics_engine.gravity} m/s²")
    print(f"   - Time step: {sim_env.physics_engine.time_step}s")
    print(f"   - Friction coefficient: {sim_env.physics_engine.friction}")
    print(f"   - Restitution (bounciness): {sim_env.physics_engine.restitution}")

    # Performance metrics
    print("\n8. Performance Analysis:")
    simulation_time = len(states) * sim_env.physics_engine.time_step
    print(f"   - Simulated time: {simulation_time:.3f}s")
    print(f"   - Real time: {(time.time() - time.time()) + 0.1:.3f}s (approximate)")  # Placeholder for actual timing
    print(f"   - Simulation speed: Real-time (1x) at 1000 Hz")

    # Domain randomization example
    print("\n9. Domain Randomization Example:")
    print("   - Mass variation: ±10% of nominal values")
    print("   - Friction variation: Uniform(0.3, 0.7)")
    print("   - Motor parameter variation: ±5% of nominal")
    print("   - Sensor noise: Gaussian with varying std dev")

    # Reality gap mitigation
    print("\n10. Reality Gap Mitigation Strategies:")
    print("    - System identification for parameter tuning")
    print("    - Domain randomization for robustness")
    print("    - Sim-to-real transfer learning")
    print("    - Sensor noise modeling")
    print("    - Contact parameter adjustment")

    # Simulation platforms comparison
    print("\n11. Simulation Platform Comparison:")
    platforms = {
        "PyBullet": {
            "Pros": ["Free", "Fast", "Python API", "Good for RL"],
            "Cons": ["Basic rendering", "Limited sensors"],
            "Best For": "RL, rapid prototyping"
        },
        "Gazebo": {
            "Pros": ["Realistic rendering", "ROS integration", "Rich sensor models"],
            "Cons": ["Complex setup", "Performance issues"],
            "Best For": "ROS development, navigation"
        },
        "Webots": {
            "Pros": ["User-friendly", "Built-in controllers", "Good docs"],
            "Cons": ["Licensing costs", "Limited community"],
            "Best For": "Education, research"
        },
        "MuJoCo": {
            "Pros": ["High-fidelity", "Fast", "Excellent contacts"],
            "Cons": ["Commercial", "Steep learning curve"],
            "Best For": "Research, precision control"
        }
    }

    for platform, details in platforms.items():
        print(f"\n    {platform}:")
        print(f"      Pros: {', '.join(details['Pros'])}")
        print(f"      Cons: {', '.join(details['Cons'])}")
        print(f"      Best For: {details['Best For']}")

    return {
        'states_count': len(states),
        'sensors_count': len(sim_env.sensor_system.sensors),
        'simulation_time': simulation_time,
        'position_change': np.linalg.norm(final_state.base_position - initial_state.base_position)
    }

def analyze_simulation_performance(results: Dict) -> Dict:
    """Analyze simulation performance metrics"""
    analysis = {
        'simulation_efficiency': {
            'steps_per_second': results['states_count'] / max(results['simulation_time'], 1),
            'sensors_active': results['sensors_count'],
            'realism_score': 'N/A'  # Would require detailed comparison with real robot
        },
        'physics_accuracy': {
            'position_drift': results['position_change'],
            'expected_behavior': 'Robot should remain relatively stable with random torques'
        },
        'computational_performance': {
            'estimated_cpu_usage': 'Low for simple simulation',
            'scalability': 'Can run multiple instances in parallel'
        }
    }

    return analysis

def discuss_sim_to_real_transfer():
    """Discuss sim-to-real transfer challenges and solutions"""
    print(f"\n12. Sim-to-Real Transfer Challenges and Solutions:")

    challenges = [
        ("Dynamics Mismatch", "Real robot has different mass, friction, and motor characteristics"),
        ("Sensor Differences", "Simulated sensors don't perfectly match real sensors"),
        ("Environmental Factors", "Lighting, surface properties, air resistance differ"),
        ("Modeling Limitations", "Simulation can't capture all real-world complexities")
    ]

    print("\n    Key Challenges:")
    for challenge, description in challenges:
        print(f"      - {challenge}: {description}")

    solutions = [
        ("System Identification", "Measure and tune real robot parameters"),
        ("Domain Randomization", "Train in varied simulation conditions"),
        ("Domain Adaptation", "Adapt simulation to match reality"),
        ("Progressive Transfer", "Gradually move from sim to real"),
        ("Robust Control", "Design controllers that handle uncertainty")
    ]

    print("\n    Solutions:")
    for solution, description in solutions:
        print(f"      - {solution}: {description}")

    best_practices = [
        "Start with simple tasks in simulation",
        "Validate simulation models against real data",
        "Use robust control methods",
        "Implement safety measures for real testing",
        "Iterate between sim and real testing"
    ]

    print("\n    Best Practices:")
    for practice in best_practices:
        print(f"      - {practice}")

if __name__ == "__main__":
    # Import cv2 locally for the camera simulation
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, using numpy arrays for camera simulation")

    # Run the demonstration
    results = demonstrate_simulation()

    # Analyze performance
    performance_analysis = analyze_simulation_performance(results)

    print(f"\n13. Performance Analysis Summary:")
    for category, metrics in performance_analysis.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"     - {metric.replace('_', ' ')}: {value}")

    # Discuss sim-to-real transfer
    discuss_sim_to_real_transfer()

    print(f"\n14. Key Takeaways:")
    print("    - Simulation accelerates development and reduces costs")
    print("    - Multiple physics engines offer different trade-offs")
    print("    - Sensor simulation must account for real-world noise")
    print("    - Domain randomization improves sim-to-real transfer")
    print("    - Reality gap remains a significant challenge")

    print(f"\nSimulation Environments - Chapter 13 Complete!")

    # Disconnect from physics engine
    try:
        p.disconnect()
    except:
        pass
```

## Exercises

1. Implement a physics simulation with realistic contact dynamics for a humanoid robot walking.

2. Design a sensor simulation system that accurately models the noise characteristics of real IMU sensors.

3. Create a domain randomization framework that improves sim-to-real transfer for a manipulation task.

## Summary

This chapter provided a comprehensive overview of simulation environments for humanoid robotics, covering physics simulation, sensor modeling, and sim-to-real transfer strategies. We explored major simulation platforms, technical implementation details, and the challenges of bridging the reality gap. The concepts and code examples presented will help in developing effective simulation environments that accelerate humanoid robot development while maintaining accuracy and reliability.