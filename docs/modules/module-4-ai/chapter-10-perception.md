---
sidebar_position: 1
---

# Chapter 10: Computer Vision and Perception

## Summary

This chapter explores the critical role of computer vision in humanoid robotics, focusing on how robots perceive and understand their visual environment. We'll examine object recognition, scene understanding, visual servoing, SLAM systems, and the integration of vision with other sensory modalities. Understanding computer vision is essential for creating robots that can operate autonomously in complex, visually-rich environments.

## Learning Outcomes

By the end of this chapter, you will be able to:
- Understand the fundamentals of computer vision for robotics applications
- Implement object detection and recognition algorithms
- Apply SLAM (Simultaneous Localization and Mapping) techniques
- Integrate vision with other sensory modalities
- Design visual servoing systems for robot control

## Key Concepts

- **Computer Vision**: Extraction of information from visual data
- **Object Detection**: Locating and identifying objects in images
- **Scene Understanding**: Interpreting environmental context
- **SLAM**: Simultaneous Localization and Mapping
- **Visual Servoing**: Controlling robot motion based on visual feedback
- **Multi-modal Perception**: Integration of vision with other sensors
- **Real-time Processing**: Processing visual data within time constraints

## Introduction to Computer Vision in Robotics

Computer vision enables humanoid robots to perceive and interpret their visual environment, which is crucial for navigation, manipulation, and interaction. Unlike traditional computer vision applications, robotics vision must operate in real-time with uncertain conditions and must integrate with other systems for complete environmental understanding.

### Vision Requirements in Robotics

**Real-time Processing**: Visual processing must keep up with robot motion and control rates
**Robustness**: Systems must handle varying lighting, occlusions, and environmental conditions
**Accuracy**: Precise measurements required for manipulation and navigation
**Integration**: Vision data must integrate with other sensors and control systems
**Efficiency**: Limited computational resources on robot platforms

### Vision System Architecture

```
Image Acquisition → Preprocessing → Feature Extraction → Recognition → Action
```

Each stage must be optimized for the robot's specific requirements and constraints.

## Object Detection and Recognition

### Traditional Approaches

**Template Matching**: Compare image patches with stored templates
- Simple to implement
- Sensitive to scale, rotation, and lighting changes
- Computationally expensive for large template sets

**Feature-based Methods**: Extract distinctive features and match them
- SIFT (Scale-Invariant Feature Transform)
- SURF (Speeded Up Robust Features)
- HOG (Histogram of Oriented Gradients)

### Deep Learning Approaches

**Convolutional Neural Networks (CNNs)**: Hierarchical feature extraction
```
Input Image → Conv Layers → Pooling → Fully Connected → Output
```

**Object Detection Networks**:
- **YOLO (You Only Look Once)**: Single-pass detection
- **SSD (Single Shot Detector)**: Multi-scale feature maps
- **R-CNN Family**: Region-based detection with proposal generation

### Recognition Pipelines

**Two-Stage Pipeline**:
1. Region Proposal: Identify potential object locations
2. Classification: Classify proposed regions

**Single-Stage Pipeline**:
- Direct prediction of bounding boxes and classes
- Faster but potentially less accurate

### 3D Object Recognition

**Multi-view Recognition**: Combine information from multiple viewpoints
**Shape-based Recognition**: Use 3D shape models for recognition
**Deep Learning on 3D Data**: Point clouds, meshes, and voxel grids

## Scene Understanding

### Semantic Segmentation

Assigning semantic labels to each pixel in an image:

```
L = argmax_l Σ_i P(label_i = l | image, position_i)
```

**FCN (Fully Convolutional Networks)**: End-to-end segmentation
**U-Net**: Encoder-decoder with skip connections
**DeepLab**: Atrous convolutions for multi-scale context

### Instance Segmentation

Distinguishing individual object instances:

**Mask R-CNN**: Extends object detection with segmentation masks
**YOLACT**: Real-time instance segmentation
**PolarMask**: Instance segmentation as instance classification

### Panoptic Segmentation

Combining semantic and instance segmentation:

- **Thing classes**: Countable objects (people, cars)
- **Stuff classes**: Uncountable materials (sky, road)

### Depth Estimation

**Stereo Vision**: Calculate depth from multiple camera views
```
depth = (baseline * focal_length) / disparity
```

**Monocular Depth Estimation**: Deep learning from single images
**LiDAR Integration**: Combining vision with active depth sensing

## Visual Servoing

### Definition and Applications

Visual servoing uses visual feedback to control robot motion:

**Task**: Move robot to achieve desired visual configuration
**Feedback**: Image features (points, lines, objects)
**Control**: Joint or Cartesian space commands

### Image-Based Visual Servoing (IBVS)

Control directly in image space:

```
Ẋ = J_x * v = J_x * J_v⁻¹ * J_θ * θ̇
```

Where:
- Ẋ = image velocity
- J_x = image Jacobian
- v = Cartesian velocity
- J_v = velocity Jacobian
- J_θ = kinematic Jacobian
- θ̇ = joint velocity

**Advantages**: Direct image feature control
**Disadvantages**: May pass through singularities

### Position-Based Visual Servoing (PBVS)

Control in Cartesian space using 3D positions:

```
v = -λ * J⁻¹ * e
```

Where:
- v = Cartesian velocity
- J = interaction matrix
- e = error vector
- λ = gain parameter

**Advantages**: Predictable Cartesian motion
**Disadvantages**: Requires 3D structure estimation

### Hybrid Approaches

Combine image and position-based control:
- Use image features for orientation
- Use 3D positions for translation
- Better robustness and convergence

## SLAM (Simultaneous Localization and Mapping)

### SLAM Fundamentals

SLAM estimates robot trajectory and map simultaneously:

```
P(x_t, m | z_1:t, u_1:t, x_0)
```

Where:
- x_t = robot pose at time t
- m = map
- z_t = observations at time t
- u_t = controls at time t

### Feature-based SLAM

**Front-end**: Extract and match features
**Back-end**: Optimize pose and map estimates
**Loop Closure**: Detect revisited locations

**ORB-SLAM**: Real-time SLAM with ORB features
**LSD-SLAM**: Direct method using line segments
**SVO**: Semi-direct visual odometry

### Direct SLAM

Use raw pixel intensities instead of features:

**DTAM**: Dense tracking and mapping
**LSD-SLAM**: Line segment detection and tracking
**SVO**: Semi-direct visual odometry

**Advantages**: Works in textureless environments
**Disadvantages**: Sensitive to lighting changes

### Deep Learning SLAM

**CNN-based Feature Extraction**: Learn optimal features
**End-to-End Learning**: Joint optimization of all components
**Uncertainty Estimation**: Learned uncertainty representations

## Multi-modal Perception

### Sensor Fusion with Vision

**Vision + IMU**: Visual-inertial odometry (VIO)
- Visual features for positioning
- IMU for motion prediction
- Robust to motion blur and fast motion

**Vision + LiDAR**: Dense mapping and localization
- LiDAR for accurate depth
- Vision for semantic information
- Complementary strengths

**Vision + Tactile**: Active exploration and recognition
- Vision for pre-grasp planning
- Tactile for contact verification
- Haptic feedback for manipulation

### Cross-modal Learning

**Vision-Language Models**: Understanding natural language commands
**Vision-Audio Integration**: Sound localization and scene understanding
**Vision-Touch Fusion**: Improved object property estimation

## Technical Depth: Mathematical Foundations

### Camera Models

**Pinhole Camera Model**:
```
[u]   [f_x  0   c_x] [R | t] [X]
[v] = [0   f_y  c_y] [0 | 1] [Y]
[1]   [0   0    1  ]         [Z]
                          [1]
```

Where:
- (u,v) = pixel coordinates
- (f_x,f_y) = focal lengths
- (c_x,c_y) = principal point
- [R|t] = extrinsic parameters

**Distortion Models**:
```
x_corrected = x * (1 + k_1*r² + k_2*r⁴ + k_3*r⁶) + 2*p_1*x*y + p_2*(r² + 2*x²)
y_corrected = y * (1 + k_1*r² + k_2*r⁴ + k_3*r⁶) + p_1*(r² + 2*y²) + 2*p_2*x*y
```

### Epipolar Geometry

For stereo vision and multi-view geometry:

**Fundamental Matrix**:
```
x₂ᵀ * F * x₁ = 0
```

**Essential Matrix**:
```
x₂ᵀ * E * x₁ = 0
```

Where E = K₂ᵀ * F * K₁

### Visual Jacobian

Relate image feature velocities to camera motion:

```
ṗ = J_p * ξ
```

Where:
- ṗ = image feature velocity
- J_p = interaction matrix
- ξ = camera velocity (6 DOF)

## Practical Applications

### Navigation and Mapping

**Visual Navigation**: Using vision for path planning and obstacle avoidance
**Semantic Mapping**: Creating maps with object and area labels
**Dynamic Scene Handling**: Dealing with moving objects and changing environments

### Manipulation and Grasping

**Visual Servoing for Grasping**: Precise positioning for object manipulation
**Object Pose Estimation**: 6D pose for robotic manipulation
**Grasp Planning**: Using visual information to plan grasps

### Human-Robot Interaction

**Gaze Following**: Directing attention based on human gaze
**Gesture Recognition**: Understanding human gestures and actions
**Social Navigation**: Navigating around humans safely

## Challenges

### Real-time Processing

Processing high-resolution images at control rates (100-1000Hz) is computationally demanding.

### Environmental Variability

Lighting changes, occlusions, and dynamic environments affect recognition performance.

### Scale and Resource Constraints

Limited computational power and memory on robot platforms.

### Integration Complexity

Fusing vision with other sensors and control systems requires careful coordination.

## Figure List

1. **Figure 10.1**: Computer vision pipeline for robotics
2. **Figure 10.2**: Object detection and recognition framework
3. **Figure 10.3**: Visual servoing control loop
4. **Figure 10.4**: SLAM system architecture
5. **Figure 10.5**: Multi-modal perception integration

## Code Example: Computer Vision Implementation

```python
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import time

@dataclass
class ImageFeature:
    """Represents a visual feature in an image"""
    point: np.ndarray  # 2D image coordinates [u, v]
    descriptor: np.ndarray  # Feature descriptor vector
    id: int  # Unique feature identifier
    scale: float = 1.0  # Scale of detection

@dataclass
class ObjectDetection:
    """Result of object detection"""
    bbox: np.ndarray  # Bounding box [x1, y1, x2, y2]
    confidence: float  # Detection confidence
    class_id: int  # Object class identifier
    class_name: str  # Object class name
    mask: Optional[np.ndarray] = None  # Segmentation mask

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion k1
    k2: float = 0.0  # Radial distortion k2
    p1: float = 0.0  # Tangential distortion p1
    p2: float = 0.0  # Tangential distortion p2

class FeatureExtractor:
    """Extract visual features from images"""

    def __init__(self, method: str = "orb", max_features: int = 1000):
        self.method = method
        self.max_features = max_features

        if method == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif method == "akaze":
            self.detector = cv2.AKAZE_create()

    def extract_features(self, image: np.ndarray) -> List[ImageFeature]:
        """Extract features from an image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if keypoints is None or descriptors is None:
            return []

        # Convert to our format
        features = []
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            feature = ImageFeature(
                point=np.array([kp.pt[0], kp.pt[1]]),
                descriptor=desc,
                id=i,
                scale=kp.size
            )
            features.append(feature)

        return features[:self.max_features]

    def match_features(self, features1: List[ImageFeature],
                      features2: List[ImageFeature]) -> List[Tuple[int, int]]:
        """Match features between two sets"""
        if not features1 or not features2:
            return []

        # Extract descriptors
        desc1 = np.array([f.descriptor for f in features1])
        desc2 = np.array([f.descriptor for f in features2])

        # Use brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING if self.method == "orb" else cv2.NORM_L2)

        matches = bf.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))

        return good_matches

class ObjectDetector:
    """Object detection for robotics applications"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = self._load_model()
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def _load_model(self):
        """Load object detection model (simplified)"""
        # In a real implementation, this would load a trained model
        # For this example, we'll create a simple mock detector
        return "mock_model"

    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[ObjectDetection]:
        """Detect objects in an image"""
        # In a real implementation, this would run the detection model
        # For this example, we'll simulate detection results

        # Convert image to suitable format if needed
        if len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape

        # Simulate some detections
        detections = []

        # Add a few mock detections
        mock_objects = [
            (width * 0.2, height * 0.3, width * 0.4, height * 0.6, 0.8, 0, "person"),
            (width * 0.6, height * 0.4, width * 0.8, height * 0.7, 0.9, 56, "chair"),
            (width * 0.4, height * 0.5, width * 0.6, height * 0.8, 0.7, 39, "bottle")
        ]

        for x1, y1, x2, y2, conf, class_id, class_name in mock_objects:
            if conf >= confidence_threshold:
                detection = ObjectDetection(
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)

        return detections

    def detect_and_track(self, image: np.ndarray, prev_detections: List[ObjectDetection] = None) -> List[ObjectDetection]:
        """Detect and potentially track objects across frames"""
        current_detections = self.detect_objects(image)

        # In a real implementation, this would include tracking logic
        # For this example, we'll just return the current detections
        return current_detections

class VisualServoingController:
    """Controller for visual servoing applications"""

    def __init__(self, camera_intrinsics: CameraIntrinsics,
                 control_gain: float = 1.0):
        self.camera_intrinsics = camera_intrinsics
        self.control_gain = control_gain
        self.reference_features = None
        self.current_features = None

    def set_reference_image(self, image: np.ndarray, features: List[ImageFeature]):
        """Set the reference image and features for visual servoing"""
        self.reference_features = features
        self.reference_image = image

    def compute_image_jacobian(self, point: np.ndarray) -> np.ndarray:
        """Compute image Jacobian for a 2D point"""
        # Simplified image Jacobian computation
        # This is a basic version - real implementation would be more complex
        u, v = point

        # Camera intrinsic parameters
        fx, fy = self.camera_intrinsics.fx, self.camera_intrinsics.fy
        cx, cy = self.camera_intrinsics.cx, self.camera_intrinsics.cy

        # Image Jacobian (interaction matrix)
        L = np.zeros((2, 6))
        L[0, 0] = -fx  # ∂u/∂x
        L[0, 2] = -(u - cx)  # ∂u/∂z
        L[0, 3] = -(u - cx) * (v - cy) / fy  # ∂u/∂ωx
        L[0, 4] = (fx + (u - cx) * (u - cx) / fx)  # ∂u/∂ωy
        L[0, 5] = -(v - cy)  # ∂u/∂ωz

        L[1, 1] = -fy  # ∂v/∂y
        L[1, 2] = -(v - cy)  # ∂v/∂z
        L[1, 3] = -(fy + (v - cy) * (v - cy) / fy)  # ∂v/∂ωx
        L[1, 4] = (u - cx) * (v - cy) / fx  # ∂v/∂ωy
        L[1, 5] = (u - cx)  # ∂v/∂ωz

        return L

    def compute_control(self, current_features: List[ImageFeature]) -> np.ndarray:
        """Compute control velocities based on feature errors"""
        if not self.reference_features or not current_features:
            return np.zeros(6)  # No motion if no features

        # Match features between reference and current
        extractor = FeatureExtractor()
        matches = extractor.match_features(self.reference_features, current_features)

        if not matches:
            return np.zeros(6)

        # Compute error for matched features
        errors = []
        jacobians = []

        for ref_idx, curr_idx in matches[:10]:  # Limit to first 10 matches
            if ref_idx < len(self.reference_features) and curr_idx < len(current_features):
                ref_pt = self.reference_features[ref_idx].point
                curr_pt = current_features[curr_idx].point

                error = curr_pt - ref_pt
                jacobian = self.compute_image_jacobian(ref_pt)

                errors.append(error)
                jacobians.append(jacobian)

        if not errors:
            return np.zeros(6)

        # Stack errors and jacobians
        E = np.hstack(errors)  # Error vector
        J = np.vstack(jacobians)  # Stacked Jacobians

        # Compute control using pseudo-inverse
        try:
            # Regularized control law
            lambda_reg = 0.01
            control_vel = -self.control_gain * np.linalg.pinv(J.T @ J + lambda_reg * np.eye(6)) @ J.T @ E
        except np.linalg.LinAlgError:
            # Fallback to damped least squares if matrix is singular
            control_vel = np.zeros(6)

        # Limit control velocities
        max_vel = 0.1  # 10 cm/s
        control_vel = np.clip(control_vel, -max_vel, max_vel)

        return control_vel

class SLAMSystem:
    """Basic SLAM system for visual mapping and localization"""

    def __init__(self, camera_intrinsics: CameraIntrinsics):
        self.camera_intrinsics = camera_intrinsics
        self.map_points = {}  # 3D points in global map
        self.keyframes = []  # Camera poses
        self.current_pose = np.eye(4)  # Current camera pose
        self.feature_extractor = FeatureExtractor()
        self.min_triangulation_angle = 10  # degrees

    def triangulate_point(self,
                         pose1: np.ndarray,
                         pose2: np.ndarray,
                         point1: np.ndarray,
                         point2: np.ndarray) -> Optional[np.ndarray]:
        """Triangulate 3D point from two camera views"""
        # Camera intrinsic matrix
        K = np.array([
            [self.camera_intrinsics.fx, 0, self.camera_intrinsics.cx],
            [0, self.camera_intrinsics.fy, self.camera_intrinsics.cy],
            [0, 0, 1]
        ])

        # Convert poses to projection matrices
        P1 = K @ pose1[:3, :]  # First camera
        P2 = K @ pose2[:3, :]  # Second camera

        # Linear triangulation (using SVD)
        A = np.zeros((4, 4))
        A[0, :] = point1[0] * P1[2, :] - P1[0, :]
        A[1, :] = point1[1] * P1[2, :] - P1[1, :]
        A[2, :] = point2[0] * P2[2, :] - P2[0, :]
        A[3, :] = point2[1] * P2[2, :] - P2[1, :]

        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[3]  # Normalize

        if X[3] != 0:
            X = X[:3] / X[3]
            return X
        else:
            return None

    def add_keyframe(self, image: np.ndarray, pose: np.ndarray):
        """Add a new keyframe to the map"""
        features = self.feature_extractor.extract_features(image)

        keyframe = {
            'image': image,
            'pose': pose,
            'features': features,
            'timestamp': time.time()
        }

        self.keyframes.append(keyframe)

    def estimate_motion(self,
                       image1: np.ndarray,
                       image2: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate relative motion between two images"""
        # Extract features from both images
        features1 = self.feature_extractor.extract_features(image1)
        features2 = self.feature_extractor.extract_features(image2)

        # Match features
        matches = self.feature_extractor.match_features(features1, features2)

        if len(matches) < 8:  # Minimum for pose estimation
            return np.eye(4), 0.0  # No motion

        # Get matched points
        pts1 = np.float32([features1[m[0]].point for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([features2[m[1]].point for m in matches]).reshape(-1, 1, 2)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2,
                                      np.array([[self.camera_intrinsics.fx, 0, self.camera_intrinsics.cx],
                                               [0, self.camera_intrinsics.fy, self.camera_intrinsics.cy],
                                               [0, 0, 1]]),
                                      method=cv2.RANSAC, threshold=1.0)

        if E is None:
            return np.eye(4), 0.0

        # Recover pose from essential matrix
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        # Calculate motion magnitude
        motion_magnitude = np.linalg.norm(t) + np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        return T, motion_magnitude

    def build_map(self, images: List[np.ndarray]) -> Dict:
        """Build a simple map from a sequence of images"""
        if len(images) < 2:
            return {'points': [], 'poses': []}

        # Add first keyframe
        self.add_keyframe(images[0], np.eye(4))

        # Process remaining images
        current_pose = np.eye(4)
        all_poses = [current_pose.copy()]

        for i in range(1, len(images)):
            # Estimate motion
            motion, motion_mag = self.estimate_motion(images[i-1], images[i])

            if motion_mag > 0.01:  # Only add if significant motion
                current_pose = current_pose @ motion
                self.add_keyframe(images[i], current_pose)
                all_poses.append(current_pose.copy())

        # Return map information
        return {
            'num_keyframes': len(self.keyframes),
            'total_poses': len(all_poses),
            'final_pose': current_pose,
            'estimated_trajectory_length': len(all_poses)
        }

def demonstrate_vision_systems():
    """Demonstrate computer vision concepts"""
    print("Computer Vision and Perception - Chapter 10")
    print("=" * 50)

    # Initialize camera intrinsics (typical for robot head camera)
    camera_intrinsics = CameraIntrinsics(
        fx=525.0,  # Focal length in pixels
        fy=525.0,
        cx=319.5,  # Principal point
        cy=239.5
    )

    print("1. Feature Extraction Demo:")
    # Create a sample image (in practice, this would come from a camera)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some artificial features (simulated)
    cv2.rectangle(sample_image, (100, 100), (200, 200), (255, 0, 0), 2)
    cv2.circle(sample_image, (300, 300), 20, (0, 255, 0), 2)

    # Extract features using different methods
    orb_extractor = FeatureExtractor(method="orb", max_features=50)
    orb_features = orb_extractor.extract_features(sample_image)
    print(f"   - ORB features extracted: {len(orb_features)}")

    # Object detection demo
    print("\n2. Object Detection Demo:")
    detector = ObjectDetector()
    detections = detector.detect_objects(sample_image)
    print(f"   - Objects detected: {len(detections)}")
    for det in detections[:3]:  # Show first 3 detections
        bbox_str = f"[{det.bbox[0]:.1f}, {det.bbox[1]:.1f}, {det.bbox[2]:.1f}, {det.bbox[3]:.1f}]"
        print(f"     - {det.class_name}: {det.confidence:.2f}, bbox: {bbox_str}")

    # Visual servoing demo
    print("\n3. Visual Servoing Demo:")
    servo_controller = VisualServoingController(camera_intrinsics)

    # Simulate reference and current features
    ref_features = [
        ImageFeature(point=np.array([100, 100]), descriptor=np.random.rand(32), id=0),
        ImageFeature(point=np.array([200, 150]), descriptor=np.random.rand(32), id=1),
        ImageFeature(point=np.array([150, 200]), descriptor=np.random.rand(32), id=2)
    ]

    curr_features = [
        ImageFeature(point=np.array([105, 105]), descriptor=np.random.rand(32), id=0),  # Slightly moved
        ImageFeature(point=np.array([205, 155]), descriptor=np.random.rand(32), id=1),  # Slightly moved
        ImageFeature(point=np.array([155, 205]), descriptor=np.random.rand(32), id=2)   # Slightly moved
    ]

    servo_controller.set_reference_image(sample_image, ref_features)
    control_vel = servo_controller.compute_control(curr_features)
    print(f"   - Computed control velocity: [{control_vel[0]:.3f}, {control_vel[1]:.3f}, {control_vel[2]:.3f}, "
          f"{control_vel[3]:.3f}, {control_vel[4]:.3f}, {control_vel[5]:.3f}]")

    # SLAM demo
    print("\n4. SLAM Demo:")
    slam_system = SLAMSystem(camera_intrinsics)

    # Simulate a sequence of images (in practice, these would be captured from camera)
    simulated_images = []
    for i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add distinctive features
        cv2.rectangle(img, (50 + i*20, 50 + i*10), (100 + i*20, 100 + i*10), (255, 255, 0), 2)
        simulated_images.append(img)

    slam_result = slam_system.build_map(simulated_images)
    print(f"   - SLAM results:")
    print(f"     - Keyframes created: {slam_result['num_keyframes']}")
    print(f"     - Total poses: {slam_result['total_poses']}")
    print(f"     - Final position: [{slam_result['final_pose'][0,3]:.3f}, "
          f"{slam_result['final_pose'][1,3]:.3f}, {slam_result['final_pose'][2,3]:.3f}]")

    # Performance analysis
    print("\n5. Performance Analysis:")

    # Feature extraction performance
    start_time = time.time()
    for _ in range(10):
        orb_extractor.extract_features(sample_image)
    end_time = time.time()
    avg_feature_time = (end_time - start_time) / 10
    print(f"   - Average feature extraction time: {avg_feature_time:.4f}s ({1/avg_feature_time:.1f} Hz)")

    # Object detection performance
    start_time = time.time()
    for _ in range(5):
        detector.detect_objects(sample_image)
    end_time = time.time()
    avg_detection_time = (end_time - start_time) / 5
    print(f"   - Average detection time: {avg_detection_time:.4f}s ({1/avg_detection_time:.1f} Hz)")

    # Memory usage estimation
    import sys
    image_memory = sys.getsizeof(sample_image.tobytes())
    print(f"   - Sample image memory: {image_memory / 1024:.1f} KB")

    print("\n6. Multi-modal Integration:")
    print("   - Vision can be integrated with:")
    print("     - IMU for visual-inertial odometry")
    print("     - LiDAR for dense mapping")
    print("     - Tactile sensors for active exploration")
    print("     - Audio for scene understanding")

    return {
        'features': len(orb_features),
        'detections': len(detections),
        'control_velocity': control_vel,
        'slam_results': slam_result
    }

def analyze_vision_performance(results: Dict) -> Dict:
    """Analyze computer vision performance metrics"""
    analysis = {
        'detection_performance': {
            'objects_detected': results['detections'],
            'detection_rate': 'Simulated'  # Would be calculated from real data
        },
        'feature_performance': {
            'features_extracted': results['features'],
            'feature_density': results['features'] / (640 * 480) * 10000  # per 10,000 pixels
        },
        'control_performance': {
            'max_control_magnitude': np.max(np.abs(results['control_velocity'])),
            'control_dimensions': len(results['control_velocity'])
        },
        'mapping_performance': {
            'keyframes_created': results['slam_results']['num_keyframes'],
            'trajectory_length': results['slam_results']['estimated_trajectory_length']
        }
    }

    return analysis

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_vision_systems()

    # Analyze performance
    performance_analysis = analyze_vision_performance(results)

    print(f"\n7. Performance Analysis Summary:")
    for category, metrics in performance_analysis.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"     - {metric.replace('_', ' ')}: {value}")

    print(f"\n8. Vision System Capabilities:")
    print("   - Real-time object detection and recognition")
    print("   - Visual servoing for precise control")
    print("   - SLAM for mapping and localization")
    print("   - Multi-modal sensor integration")
    print("   - Robust feature extraction and matching")

    print(f"\nComputer Vision and Perception - Chapter 10 Complete!")
```

## Exercises

1. Implement a simple SLAM system using ORB features and essential matrix decomposition.

2. Design a visual servoing controller that can guide a robot arm to grasp an object.

3. Create an object detection pipeline that works with depth information for 3D object localization.

## Summary

This chapter provided a comprehensive overview of computer vision and perception systems for humanoid robots, covering object detection, scene understanding, visual servoing, SLAM systems, and multi-modal integration. We explored mathematical foundations, practical implementations, and the integration of vision with other sensory modalities. The concepts and code examples presented will help in developing robust vision systems for humanoid robots that can perceive and interact with their environment effectively.