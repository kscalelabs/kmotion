import sys

import bvhio
import matplotlib.pyplot as plt

# Replace with your actual BVH file path
bvh_path = "tests/assets/walk-relaxed_287304/walk-relaxed_loop_251148.bvh"


# Load hierarchy
root = bvhio.readAsHierarchy(bvh_path)

# Find the Hip joint (adjust the filter if needed)
hip_joints = root.filter("Hip")
if not hip_joints:
    print("WARNING: Hip joint not found.")
    sys.exit(1)
hip = hip_joints[0]

# Determine total number of frames from the Hip joint keyframes
num_frames = len(hip.Keyframes)
print(f"Total frames: {num_frames}")

# Initialize lists to store positions
frames = list(range(num_frames))
x_pos = []
y_pos = []
z_pos = []

# Iterate over each frame and record Hip joint positions
for frame in frames:
    root.loadPose(frame)
    pos = hip.PositionWorld
    x_pos.append(pos.x)
    y_pos.append(pos.y)
    z_pos.append(pos.z)

# Create three 2D plots in one figure (subplots for X, Y, and Z)
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle(f"Hip Joint Local Position Over {num_frames} Frames", fontsize=16)

axes[0].plot(frames, x_pos, label="X Position", color="blue", linewidth=2)
axes[0].set_ylabel("X Position")
axes[0].legend()
axes[0].grid(True)
axes[0].autoscale(True)

axes[1].plot(frames, y_pos, label="Y Position", color="green", linewidth=2)
axes[1].set_ylabel("Y Position")
axes[1].legend()
axes[1].grid(True)
axes[1].autoscale(True)

axes[2].plot(frames, z_pos, label="Z Position", color="red", linewidth=2)
axes[2].set_ylabel("Z Position")
axes[2].set_xlabel("Frame")
axes[2].legend()
axes[2].grid(True)
axes[2].autoscale(True)

# Add padding between subplots
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])  # rect adjusts space for the title

# Show the plot
plt.show()
