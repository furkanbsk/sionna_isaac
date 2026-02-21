
import sionna.rt
import numpy as np

# Load the scene
scene = sionna.rt.load_scene("Terrains/test_scene.xml")

# Initialize min/max coordinates
min_coords = np.array([float('inf'), float('inf'), float('inf')])
max_coords = np.array([float('-inf'), float('-inf'), float('-inf')])

print(f"Analyzing {len(scene.objects)} objects...")

# Iterate through objects to find the bounding box
for name, obj in scene.objects.items():
    # Some objects might not have a loaded geometry yet if not accessed, 
    # but Sionna usually loads them. 
    # We can use the object's position as a rough estimate if vertices aren't accessible easily
    # without deeper API digging, but let's try to see if we can get vertices.
    # Actually, a safer way in a script is just to look at positions.
    
    pos = obj.position.numpy()[0]
    min_coords = np.minimum(min_coords, pos)
    max_coords = np.maximum(max_coords, pos)

print("-" * 30)
print(f"Scene Bounding Box Center: {(min_coords + max_coords) / 2}")
print(f"Scene Extents (Size): {max_coords - min_coords}")
print("-" * 30)

center = (min_coords + max_coords) / 2
size = max_coords - min_coords
max_dim = np.max(size)

# Suggest a camera position
# Place camera "max_dim" away from the center, looking at the center
cam_pos = center + np.array([max_dim, max_dim, max_dim]) * 0.8 

print("SUGGESTED CAMERA SETTINGS:")
print(f"position={list(cam_pos)}")
print(f"look_at={list(center)}")
print("-" * 30)
