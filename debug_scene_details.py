
import sionna.rt
import numpy as np

print("Loading scene with merge_shapes=False...")
try:
    scene = sionna.rt.load_scene("Terrains/test_scene.xml", merge_shapes=False)
except Exception as e:
    print(f"Error loading scene: {e}")
    exit(1)

print(f"Total objects found: {len(scene.objects)}")
print(f"First 10 object names: {list(scene.objects.keys())[:10]}")

# Find bounds again
all_min = np.array([float('inf')]*3)
all_max = np.array([float('-inf')]*3)

for name, obj in scene.objects.items():
    pos = obj.position.numpy()[0]
    all_min = np.minimum(all_min, pos)
    all_max = np.maximum(all_max, pos)

print("-" * 50)
print(f"Global Position Bounds:")
print(f"Min: {all_min}")
print(f"Max: {all_max}")
print(f"Center: {(all_min + all_max)/2}")

# Let's try to find a building
buildings = [name for name in scene.objects.keys() if "Yurt" in name or "Building" in name]
print(f"Found {len(buildings)} buildings/Yurts.")
if buildings:
    print(f"Sample building '{buildings[0]}' position: {scene.objects[buildings[0]].position.numpy()[0]}")
