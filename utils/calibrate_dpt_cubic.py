import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

CALIBRATION_IMAGE_PATH = "../videos/stripes.png" 


CALIBRATION_POINTS = [
    # --- Tiles (163 cm range) ---
    (567, 495, 163),
    (488, 496, 163),
    (647, 491, 163),
    
    # # --- Tiles (193 cm range) ---
    (496, 474, 193),
    (564, 473, 193),
    (633, 470, 193),
    
    # --- Yellow Markings @ (590 cm range) ---
    (389, 367, 590),
    (410, 367, 590),
    (455, 366, 590),
    (481, 364, 590),
    (527, 361, 590),
    (550, 360, 590),
    (597, 357, 590),
    (621, 354, 590),
    (670, 350, 590),
    
    # --- Yellow Markings @ (150 cm range) ---
    (427, 486, 150),
    (509, 480, 150),
    (679, 476, 150),
    # (760, 464, 150),
    # (296, 479, 150),
    
    # --- Yellow Markings @ (300 cm range) ---
    (625, 395, 300),
    (536, 400, 300),
    (492, 401, 300),
    # (797, 376, 300),
    # (863, 366, 300),
    # (895, 361, 300),
    (413, 403, 300),
    # (373, 404, 300),
    # (313, 403, 300),
    # (289, 399, 300)
]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform


def get_coordinate_depth(depth_map, x, y, window_size=5):
    """Extracts the median depth around a specific (x, y) coordinate."""
    x, y = int(x), int(y)
    half_w = window_size // 2
    h, w = depth_map.shape
    
    y1, y2 = max(0, y - half_w), min(h, y + half_w + 1)
    x1, x2 = max(0, x - half_w), min(w, x + half_w + 1)
    
    coordinate_patch = depth_map[y1:y2, x1:x2]
    return float(np.median(coordinate_patch)) if coordinate_patch.size > 0 else 0.0

# Load image
img = cv2.imread(CALIBRATION_IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image at {CALIBRATION_IMAGE_PATH}")

# Convert BGR to RGB (MiDaS transforms require RGB numpy arrays)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply the native transform pipeline and move tensor to target device
input_batch = transform(rgb_img).to(device)

print("Processing image through MiDaS DPT_Hybrid...")
with torch.no_grad():
    prediction = midas(input_batch)
    
    # Resample depth prediction back to match the original image frame dimensions
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    depth_map = prediction.cpu().numpy()

# Extract raw depth values for each marked physical distance
extracted_depths = []
actual_distances = []

for x_px, y_px, true_dist in CALIBRATION_POINTS:
    raw_depth = get_coordinate_depth(depth_map, x_px, y_px)
    extracted_depths.append(raw_depth)
    actual_distances.append(true_dist)

X = np.array(extracted_depths)
y = np.array(actual_distances)

# Compute 3rd-order polynomial regression coefficients (y = a3*x^3 + a2*x^2 + a1*x + a0)
coefficients = np.polyfit(X, y, 3)
a3, a2, a1, a0 = coefficients

print("\n" + "="*50)
print("coefficientst:")
print(f"a3 = {a3:.8f}")
print(f"a2 = {a2:.8f}")
print(f"a1 = {a1:.8f}")
print(f"a0 = {a0:.8f}")
print("="*50)



poly_function = np.poly1d(coefficients)
x_line = np.linspace(min(X), max(X), 100)
y_line = poly_function(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', s=50, label='Your Marked Image Points')
plt.plot(x_line, y_line, color='blue', linewidth=2, label='3rd-Order Regression Fit')
plt.xlabel('Raw MiDaS Depth Output')
plt.ylabel('Real Distance (cm)')
plt.title('DPT_Hybrid Calibration Curve Verification')
plt.legend()
plt.gca().invert_xaxis()  # MiDaS values drop as object distance increases
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()