import cv2
import torch
import time
from pathlib import Path
from moge.model.v2 import MoGeModel

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
model.half()

# Path to the directory containing images
image_dir = Path("../data/front_center")
image_files = sorted([f for f in image_dir.glob("*.jpg")])

total_time = 0
num_images = len(image_files)

print(f"Found {num_images} images in {image_dir}")

for idx, image_path in enumerate(image_files, 1):
    # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
    input_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Measure inference time
    start_time = time.time()
    output = model.infer(input_image)
    inference_time = time.time() - start_time
    total_time += inference_time

    print(f"Image {idx}/{num_images} ({image_path.name}): {inference_time:.3f} seconds")

# Print average inference time
avg_time = total_time / num_images
print(f"\nAverage inference time: {avg_time:.3f} seconds")



"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""