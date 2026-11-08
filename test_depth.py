from transformers import pipeline
from PIL import Image

print("Loading depth model...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

print("Running on image...")
image = Image.open("bigbus.jpg")  # tumhari wahi image
result = depth_pipe(image)

result["depth"].save("depth_output.png")
print("Depth map saved as depth_output.png")