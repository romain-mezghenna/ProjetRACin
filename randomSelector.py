import random

# Function to select a number of images based on scene length
def select_random_images(scene, images):
    scene_length = scene[1] - scene[0]
    # Determine the number of images to select based on the scene length
    num_images = max(1, min(6, scene_length // 60))  # Adjust the divisor to control selection range
    return random.sample(images, num_images)

