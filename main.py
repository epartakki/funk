from PIL import Image
import numpy as np

def add_noise(image):
    noise_intensity = np.random.uniform(0.02, 0.1)
    noise = np.random.normal(0, 255 * noise_intensity, image.shape)
    return image + noise

def color_shift(image):
    shift_value = np.random.randint(-50, 50)
    shift_channel = np.random.randint(0, 3)
    image[:, :, shift_channel] = np.clip(image[:, :, shift_channel] + shift_value, 0, 255)
    return image

def wave_distortion(image):
    x_indices = np.tile(np.arange(image.shape[1]), (image.shape[0], 1))
    y_wave = np.sin(2 * np.pi * x_indices / np.random.uniform(50, 150))
    y_indices = (np.arange(image.shape[0])[:, None] + y_wave * np.random.uniform(5, 20)).astype(np.int32)
    y_indices = np.clip(y_indices, 0, image.shape[0] - 1)
    return image[y_indices, np.arange(image.shape[1])]

def funky_filter(image):
    image = image.astype(np.float32)

    # Apply original transformations
    brightness_factor = np.random.uniform(0.5, 1.5)
    contrast_factor = np.random.uniform(0.5, 1.5)
    image = image * brightness_factor
    image = (image - image.mean()) * contrast_factor + image.mean()

    # Apply new transformations
    image = add_noise(image)
    image = color_shift(image)
    image = wave_distortion(image)

    return np.clip(image, 0, 255).astype(np.uint8)

# Load and apply the filter
image = Image.open('input.jpg')
image = np.array(image)
funky_image = funky_filter(image)

# Output final image
output_image = Image.fromarray(funky_image)
output_image.save('output.jpg')
