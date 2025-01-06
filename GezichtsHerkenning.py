import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
from skimage.measure import label   

def load_image(image_path):
    image = mpimg.imread(image_path)
    return image

def to_grayscale(image):
    if image.ndim == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image

def otsu_threshold(image):
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
    total = image.size
    current_max = 0
    threshold = 0
    sum_total, sum_background = np.sum(hist * np.arange(256)), 0
    weight_background, weight_foreground = 0, 0

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between_variance > current_max:
            current_max = between_variance
            threshold = i

    return threshold / 255

def create_disk_kernel(radius):
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    mask = (x - radius)**2 + (y - radius)**2 <= radius**2
    kernel = np.zeros((size, size))
    kernel[mask] = 1
    return kernel / kernel.sum()

def normalize(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr

def plot_image(image):
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()
    
def apply_otsu(image):
    threshold = otsu_threshold(image)
    print("Applying OTSU with threshold: ", threshold.__round__(2))
    return (image > threshold).astype(np.float32)

def erode(image, disk_size=9):
    segmentation = image > otsu_threshold(image)
    small_kernel = create_disk_kernel(disk_size)
        
    result = binary_erosion(segmentation, structure=small_kernel)
    
    return result

def dilate(image):
    small_kernel = create_disk_kernel(1)
    
    result = binary_dilation(image, structure=small_kernel)
    
    return result

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 0
    
    y, x = np.where(largestCC)
    
    return x[0], y[0]

def grow_search(image, background, itterations = 1000):
    last_overlap = 0
    mask = image * background
    for i in range(itterations):
        mask = dilate(mask)
        mask = mask * image
        if mask.sum() == last_overlap:
            break
        last_overlap = mask.sum()
        if i % 25 == 0:
            print("Itteration: ", i, "Overlap: ", last_overlap)
            plt.axis('off')
            plt.imshow(mask, cmap='gray')
            plt.savefig('images/overlap/overlap' + str(i) + '.png', bbox_inches='tight')
            
    result = image ^ mask
    largestCC = getLargestCC(result)
    return (largestCC[0], largestCC[1])

def main():
    image = load_image('images/makkelijk.jpg')
    grayscale_image = to_grayscale(image)
    normalised_image = normalize(grayscale_image)
    img = erode(normalised_image, disk_size=16)    
    background = erode(normalised_image, disk_size=50)
    
    overlap = grow_search(img, background)
    
    x = overlap[0].item()
    y = overlap[1].item()
    
    fig, ax = plt.subplots()
    box_size = 100
    rectangle = patches.Rectangle((x-box_size/2, y-box_size/2), box_size, box_size, linewidth=2, edgecolor='r', facecolor='none')
    
    ax.add_patch(rectangle)
    ax.imshow(image)
    plt.show()
    
    
if __name__ == '__main__':
    main()