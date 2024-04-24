import os
import imageio
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.color import label2rgb


def process_image(im, image_name):
    # Canny edge detection
    grayscale = rgb2gray(im)
    edges = canny(grayscale)
    thick_edges = dilation(dilation(edges))
    segmentation = ndi.binary_fill_holes(thick_edges)
    labels = label(segmentation)
    # Image.fromarray(np.uint8(label2rgb(labels, bg_label=0) * 255)).show()

    #
    regions = regionprops(labels)
    panels = []
    def overlap_area(a, b):
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx>=0) and (dy>=0):
            return dx*dy
        return 0

    for region in regions:
        for i, panel in enumerate(panels):
            overlap = overlap_area(region.bbox, panel)
            total_area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]) + (panel[2] - panel[0]) * (panel[3] - panel[1]) - overlap
            if overlap / total_area > 0.5:  # adjust this threshold as needed
                panels[i] = (
                    min(panel[0], region.bbox[0]),
                    min(panel[1], region.bbox[1]),
                    max(panel[2], region.bbox[2]),
                    max(panel[3], region.bbox[3])
                )
                break
        else:
            panels.append(region.bbox)

    panels = [bbox for bbox in panels if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0.05 * im.shape[0] * im.shape[1]]
    #

    panel_img = np.zeros_like(labels)
    for i, bbox in enumerate(panels, start=1):
        panel_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = i
    rgb_img = label2rgb(panel_img.astype(np.uint8), bg_label=0)
    # Image.fromarray((rgb_img * 255).astype(np.uint8)).show()


    def are_bboxes_aligned(a, b, axis):
        return (
            a[0 + axis] < b[2 + axis] and
            b[0 + axis] < a[2 + axis]
        )
    def cluster_bboxes(bboxes, axis=0):
        clusters = []
        for bbox in bboxes:
            for cluster in clusters:
                if any(are_bboxes_aligned(b, bbox, axis=axis) for b in cluster):
                    cluster.append(bbox)
                    break
            else:
                clusters.append([bbox])
        return clusters
    clusters = cluster_bboxes(panels)
    img = Image.fromarray(im)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Load default font
    def flatten(l):
        for el in l:
            if isinstance(el, list):
                yield from flatten(el)
            else:
                yield el
    for i, bbox in enumerate(flatten(clusters), start=1):
        text_size = draw.textsize(str(i), font=font)
        w, h = text_size
        x = (bbox[1] + bbox[3] - w) / 2
        y = (bbox[0] + bbox[2] - h) / 2
        draw.text((x, y), str(i), (255, 215, 0), font=font)
    os.makedirs('panels', exist_ok=True)
    for i, bbox in enumerate(flatten(clusters)):
        panel = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        Image.fromarray(panel).save(f'panels/{image_name}_{i}.png')

# Get a list of all files in the directory
image_dir = 'imgs'
image_files = os.listdir(image_dir)

# Process each image file
for image_file in image_files:
    # Check if the file is an image
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        im = imageio.v2.imread(os.path.join(image_dir, image_file))
        image_name = os.path.splitext(image_file)[0]  # get the name of the image file without extension
        process_image(im, image_name)