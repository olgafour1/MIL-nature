import cv2
import numpy as np
from openslide import OpenSlideUnsupportedFormatError, OpenSlide
import xml.etree.ElementTree as ET
from PIL import Image

def read_wsi(wsi_path):
    try:
        img_name = "_".join(wsi_path.split("/")[-1].split("_")[:3])
        wsi_image = OpenSlide(wsi_path)
        level_used = wsi_image.level_count - 1
        rgba_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                    wsi_image.level_dimensions[level_used]))
    except OpenSlideUnsupportedFormatError:
        raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

    return img_name, wsi_image, rgba_image, level_used


def get_image_open(rgba_image):
    hsv = cv2.cvtColor(rgba_image, cv2.COLOR_RGB2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return image_open


def get_bbox(cont_img, rgba_image=None):
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgba_contour = None
    if not (rgba_image is None):
        rgba_contour = rgba_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgba_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgba_contour, contours


def draw_bbox(image, bounding_boxes):
    rgba_bbox = image.copy()
    for i, bounding_box in enumerate(bounding_boxes):
        x = int(bounding_box[0])
        y = int(bounding_box[1])
        cv2.rectangle(rgba_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
                      thickness=2)
    return rgba_bbox


def hematoxylin_eosin_aug(image, low=0.7, high=1.3, seed=None):
    """
    "Quantification of histochemical staining by color deconvolution"
    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf
    Performs random hematoxylin-eosin augmentation
    """
    D = np.array([[1.88, -0.07, -0.60],
                  [-1.02, 1.13, -0.48],
                  [-0.55, -0.13, 1.57]])
    M = np.array([[0.65, 0.70, 0.29],
                  [0.07, 0.99, 0.11],
                  [0.27, 0.57, 0.78]])
    Io = 240

    h, w, c = image.shape
    OD = -np.log10((image.astype("uint16") + 1) / Io)
    C = np.dot(D, OD.reshape(h * w, c).T).T
    r = np.ones(3)
    r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
    img_aug = np.dot(C, M) * r

    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
    return img_aug


def read_xml_file(xml_file, wsi_image, rgba_image):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    Ximageorg, Yimageorg = wsi_image.dimensions
    dims = rgba_image.shape[:2]

    annotations = []

    for enum, (coord_tag, type_tag) in enumerate(
            zip(root.findall('Annotations/Annotation/Coordinates'), root.findall('Annotations/Annotation'))):
        rank = type_tag.get('PartOfGroup')
        if rank == "_0":
            annotation_group = []
            for coord in coord_tag.iter('Coordinate'):
                x_coord = float(coord.get('X'))
                x_coord = ((x_coord) * dims[1]) / Ximageorg

                y_coord = float(coord.get('Y'))
                y_coord = ((y_coord) * dims[0]) / Yimageorg

                annotation_group.append([x_coord, y_coord])
            annotations.append(np.array(annotation_group, dtype=np.int32))
    return annotations


def normalize_staining(image):
    Io = 240
    beta = 0.15
    alpha = 1
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = image.shape
    img = image.reshape(h * w, c)
    OD = -np.log((img.astype("uint16") + 1) / Io)
    ODhat = OD[(OD >= beta).all(axis=1)]

    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

    Vec = -V.T[:2][::-1].T
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = HE.T
    Y = OD.reshape(h * w, c).T

    C = np.linalg.lstsq(HE, Y, rcond=None)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

    return Inorm


def resize_image(img, mode, dim):
    if mode == "bilinear":
        mode = cv2.INTER_AREA
    elif mode == "bicubic":
        mode = cv2.INTER_CUBIC
    elif mode == "nearest-neighbor":
        mode = cv2.INTER_NEAREST
    else:
        raise ValueError("Unsupported resize mode '{}'".format(mode))

    resized = cv2.resize(img, dim, interpolation=mode)
    return resized


def find_roi_bbox(rgba_image):
    image_open = get_image_open(rgba_image)

    bounding_boxes, rgba_contour, contours = get_bbox(image_open, rgba_image=rgba_image)

    return rgba_image, bounding_boxes, rgba_contour, image_open


def get_segmented_tissue_thumbnail(wsi_path, mode, dim):
    img_name, wsi_image, rgba_image, level_used = read_wsi(wsi_path)
    image_open = get_image_open(rgba_image)

    bounding_boxes, _, contours = get_bbox(image_open, rgba_image=None)
    max_area = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            cnt = contours[i]
            max_area = area

    bounding_boxes = [cv2.boundingRect(cnt)]
    x, y, w, h = bounding_boxes[0]
    Ximageorg, Yimageorg = wsi_image.dimensions

    dims = rgba_image.shape[:2]

    x_coord = int((x * Ximageorg) / dims[1])
    y_coord = int((y * Yimageorg) / dims[0])

    thumbnail_level = wsi_image.level_count - 3
    new_dims = wsi_image.level_dimensions[thumbnail_level]
    new_width = int((w * new_dims[0]) / dims[1])
    new_height = int((h * new_dims[1]) / dims[0])

    segmented_tissue = wsi_image.read_region((x_coord, y_coord), thumbnail_level, (new_width, new_height))
    resized_image = resize_image(np.array(segmented_tissue), mode, dim)
    PIL_image = Image.fromarray(resized_image.astype('uint8'), 'RGBA')
    rgb_image = Image.new("RGB", PIL_image.size, (255, 255, 255))
    rgb_image.paste(PIL_image, mask=PIL_image.split()[3])  # 3 is the alpha channel
    norm_image = normalize_staining(np.array(rgb_image))
    return norm_image, img_name


