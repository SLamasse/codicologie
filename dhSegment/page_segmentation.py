import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize

import tensorflow as tf
import tensorflow.compat.v1 as tfc

import dh_segment
from dh_segment.inference import LoadedModel
from dh_segment.io import PAGE
from dh_segment.post_processing import thresholding, cleaning_binary, cleaning_probs
from dh_segment.post_processing import boxes_detection
from dh_segment.post_processing import hysteresis_thresholding
from dh_segment.post_processing import line_vectorization


def get_page_coords(list_raw_images, model_page):
    '''
    List[] * dh_segment.inference.loader.LoadedModel -> List[] * List[]
    Returns a tuple of a list of treated images and a list of pages
    coordinates
    '''
    list_images = []
    list_page_coords = []

    for image in list_raw_images:
        img = io.imread(image)
        list_images.append(img)
        output_page = model_page.predict(image)
        page_probs = output_page['probs'][0,:,:,1]
        page_mask = thresholding(page_probs, threshold=0.7) # binarisation de l'image ; threshold : [0,1]
        page_mask = cleaning_binary(page_mask, kernel_size=7).astype(np.uint8)*255
        page_coords = boxes_detection.find_boxes(resize(page_mask, img.shape[:2]).astype(np.uint8), n_max_boxes=1)
        list_page_coords.append(page_coords)
    
    return list_images, list_page_coords


def get_page_info(list_images, list_page_coords):
    '''
    List[] * List[] -> List[]
    Returns
    '''
    list_PAGE_info = []

    for img, page_coords in zip(list_images, list_page_coords):
        PAGE_info = PAGE.Page(image_width=img.shape[1], image_height=img.shape[0],
                              page_border=PAGE.Border(PAGE.Point.list_to_point(list(page_coords))))
        list_PAGE_info.append(PAGE_info)
    
    for img, PAGE_info in zip(list_images, list_PAGE_info):
        plot_img = img.copy()
        PAGE_info.draw_page_border(plot_img, autoscale=True, fill=False, thickness=15)
    
    return list_PAGE_info


def get_textlines_probs(list_raw_images, model_textline):
    '''
    List[] * dh_segment.inference.loader.LoadedModel -> List[]
    Returns
    '''
    list_output_textline = []
    for image in list_raw_images:
        output_textline = model_textline.predict(image)
        list_output_textline.append(output_textline)
    
    list_textline_probs = []
    for output_textline in list_output_textline:
        textline_probs = output_textline['probs'][0,:,:,1]
        list_textline_probs.append(textline_probs)
    
    return list_textline_probs


def get_textlines_mask(list_raw_images, model_page, model_textline):
    '''
    List[] * dh_segment.inference.loader.LoadedModel * dh_segment.inference.loader.LoadedModel -> List[]
    Returns
    '''
    list_PAGE_info = get_page_info(list_raw_images, model_page)
    list_textline_probs = get_textlines_probs(list_raw_images, model_textline)
    list_textline_mask = []

    for textline_probs, PAGE_info in zip(list_textline_probs, list_PAGE_info):
        textline_probs2 = cleaning_probs(textline_probs, 2)
        extracted_page_mask = np.zeros(textline_probs.shape, dtype=np.uint8)
        PAGE_info.draw_page_border(extracted_page_mask, color=(255,))
        textline_mask = hysteresis_thresholding(textline_probs2, low_threshold=0.4, high_threshold=0.6,
                                                candidates_mask=extracted_page_mask>0)
        list_textline_mask.append(textline_mask)
    
    return list_textline_mask


def get_lines(list_raw_images, list_images, model_page, model_textline):
    '''
    List[] * List[] * dh_segment.inference.loader.LoadedModel * dh_segment.inference.loader.LoadedModel -> List[]
    Returns
    '''
    list_textline_mask = get_textlines_mask(list_raw_images, model_page, model_textline)
    list_lines = []

    for textline_mask, img in zip(list_textline_mask, list_images):
        lines = line_vectorization.find_lines(resize(textline_mask, img.shape[:2]))
        list_lines.append(lines)

    return list_lines


def determine_textregions(list_PAGE_info, list_raw_images, list_images, model_page, model_textline):
    '''
    List[] * List[] * List[] * dh_segment.inference.loader.LoadedModel * dh_segment.inference.loader.LoadedModel -> None
    ...
    '''
    list_lines = get_lines(list_raw_images, list_images, model_page, model_textline)

    for lines, PAGE_info in zip(list_lines, list_PAGE_info):
        text_region = PAGE.TextRegion()
        text_region.text_lines = [PAGE.TextLine.from_array(line) for line in lines]
        PAGE_info.text_regions.append(text_region)


def plot_pages(list_images, list_PAGE_info):
    '''
    List[] * List[] -> None
    ...
    '''
    for img, PAGE_info in zip(list_images, list_PAGE_info):
        plot_img = img.copy()
        PAGE_info.draw_page_border(plot_img, autoscale=True, fill=False, thickness=15)
        PAGE_info.draw_lines(plot_img, autoscale=True, fill=False, thickness=5, color=(0,255,0))
        plt.figure(figsize=(15,15))
        plt.imshow(plot_img)


def page_to_dict(list_PAGE_info):
    '''
    List[] -> List[Dict[]]
    Returns a list of dictionary with
    '''
    list_PAGE_dict = []

    for PAGE_info in list_PAGE_info:
        PAGE_dict = PAGE_info.to_json()
        list_PAGE_dict.append(PAGE_dict)

    return list_PAGE_dict


def main_segmentation(path_ark, path_models, logfile_path):
    '''
    str * str * str -> None
    '''
    with open(logfile_path, 'a') as logfile:

        sess1 = tfc.InteractiveSession()
        with sess1.graph.as_default():
            model_page = LoadedModel(path_models)

        model_dir = 'page_model/export'
        if not os.path.exists(model_dir):
            model_dir = 'model/'
        assert(os.path.exists(model_dir))
        sess1 = tfc.InteractiveSession()
        model_page = LoadedModel(model_dir, predict_mode='filename')

        sess2 =tfc.InteractiveSession(graph=tf.Graph())
        with sess2.graph.as_default():
            model_textline = LoadedModel("/content/polylines/")

        list_raw_images = [page for page in os.listdir(path_ark) if not os.path.isdir(path_ark + page)]
        list_images, list_page_coords = get_page_coords(list_raw_images, model_page)
        list_PAGE_info = get_page_info(list_images, list_page_coords)
        determine_textregions(list_PAGE_info, list_raw_images, list_images, model_page, model_textline)
        list_PAGE_dict = page_to_dict(list_PAGE_info)

    return list_PAGE_dict