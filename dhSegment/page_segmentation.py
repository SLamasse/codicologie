import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

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



def organise_models_dirs(root, logfile):
    '''
    str * file -> None
    Organises the directory for good use
    of the models
    '''
    try:
        path_dhs = root + "dhsegment/"
        if not os.path.exists(path_dhs):
            os.makedirs(path_dhs)
    except:
        print("Error during the creation of the dhsegment directory")
        logfile.write("Error during the creation of the dhsegment directory\n")
        return
    
    try:
        path_model = path_dhs + "model/"
        if not os.path.exists(path_model):
            shutil.move(root + "model/", path_model)
    except:
        print("No directory with model at", root)
        logfile.write("No directory with model at " + root + "\n")
    
    try:
        path_pl = path_dhs + "polylines/"
        if not os.path.exists(path_pl):
            shutil.move(root + "polylines/", path_pl)
    except:
        print("No directory with polyline model at", root)
        logfile.write("No directory with polyline model at " + root + "\n")


def close_tf_session(session, logfile):
    '''
    tf.session * file -> None
    '''
    try:
        session.close()
        print("Closing of the tensorflow session")
    except:
        print("Closing of the tensorflow session has failed")
        logfile.write("Closing of the tensorflow session has failed\n")


def open_model_page_session(path_model, logfile):
    '''
    str * file -> Optional[Tuple[]]
    '''
    try:
        sess = tfc.InteractiveSession()
        with sess.graph.as_default():
            model_page = LoadedModel(path_model)
        print("Opening of a tensorflow session for the page model")
        return sess, model_page
    except:
        print("Opening of a tensorflow session for the page model has failed")
        logfile.write("Opening of a tensorflow session for the page model has failed\n")
        return None


def open_model_polylines_session(path_polylines, logfile):
    '''
    str * file -> Optional[Tuple[]]
    '''
    try:
        sess = tfc.InteractiveSession(graph=tf.Graph())
        with sess.graph.as_default():
            model_textline = LoadedModel(path_polylines)
        print("Opening of a tensorflow session for the polylines model")
        return sess, model_textline
    except:
        print("Opening of a tensorflow session for the polylines model has failed")
        logfile.write("Opening of a tensorflow session for the polylines model has failed\n")
        return None


def get_page_coords(image, model_page):
    '''
    str * model -> Tuple[]
    '''
    img = io.imread(image)
    output_page = model_page.predict(image)
    page_probs = output_page['probs'][0,:,:,1]
    page_mask = thresholding(page_probs, threshold=0.7) # binarisation de l'image ; threshold : [0,1]
    page_mask = cleaning_binary(page_mask, kernel_size=7).astype(np.uint8)*255
    page_coords = boxes_detection.find_boxes(resize(page_mask, img.shape[:2]).astype(np.uint8), n_max_boxes=1)
    return img, page_coords


def get_page_coords_ms(ark, list_raw_images, model_page, logfile):
    '''
    List[str] -> Tuple[List[], List[]]
    '''
    i = 1
    list_images = []
    list_page_coords = []
    for image in list_raw_images:
        try:
            img, page_coords = get_page_coords(image, model_page)
            list_images.append(img)
            list_page_coords.append(page_coords)
            i += 1
        except:
            print("Failure of the acquisition of page coordinates for", ark, "page", i)
            logfile.write("Failure of the acquisition of page coordinates for " + ark + " page " + str(i) + "\n")
            i += 1
    return list_images, list_page_coords


def get_page_coords_corpus(path, list_ark, model_page, logfile):
    '''
    str * List[str] * model * file -> Dict[str, List[]]
    '''
    corpus = dict()
    for ark in list_ark:
        path_ark = path + "images/" + ark + "/reframed/"
        list_raw_images = [path_ark + page for page in os.listdir(path_ark) if not os.path.isdir(path_ark + page)]
        list_images, list_page_coords = get_page_coords_ms(ark, list_raw_images, model_page, logfile)
        if corpus == dict():
            corpus = {ark: {'list_raw_images': list_raw_images,
                            'list_images': list_images,
                            'list_page_coords': list_page_coords}}
        else:
            corpus[ark] = {'list_raw_images': list_raw_images,
                           'list_images': list_images,
                           'list_page_coords': list_page_coords}
    return corpus


def get_page_info(img, page_coords):
    '''
    image * coords -> PAGE_info
    '''
    return PAGE.Page(image_width=img.shape[1], image_height=img.shape[0],
                     page_border=PAGE.Border(PAGE.Point.list_to_point(list(page_coords))))


def get_page_info_ms(ark, list_images, list_page_coords, logfile):
    '''
    str * List[] * List[] * file -> List[]
    '''
    list_PAGE_info = []
    i = 1
    for img, page_coords in zip(list_images, list_page_coords):
        try:
            PAGE_info =  get_page_info(img, page_coords)
            list_PAGE_info.append(PAGE_info)
            i += 1
        except:
            print("Failure during the acquisition of PAGE_info for", ark, "page", i)
            logfile.write("Failure during the acquisition of PAGE_info for " + ark + " page " + str(i))
            i += 1
    return list_PAGE_info


def get_page_info_corpus(corpus, logfile):
    '''
    Dict[str, Dict[]] * file -> None
    '''
    for ark in corpus:
        corpus[ark]['list_PAGE_info'] = get_page_info_ms(ark,
                                                         corpus[ark]['list_images'],
                                                         corpus[ark]['list_page_coords'],
                                                         logfile)
        

def get_output_textline(model_textline, image):
    '''
    model * image ->
    '''
    return model_textline.predict(image)


def get_output_textline_ms(ark, list_raw_images, model_textline, logfile):
    '''
    '''
    i = 1
    list_output_textline = []
    for image in list_raw_images:
        try:
            output_textline = get_output_textline(model_textline, image)
            list_output_textline.append(output_textline)
            i += 1
        except:
            print("Failure during the acquisition of output_textline of", ark, "page", i)
            logfile.write("Failure during the acquisition of output_textline of " + ark + " page " + str(i))
            i += 1
    return list_output_textline


def get_output_textline_corpus(corpus, model_textline, logfile):
    '''
    '''
    for ark in corpus:
        corpus[ark]['list_output_textline'] = get_output_textline_ms(ark,
                                                                     corpus[ark]['list_raw_images'],
                                                                     model_textline,
                                                                     logfile)


def get_textline_probs(output_textline):
    '''
    '''
    return output_textline['probs'][0,:,:,1]


def get_textline_probs_ms(ark, list_output_textline, logfile):
    '''
    '''
    i = 1
    list_textline_probs = []
    for output_textline in list_output_textline:
        try:
            textline_probs = get_textline_probs(output_textline)
            list_textline_probs.append(textline_probs)
            i += 1
        except:
            print("Failure during the acquisition of textline_probs of", ark, "page", i)
            logfile.write("Failure during the acquisition of textline_probs of " + ark + " page " + str(i) + "\n")
            i += 1
    return list_textline_probs


def get_textline_probs_corpus(corpus, logfile):
    '''
    '''
    for ark in corpus:
        corpus[ark]['list_textline_probs'] = get_textline_probs_ms(ark,
                                                                    corpus[ark]['list_output_textline'],
                                                                    logfile)


def get_textline_mask(textline_probs, PAGE_info):
    '''
    '''
    textline_probs2 = cleaning_probs(textline_probs, 2)
    extracted_page_mask = np.zeros(textline_probs.shape, dtype=np.uint8)
    PAGE_info.draw_page_border(extracted_page_mask, color=(255,))
    textline_mask = hysteresis_thresholding(textline_probs2, low_threshold=0.4, high_threshold=0.6,
                                            candidates_mask=extracted_page_mask>0)
    return textline_mask


def get_textline_mask_ms(ark, list_textline_probs, list_PAGE_info, logfile):
    '''
    '''
    i = 1
    list_textline_mask = []
    for textline_probs, PAGE_info in zip(list_textline_probs, list_PAGE_info):
        try:
            textline_mask = get_textline_mask(textline_probs, PAGE_info)
            list_textline_mask.append(textline_mask)
            i += 1
        except:
            print("Failure during the acquisition of textline_mask of", ark, "page", i)
            logfile.write("Failure during the acquisition of textline_mask of " + ark + " page " + str(i) + "\n")
            i += 1
    return list_textline_mask


def get_textline_mask_corpus(corpus, logfile):
    '''
    '''
    for ark in corpus:
        corpus[ark]['list_textline_mask'] = get_textline_mask_ms(ark,
                                                                 corpus[ark]['list_textline_probs'],
                                                                 corpus[ark]['list_PAGE_info'],
                                                                 logfile)


def get_lines(textline_mask, img):
    '''
    '''
    return line_vectorization.find_lines(resize(textline_mask, img.shape[:2]))


def get_lines_ms(ark, list_textline_mask, list_images, logfile):
    '''
    '''
    i = 1
    list_lines = []
    for textline_mask, img in zip(list_textline_mask, list_images):
        try:
            lines = get_lines(textline_mask, img)
            list_lines.append(lines)
            i += 1
        except:
            print("Failure during the acquisition of lines of", ark, "page", i)
            logfile.write("Failure during the acquisition of lines of " + ark + " page " + str(i) + "\n")
            i += 1
    return list_lines


def get_lines_corpus(corpus, logfile):
    '''
    '''
    for ark in corpus:
        corpus[ark]['list_lines'] = get_lines_ms(ark,
                                                 corpus[ark]['list_textline_mask'],
                                                 corpus[ark]['list_images'],
                                                 logfile)


def get_text_region(lines, PAGE_info):
    '''
    '''
    text_region = PAGE.TextRegion()
    text_region.text_lines = [PAGE.TextLine.from_array(line) for line in lines]
    PAGE_info.text_regions.append(text_region)


def get_text_region_ms(ark, list_lines, list_PAGE_info, logfile):
    '''
    '''
    i = 1
    for lines, PAGE_info in zip(list_lines, list_PAGE_info):
        try:
            get_text_region(lines, PAGE_info)
            i += 1
        except:
            print("Failure during the acquisition of text_region of", ark, "page", i)
            logfile.write("Failure during the acquisition of text_region of " + ark + " page " + str(i) + "\n")
            i += 1


def get_text_region_corpus(corpus, logfile):
    '''
    '''
    for ark in corpus:
        get_text_region_ms(ark,
                           corpus[ark]['list_lines'],
                           corpus[ark]['list_PAGE_info'],
                           logfile)
        

def page_to_dict_ms(ark, list_PAGE_info, logfile):
    '''
    List[] -> List[Dict[]]
    Returns a list of dictionary with
    '''
    i = 1
    list_PAGE_dict = []
    for PAGE_info in list_PAGE_info:
        try:
            PAGE_dict = PAGE_info.to_json()
            list_PAGE_dict.append(PAGE_dict)
            i += 1
        except:
            print("Failure during the convertion of PAGE object into dictionary for", ark, "page", i)
            logfile.write("Failure during the convertion of PAGE object into dictionary for " + ark + " page " + str(i) + "\n")
            i += 1
    return list_PAGE_dict


def page_to_dict_corpus(corpus, logfile):
    '''
    '''
    data = dict()
    for ark in corpus:
        if data == dict():
            data = {ark: page_to_dict_ms(ark, corpus[ark]['list_PAGE_info'], logfile)}
        else:
            data[ark] = page_to_dict_ms(ark, corpus[ark]['list_PAGE_info'], logfile)
    return data


def main_segmentation(path, logfile_path):
    '''
    str * str ->
    '''
    with open(logfile_path, 'a') as logfile:
        organise_models_dirs(path, logfile)
        path_dhs = path + "dhsegment/"
        list_ark = [ark for ark in os.listdir(path + "images/") if os.path.isdir(path + "images/") and ark != '.ipynb_checkpoints']

        # Opening of the tensorflow session for the page model
        sess, model_page = open_model_page_session(path_dhs + "model/", logfile)

        corpus = get_page_coords_corpus(path, list_ark, model_page, logfile)
        get_page_info_corpus(corpus, logfile)

        # Closing of the tensorflow session for the page model
        close_tf_session(sess, logfile)

        # Opening of the tensorflow session for the polylines model
        sess, model_textline = open_model_polylines_session(path_dhs + "polylines/", logfile)

        get_output_textline_corpus(corpus, model_textline,logfile)
        get_textline_probs_corpus(corpus, logfile)
        get_textline_mask_corpus(corpus, logfile)
        get_lines_corpus(corpus, logfile)
        get_text_region_corpus(corpus, logfile)

        # Closing of the tensorflow session for the polylines model
        close_tf_session(sess, logfile)

        #print(corpus)
        data_res = page_to_dict_corpus(corpus, logfile)

        logfile.close()

        return data_res
