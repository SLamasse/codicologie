from PIL import Image
from get_page import page_number_to_str
import numpy as np
import os



def blackest(a):
    '''
    np.array -> int
    Returns the index of the darkest column of a
    if the result is not in the middle of the array
    returns the exact middle
    '''
    i = 0
    moy = 257.0
    imin = -1
    at = np.transpose(a)
    for l in at:
        lmoy = np.mean(l)
        if lmoy < moy:
            moy = lmoy
            imin = i
        i += 1
    l = a.shape[1]
    if imin > l // 2 - l * 0.1 and imin < l // 2 + l * 0.1:
        return imin
    return l // 2


def crop(img, a, path, page_number):
    '''
    image * np.array * str * int -> None
    Crops img given its array a, its file's path and
    its page number
    '''
    path_reframed = path + "reframed/"
    if not os.path.exists(path_reframed):
        os.makedirs(path_reframed)
    im1 = img.crop((0, 0, blackest(a), a.shape[0]))
    im2 = img.crop((blackest(a), 0, a.shape[1], a.shape[0]))
    im1.save(path_reframed + "page" + page_number_to_str(page_number) + ".jpeg", "jpeg")
    im2.save(path_reframed + "page" + page_number_to_str(page_number + 1) + ".jpeg", "jpeg")


def main_crop(path, logfile_path):
    '''
    str * str -> None
    Crops all the images of manuscript pages
    in the path directory
    Updates the logfile at logfile_path and
    saves all the errors in
    '''
    with open(logfile_path, 'a') as logfile:
        list_ark = [dir for dir in os.listdir(path) if os.path.isdir(path + dir)]
        for ark in list_ark:
            try:
                nb_page = 1
                page_path = path + ark + '/'
                list_pages = [page for page in os.listdir(page_path) if not os.path.isdir(page_path + page)]
                for page in list_pages:
                    page_img = Image.open(page_path + page)
                    page_array = np.array(page_img)
                    if page_array.shape[0] < page_array.shape[1]:
                        try:
                            crop(page_img, page_array, page_path, nb_page)
                            print("file:", page, "successfully reframed as page number", nb_page, "and", nb_page + 1)
                            nb_page += 2
                        except OSError:
                            print("Error with the file or the directory during treatment of the pages", nb_page, "and",
                                  nb_page + 1, "of the", ark)
                            print("Pages", nb_page, "and", nb_page + 1, "had not been crop")
                            logfile.write("Error with the file or the directory during treatment of the pages " + str(nb_page) +
                                          " and " + str(nb_page + 1) + " of the " + ark + "\n")
                            logfile.write("\tPages " + str(nb_page) + " and " + str(nb_page + 1) + " had not been crop\n")
                            nb_page += 2
                    else:
                        try:
                            path_reframed = page_path + "reframed/"
                            if not os.path.exists(path_reframed):
                                os.makedirs(path_reframed)
                            page_img.save(page_path + "reframed/" + "page" + page_number_to_str(nb_page) + ".jpeg", "jpeg")
                            print("file:", page, "doesn't need to be reframed, set as page", nb_page)
                            nb_page += 1
                        except OSError:
                            print("Error with the file or the directory during treatment of the pages", nb_page, "of the", ark)
                            print("Pages", nb_page, "has not been crop")
                            logfile.write("Error with the file or the directory during treatment of the pages " + str(nb_page) +
                                          " of the " + ark + "\n")
                            logfile.write("\tPages " + str(nb_page) + " has not been crop\n")
                            nb_page += 1
            except OSError:
                print("Error during the treatment of the all manuscript", ark)
                logfile.write("Error during the treatment of the all manuscript" + ark + "\n")
