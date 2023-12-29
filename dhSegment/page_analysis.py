import pandas as pd
import math
import re
import numpy as np
import os

from page_segmentation import main_segmentation



def df_of_textlines_coords(PAGE_dict):
    '''
    Dict[] -> padas.DataFrame
    Returns a dataframe of lines coordinates from PAGE_dict
    '''
    text_lines = dict()

    i = 0
    j = 0
    for i in range(len(PAGE_dict['text_regions'])):
        tr = PAGE_dict['text_regions'][i]
        tl = tr['text_lines']
        for e in tl:
            for k, v in e.items():
                if k not in text_lines:
                    text_lines[k] = {j: v}
                else:
                    text_lines[k][j] = v
            j += 1

    df = pd.DataFrame(text_lines)
    df = df.drop(columns=['custom_attribute', 'baseline', 'text', 'line_group_id', 'column_group_id', 'id'])

    return df


def list_df_textlines_coords(list_PAGE_dict):
    '''
    List[Dict[]] -> List[pandas.DataFrame]
    Returns a list of dataframes of lines coordiantes
    from a list of PAGE_dict
    '''
    list_df = []

    for PAGE_dict in list_PAGE_dict:
        df = df_of_textlines_coords(PAGE_dict)
        list_df.append(df)

    return list_df


def extrema_line(pts_list):
    '''
    List[List[int]] -> Tuple[Tuple[int, int], Tuple[int, int]]
    Returns the two extrema points of the line from a list of points
    '''
    hmin = pts_list[0][0]
    hmax = pts_list[0][0]
    vmin = pts_list[0][1]
    vmax = pts_list[0][1]

    for pt in pts_list:
        if pt[0] < hmin:
            hmin = pt[0]
        if pt[0] > hmax:
            hmax = pt[0]
        if pt[1] < vmin:
            vmin = pt[1]
        if pt[1] > vmax:
            vmax = pt[1]

    return (hmin, vmin), (hmax, vmax)


def page_width(page_dict):
    '''
    Dict[] -> int
    Returns the width of the page
    '''
    borders = extrema_line(page_dict['page_border']['coords'])
    (hmin, _), (hmax, _) = borders
    return hmax - hmin


def page_height(page_dict):
    '''
    Dict[] -> int
    Returns the height of the page
    '''
    borders = extrema_line(page_dict['page_border']['coords'])
    (_, vmin), (_, vmax) = borders
    return vmax - vmin


def line_width(extrema):
    '''
    Tuple[Tuple[int, int], Tuple[int, int]] -> float
    Returns the width of the lin from its extrema
    '''
    (hmin, vmin), (hmax, vmax) = extrema
    return math.sqrt((hmax - hmin) ** 2 + (vmax - vmin) ** 2)


def horizontal_beginning(extrema):
    '''
    Tuple[Tuple[int, int], Tuple[int, int]] -> int
    Returns the horizontal position where the line starts
    '''
    (hmin, _), (_, _) = extrema
    return hmin


def vertical_barycenter(extrema):
    '''
    Tuple[Tuple[int, int], Tuple[int, int]] -> float
    Returns the vertical barycenter of a line
    '''
    (_, vmin), (_, vmax) = extrema
    return (vmin + vmax) / 2


def df_textlines_info(PAGE_dict, df):
    '''
    Dict[] * pandas.DataFrame -> pandas.DataFrame
    Returns a dataframe of lines with all calculated informations
    '''
    pwidth = page_width(PAGE_dict)
    pheight = page_height(PAGE_dict)

    extrema = {'etrema_line': dict()}
    width = {'line_width': dict()}
    hpos = {'hpos': dict()}
    vpos = {'vpos': dict()}
    pw = {'page_width': dict()}
    ph = {'page_height': dict()}

    for i in range(df.shape[0]):
        if extrema['etrema_line'] == dict():
            extrema['etrema_line'] = {i: extrema_line(df.loc[i,'coords'])}
            width['line_width'] = {i: line_width(extrema['etrema_line'][i])}
            hpos['hpos'] = {i: horizontal_beginning(extrema['etrema_line'][i])}
            vpos['vpos'] = {i: vertical_barycenter(extrema['etrema_line'][i])}
            pw['page_width'] = {i: pwidth}
            ph['page_height'] = {i: pheight}
        else:
            extrema['etrema_line'][i] = extrema_line(df.loc[i,'coords'])
            width['line_width'][i] = line_width(extrema['etrema_line'][i])
            hpos['hpos'][i] = horizontal_beginning(extrema['etrema_line'][i])
            vpos['vpos'][i] = vertical_barycenter(extrema['etrema_line'][i])
            pw['page_width'][i] = pwidth
            ph['page_height'][i]= pheight

    extrema = pd.DataFrame(extrema)
    width = pd.DataFrame(width)
    hpos = pd.DataFrame(hpos)
    vpos = pd.DataFrame(vpos)
    pw = pd.DataFrame(pw)
    ph = pd.DataFrame(ph)

    df = pd.merge(df, extrema, left_index=True, right_index=True)
    df = pd.merge(df, width, left_index=True, right_index=True)
    df = pd.merge(df, hpos, left_index=True, right_index =True)
    df = pd.merge(df, vpos, left_index=True, right_index=True)
    df = pd.merge(df, pw, left_index=True, right_index=True)
    df = pd.merge(df, ph, left_index=True, right_index=True)

    return df


def list_df_textlines_info(list_PAGE_dict, list_textlines_coords):
    '''
    List[Dict[]] * List[List[int]] -> List[pandas.DataFrame]
    Returns a list of dataframes of lines with all the calculated informations
    '''
    list_df = []
    for PAGE_dict, textlines_coords in zip(list_PAGE_dict, list_textlines_coords):
        list_df.append(df_textlines_info(PAGE_dict, textlines_coords))
    return list_df


def str_to_int(df):
    '''
    pandas.DataFrame -> None
    Changes the type from string to integer for the
    columns: HEIGHT, HEIGHT, HPOS, VPOS
    '''
    df['width'] = pd.to_numeric(df['line_width'])
    df['hpos'] = pd.to_numeric(df['hpos'])
    df['vpos'] = pd.to_numeric(df['vpos'])


def mean_median_standard_deviation_width(df):
    '''
    pandas.DataFrame -> Tuple[float, float, float]
    Returns a tuple which contains the mean, the median
    and the standard deviation of the width
    '''
    width = np.array(df['line_width'])
    mean = width.mean()
    median = np.median(width)
    sd = np.std(width)
    return mean, median, sd


def filter_text_lines(df, med_width, sd_width):
    '''
    pandas.DataFrame * float * float -> padas.DataFrame
    Returns the dataframe with the text lines which are
    not annotations
    '''
    return df.loc[(df['line_width'] >= (med_width - sd_width)) & (df['line_width'] <= (med_width + sd_width)), :]


def limits_width_serie(df):
    '''
    pandas.DataFrame -> Tuple[int, int]
    Returns a tuple with the inferior limit and the superior limit of
    the width of a serie of lines
    '''
    hpos = np.sort(np.array(df['hpos']))

    Q1 = np.percentile(hpos, 25, method="midpoint") # premier quartile
    Q3 = np.percentile(hpos, 75, method="midpoint") # troisieme quartile
    IQ = Q3 - Q1 # interquartile

    i = 0
    inf_lim = 0
    while inf_lim == 0:
        if hpos[i] >= Q1 - IQ:
            inf_lim = hpos[i] # inf_lim correspond au minimum de la serie hors valeurs aberrantes
        i+=1

    i = len(hpos) - 1
    sup_lim = 0
    while sup_lim == 0:
        if hpos[i] <= Q3 + IQ:
            sup_lim = hpos[i] # sup_lim correspond au maximum de la serie hors valeurs aberrantes
        i-=1

    return (inf_lim, sup_lim)


def nb_col(df, med_width):
    '''
    pandas.DataFrame * float -> int
    Returns the number of text columns in the page
    '''
    inf_lim, sup_lim = limits_width_serie(df)
    if inf_lim + med_width < sup_lim:
        nb_col = 2
    else:
        nb_col = 1
    return nb_col


def nb_lines(std_lines, nb_c):
    '''
    float * int -> int
    Returns the number of text lines in the page
    from a dataframe of text lines filtered
    (without any annotation lines)
    '''
    if nb_c == 2:
        return std_lines.shape[0] // 2
    else:
        return std_lines.shape[0]


def presence_annotations(df, nb_l):
    '''
    pandas.DataFrale * int -> bool
    Returns True if there is annotations around the text zone.
    Else, returns False
    '''
    return df.shape[0] != nb_l


def text_zone(std_lines, med_width):
    '''
    float * float -> Tuple[Tuple[int, int], Tuple[int, int]]
    Returns the coordinates of the two opposite points of
    the rectugular text zone
    '''
    # Recuperation des coordonnees du rectangle de texte
    hpos = np.array(std_lines['hpos'])
    vpos = np.array(std_lines['vpos'])

    flh = hpos.min() # position horizontale du depart de la ligne la plus a gauche
    llh = int(hpos.max() + med_width) # position horizontale de la fin de la ligne commencant le plus a droite
    flv = vpos.min() # position verticale de la ligne la plus en haut
    llv = vpos.max() # position verticale de la ligne la plus en bas

    return (flh, flv), (llh, llv)


def edges_space_proportion(rect_op_pts, dim_page, side):
    '''
    Tuple[Tuple[int, int], Tuple[int, int]] * Tuple[int, int] * str -> Tuple[float, float, float, float]
    Returns a tuple with the proportion of the space between
    the edges of the page and the edges of the text zone
    '''
    (flh, flv), (llh, llv) = rect_op_pts
    lp, hp = dim_page

    # Calcul de la proportion des espacements haut et bas
    prop_up_sp = 100 * (flv / hp)
    prop_down_sp = 100 * ((hp - llv) / hp)

    # Calcul de la proportion des espacements extérieur et intérieur
    if side == "recto":
        prop_ext_sp = 100 * ((lp - llh) / lp)
        prop_int_sp = 100 * (flh / lp)
    else:
        prop_ext_sp = 100 * (flh / lp)
        prop_int_sp = 100 * ((lp - llh) / lp)

    # Si une proportion négative, établissement d'une valeur nulle
    if prop_ext_sp < 0:
      prop_ext_sp = 0
    if prop_int_sp < 0:
      prop_int_sp = 0
    if prop_up_sp < 0:
      prop_up_sp = 0
    if prop_down_sp < 0:
      prop_down_sp = 0

    return prop_up_sp, prop_down_sp, prop_ext_sp, prop_int_sp


def proportion_black_space(rect_op_pts, dim_page):
    '''
    Tuple[Tuple[int, int], Tuple[int, int]] * Tuple[int, int] -> float
    Returns the proportion of the "black" in the page
    (i.e. the space taken by the text)
    '''
    (flh, flv), (llh, llv) = rect_op_pts
    lp, hp = dim_page
    rect_width = llh - flh
    rect_height = llv - flv
    return 100 * ((rect_width * rect_height) / (lp * hp))


def define_side(image_file_name):
    '''
    str -> str
    Returns the side recto or verso of a page from
    its image file name
    '''
    side = re.search("[rv]", image_file_name)
    if not side:
      return "inconnu"
    elif side.group(0) == "r":
      return "recto"
    else:
      return "verso"


def dict_line(data, df, side):
    '''
    Dict[] * pandas.DataFrame * str -> None
    Updates the dictionnary data with all the mesurements.
    If data is an empty dictionnary it creates one.
    Else it updates the dictionnary with the values of
    all mesurements for the page wich is contained in dataframe
    '''
    # Variables nécessaires
    _, med_width, sd_width = mean_median_standard_deviation_width(df)
    std_lines = filter_text_lines(df, med_width, sd_width)
    rect_op_pts = text_zone(std_lines, med_width)
    dim_page = df.loc[0,'page_width'], df.loc[0,'page_height']

    # Récupération des valeurs
    nb_c = nb_col(df, med_width)
    nb_l = nb_lines(std_lines, nb_c)
    annot = presence_annotations(df, nb_l)
    prop_up_sp, prop_down_sp, prop_ext_sp, prop_int_sp = edges_space_proportion(rect_op_pts, dim_page, side)
    prop_black = proportion_black_space(rect_op_pts, dim_page)

    # Enregistrement des valeurs dans le dictionnaire data
    if data == dict():
        data['nbr_columns'] = [nb_c]
        data['nbr_lines'] = [nb_l]
        data['presence_annotations'] = [annot]
        data['prop_up_space'] = [prop_up_sp]
        data['prop_down_space'] = [prop_down_sp]
        data['prop_ext_space'] = [prop_ext_sp]
        data['prop_int_space'] = [prop_int_sp]
        data['prop_black_space'] = [prop_black]
    else:
        data['nbr_columns'].append(nb_c)
        data['nbr_lines'].append(nb_l)
        data['presence_annotations'].append(annot)
        data['prop_up_space'].append(prop_up_sp)
        data['prop_down_space'].append(prop_down_sp)
        data['prop_ext_space'].append(prop_ext_sp)
        data['prop_int_space'].append(prop_int_sp)
        data['prop_black_space'].append(prop_black)


def df_pages(list_page_lines_info, list_raw_images):
    '''
    List[] * List[] -> pandas.DataFrame
    Returns the dataframe of the pages from the list
    of dataframes which contains the calculted informations
    over the page's lines
    '''
    data = dict()
    for page_lines_info, image in zip(list_page_lines_info, list_raw_images):
        dict_line(data, page_lines_info, define_side(image))
    data_page = pd.DataFrame(data)
    return data_page


def page_data_analysis(path, ark, list_PAGE_dict, list_raw_images):
    '''
    str * List[Dict[]] * List -> None
    '''
    name = path + ark + "_page_data.csv"
    list_textlines_coords = list_df_textlines_coords(list_PAGE_dict)
    list_page_lines_info = list_df_textlines_info(list_PAGE_dict, list_textlines_coords)
    pages = df_pages(list_page_lines_info, list_raw_images)
    pages.to_csv(name, sep=';', header=True, index=True, encoding='utf-8')


def main_page_analsis(path, path_models, logfile_analysis, logfile_segmentation):
    '''
    str * str * str * str -> None
    '''
    with open(logfile_analysis, 'a') as logfile:
        path_df = path + "pages_data/"
        list_ark = [ark for ark in os.listdir(path) if os.path.isdir(path + ark)]
        for ark in list_ark:
            path_ark = path + ark + "reframed/"
            list_raw_images = [page for page in os.listdir(path_ark) if not os.path.isdir(path_ark + page)]
            list_PAGE_dict = main_segmentation(path_ark, path_models, logfile_segmentation)
            page_data_analysis(path_df, ark, list_PAGE_dict, list_raw_images)