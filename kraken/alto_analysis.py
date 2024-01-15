from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np 



## FONCTIONS TRAITEMENT .XML

def alto_parse(alto, **kargs):
    '''
    Convert ALTO xml file to element tree
    '''
    try:
        xml = ET.parse(alto, **kargs)
    except ET.ParseError as e:
        print(f"Parser Error in file '{alto}': {e}")
    # Register ALTO namespaces
    # https://www.loc.gov/standards/alto/ | https://github.com/altoxml
    # alto-bnf (unofficial) BnF ALTO dialect - for further info see
    # http://bibnum.bnf.fr/alto_prod/documentation/alto_prod.html
    namespace = {
        "alto-1": "http://schema.ccs-gmbh.com/ALTO",
        "alto-1-xsd": "http://schema.ccs-gmbh.com/ALTO/alto-1-4.xsd",
        "alto-2": "http://www.loc.gov/standards/alto/ns-v2#",
        "alto-2-xsd": "https://www.loc.gov/standards/alto/alto.xsd",
        "alto-3": "http://www.loc.gov/standards/alto/ns-v3#",
        "alto-3-xsd": "http://www.loc.gov/standards/alto/v3/alto.xsd",
        "alto-4": "http://www.loc.gov/standards/alto/ns-v4#",
        "alto-4-xsd": "http://www.loc.gov/standards/alto/v4/alto.xsd",
        "alto-bnf": "http://bibnum.bnf.fr/ns/alto_prod",
    }
    # Extract namespace from document root
    if "http://" in str(xml.getroot().tag.split("}")[0].strip("{")):
        xmlns = xml.getroot().tag.split("}")[0].strip("{")
    else:
        try:
            ns = xml.getroot().attrib
            xmlns = str(ns).split(" ")[1].strip("}").strip("'")
        except IndexError:
            sys.stderr.write(
                f'\nERROR: File "{alto.name}": no namespace declaration found.'
            )
            xmlns = "no_namespace_found"
    if xmlns in namespace.values():
        return alto, xml, xmlns
    else:
        sys.stdout.write(
            f'\nERROR: File "{alto.name}": namespace {xmlns} is not registered.\n'
        )


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def group_coordinates(points, modulo):
    h =[]
    v = []
    for idx,i in enumerate(points):
        if idx % modulo == 0:
            h.append(i)
        else:
            v.append(i)
    return h,v


def extraire_data(xml, xmlns, page): 
    data = []
    pagedict = {
            "largeur_page" : float(page['WIDTH']),
            "hauteur_page" : float(page['HEIGHT'])
            }
    tmp = list()
    for textblocks in xml.findall(".//{%s}TextBlock" % xmlns):
        points = [polygon.attrib["POINTS"] for polygon in textblocks.findall(".//*[@POINTS]")][0]
        pnts = [int(x) for x in points.split()]
        x,y = group_coordinates(pnts, 2)
        # noirs ?
        tmp.append(PolyArea(x,y))
        #marge_basse = textblocks.attrib['HEIGHT']
        #marge_exterieur = textblocks.attrib['WIDTH']
        #marge_haute = textblocks.attrib['VPOS']
        #marge_interieur = textblocks.attrib['HPOS']

    areablocks = sum(tmp)
        
    for textlines in xml.findall(".//{%s}TextLine" % xmlns):
        # on récupère le polygon
        points = [polygon.attrib["POINTS"] for polygon in textlines.findall(".//*[@POINTS]")][0]
        pnts = [int(x) for x in points.split()]
        x,y = group_coordinates(pnts, 2)
        area = PolyArea(x,y)
        # résultat pour chaque ligne
        res = textlines.attrib

        # traitement des lignes 
        # Chaque ligne est une liste de points horizontale et verticale 
        baseline = res['BASELINE'].split()
        nb = len(baseline)
        debut_ligne = float(baseline[0])
        if nb > 4 : 
            fin_ligne = float(baseline[-2])
        else :
            fin_ligne = float(baseline[2])

        res.update({"début_ligne":debut_ligne})
        res.update({"fin_ligne":fin_ligne})            
        res.update({"Surface_ligne":area})
        res.update({"Surface_écrite":areablocks})
        res.update(pagedict)
        data.append(res)
    lines = pd.DataFrame.from_dict(data)
    return lines



## FONCTIONS TRAITEMENTS DONNEES

def define_side(image_file_name):
    '''
    str -> str
    Returns the side recto or verso of a page from
    its image file name
    '''
    inum = define_page_nbr(image_file_name)
    if inum % 2 == 0:
        if inum == 0:
            print("Warning page number is 0, problem during its definition.")
        return "verso"
    else:
        return "recto"
    

def define_page_nbr(image_file_name):
    '''
    str -> int
    Returns the page number given the image file name.
    '''
    try:
        page = re.search("page\d{3}", image_file_name)
        num = re.search("\d{3}", page.group(0))
        inum = int(num.group(0))
        return inum
    except:
        print("Error during the definition of the page number in define_page_number. Page number set as 0.")
        return 0


def str_to_int(df):
    '''
    pandas.DataFrame -> None
    Changes the type from string to integer for the
    columns: HEIGHT, HEIGHT, HPOS, VPOS
    '''
    df['WIDTH'] = pd.to_numeric(df['WIDTH'])
    df['HEIGHT'] = pd.to_numeric(df['HEIGHT'])
    df['HPOS'] = pd.to_numeric(df['HPOS'])
    df['VPOS'] = pd.to_numeric(df['VPOS'])


def black_space(df):
    '''
    pandas.DataFrame -> None
    Returns df with a new column called "black_space"
    which calculate the aera occupied by the line
    '''
    df['black_space'] = df['WIDTH'] * df['HEIGHT']


def mean_median_standard_deviation_width(df):
    '''
    pandas.DataFrame -> None
    Returns a tuple which contains the mean, the median
    and the standard deviation of the width
    '''
    width = np.array(df['WIDTH'])
    mean = width.mean()
    median = np.median(width)
    sd = np.std(width)
    return mean, median, sd


def filter_text_lines(df, med_width, sd_width):
    return df.loc[(df['WIDTH'] >= (med_width - sd_width)) & (df['WIDTH'] <= (med_width + sd_width)), :]


def limits_width_serie(df):
    '''
    pandas.DataFrame -> int * int
    Returns a tuple with the inferior limit and the superior limit of
    the width of a serie of lines
    '''
    hpos = np.sort(np.array(df['HPOS']))

    Q1 = np.percentile(hpos, 25, method="midpoint") # premier quartile
    Q3 = np.percentile(hpos, 75, method="midpoint") # troiseme quartile
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
    pandas.DataFrame * int -> int
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
    pandas.DataFrame * int -> bool
    Returns True if there is annotations around the text zone.
    Else, returns False
    '''
    return df.shape[0] != nb_l


def text_zone(std_lines, med_width):
    '''
    pandas.DataFrame * float -> Tuple[Tuple[int, int], Tuple[int, int]]
    Returns the coordinates of the two opposite points of
    the rectugular text zone
    '''
    # Recuperation des coordonnees du rectangle de texte
    hpos = np.array(std_lines['HPOS'])
    vpos = np.array(std_lines['VPOS'])

    # Calcul de la médiane de la hauteur des lignes
    med_haut = np.median(np.array(std_lines['HEIGHT'])) # calcul fait dans la fonction car ne sert qu'ici

    flh = hpos.min() # position horizontale du depart de la ligne la plus a gauche
    llh = int(hpos.max() + med_width) # position horizontale de la fin de la ligne commencant le plus a droite
    flv = int(vpos.min() + med_haut) # position verticale de la ligne la plus en haut
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
        prop_ext_sp = 0.0
    if prop_int_sp < 0:
        prop_int_sp = 0.0
    if prop_up_sp < 0:
        prop_up_sp = 0.0
    if prop_down_sp < 0:
        prop_down_sp = 0.0
    
    return prop_up_sp, prop_down_sp, prop_ext_sp, prop_int_sp


def proportion_black_space(std_lines):
    '''
    pandas.DataFrame -> float
    Returns the proportion of the "black" in the page
    (i.e. the space taken by the text)
    '''
    lines = std_lines['black_space']
    sum_black = 0
    for i in range(len(lines)):
        sum_black += lines[i]
    return sum_black


def proportion_black_space_alt(rect_op_pts, dim_page):
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


def abs_proportion_black_space(df):
    '''
    pandas.DataFrame -> float
    Returns the absolute proportion of the "black" in the page
    (i.e. the space take by the text and the annotations)
    '''
    lines = df['black_space']
    sum_black = 0
    for i in range(len(lines)):
        sum_black += lines[i]
    return sum_black


def dict_line(data, df, side, page_nbr):
    '''
    Dict[str, List[]] * pandas.DataFrame * str -> None
    Returns a dictionnary with all the mesurements.
    If data is an empty dictionnary it creates one.
    Else it updates the dictionnary with the values of
    all mesurements for the page wich is contained in df.
    '''
    # Variables nécessaires
    _, med_width, sd_width = mean_median_standard_deviation_width(df)
    std_lines = filter_text_lines(df, med_width, sd_width)
    rect_op_pts = text_zone(std_lines, med_width)
    dim_page = df.iloc[0,11], df.iloc[0,12]

    # Récupération des valeurs
    nb_c = nb_col(df, med_width)
    nb_l = nb_lines(std_lines, nb_c)
    annot = presence_annotations(df, nb_l)
    prop_up_sp, prop_down_sp, prop_ext_sp, prop_int_sp = edges_space_proportion(rect_op_pts, dim_page, side)
    prop_black = proportion_black_space_alt(rect_op_pts, dim_page)

    # Enregistrement des valeurs dans le dictionnaire data
    if data == dict():
        data['num_page'] = [page_nbr]
        data['nbr_columns'] = [nb_c]
        data['nbr_lines'] = [nb_l]
        data['presence_annotations'] = [annot]
        data['prop_up_space'] = [prop_up_sp]
        data['prop_down_space'] = [prop_down_sp]
        data['prop_ext_space'] = [prop_ext_sp]
        data['prop_int_space'] = [prop_int_sp]
        data['prop_black_space'] = [prop_black]
    else:
        data['num_page'].append(page_nbr)
        data['nbr_columns'].append(nb_c)
        data['nbr_lines'].append(nb_l)
        data['presence_annotations'].append(annot)
        data['prop_up_space'].append(prop_up_sp)
        data['prop_down_space'].append(prop_down_sp)
        data['prop_ext_space'].append(prop_ext_sp)
        data['prop_int_space'].append(prop_int_sp)
        data['prop_black_space'].append(prop_black)


def extract_data_manuscript(folder_path, data, image_file_name):
    '''
    str * Dict[List[]] -> None
    Returns a dataframe in which each line
    is a page of the manuscript
    '''
    list_pages= [page_file for page_file in os.listdir(folder_path) if not os.path.isdir(folder_path + page_file)]
    for page_file in list_pages:
        file = open(folder_path + page_file, "r", encoding="utf-8")
        _, xml, xmlns = alto_parse(file)
        page = xml.find(".//{%s}Page" % xmlns).attrib
        df = extraire_data(xml, xmlns, page)
        df = df.convert_dtypes()
        str_to_int(df)
        dict_line(data, df, define_side(image_file_name), define_page_nbr(image_file_name))

    

## EXECUTION DU PROGRAMME
    
folder_path = "___PATH___"
data = dict()
extract_data_manuscript(folder_path, data)
df = pd.DataFrame(data)
df.to_csv("___PATH___", index=False)
print(df)
