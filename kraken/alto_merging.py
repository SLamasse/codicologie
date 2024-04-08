import pandas as pd
import os
import csv
import re


def convert_in_int(num):
    '''
    str -> int
    '''
    if num == '':
        return 0
    return int(num)


def get_info(info_file):
    '''
    str -> Dict[str, List[Tuple[str, str, str, int, str, str, int]]]
    '''
    dict_info = dict()
    info_csv = open(info_file, 'r', encoding='utf-8')
    info_reader = csv.reader(info_csv, delimiter=',')
    next(info_reader, None)
    for row in info_reader:
        city = str(row[0])
        library = str(row[1])
        pressmark = str(row[2])
        order = convert_in_int(row[3])
        theme = str(row[4])
        title = str(row[5])
        folio = convert_in_int(row[6])
        ark = str(row[7])
        if dict_info == dict():
            dict_info = {ark: [(city, library, pressmark, order, theme, title, 2 * folio - 1)]}
        else:
            if ark not in dict_info:
                dict_info[ark] = [(city, library, pressmark, order, theme, title, 2 * folio - 1)]
            else:
                dict_info[ark].append((city, library, pressmark, order, theme, title, 2 * folio - 1))
    info_csv.close()
    return dict_info


def add_info(df, ark, info):
    '''
    pd.DataFrame * Dict[str, List[Tuple[str, str, str, int, str, str, int]]] -> pd.DataFrame
    '''
    j = 0
    for i in range(df.shape[0]):
        df.loc[i,'ark'] = ark
        if ark in info:
            city, library, pressmark, order, theme, title, page = info[ark][j]
            df.loc[i,'city'] = city
            df.loc[i,'library'] = library
            df.loc[i,'pressmark'] = pressmark
            df.loc[i,'order_in_ms'] = order
            df.loc[i,'theme'] = theme
            df.loc[i,'title'] = title
            if i >= page and j < len(info[ark]) - 1:
                j += 1
        else:
            df.loc[i,'city'] = ''
            df.loc[i,'library'] = ''
            df.loc[i,'pressmark'] = ''
            df.loc[i,'order_in_ms'] = 1
            df.loc[i,'theme'] = ''
            df.loc[i,'title'] = ''
    return df


def main_merging(path):
    '''
    str -> None
    '''
    # Get extra information about ark
    info_file = path + 'tab_mss_test.csv'
    info =get_info(info_file)
    # Add extra information to layout information
    csv_dir = 'resultats_csv/'
    csv_list = [csv_file for csv_file in os.listdir(path + csv_dir) if os.path.isfile(path + csv_dir + csv_file)]
    ark_list = [re.search('[\w_]+', csv_file).group(0) for csv_file in csv_list]
    switch = True
    for ark, file in zip(ark_list, csv_list):
        if switch:
            df_res = pd.read_csv(path + csv_dir + file, encoding='utf-8')
            df_res = add_info(df_res, ark, info)
            switch = False
        else:
            df = pd.read_csv(path + csv_dir + file, encoding='utf-8')
            df = add_info(df, ark, info)
            df_res = pd.concat([df_res, df])
    df_res.to_csv(path + 'data_all_mss_with_theme2.csv', encoding='utf-8')



path = '___PATH___'
main_merging(path)
