import pandas as pd
import os
import re



def get_page_number(folio):
    '''
    str -> int
    Returns the page number given the folio
    If the folio notation is not of the form number-side (ex: 43r)
    '''
    fm = re.search('\d+[rv]{0,1}', folio)
    if fm:
        f = fm.group(0)
        num = re.search('\d+', f)
        side = re.search('[rv]', f)
        if not side or side.group(0) == 'r':
            return int(num.group(0)) * 2 - 1
        else:
            return int(num.group(0)) * 2
    else:
        print('Error during the recognition of the folio')
        return -1


def read_folio(folio):
    '''
    str -> List[Tuple[int,int]]
    Returns a list of page range given a foliotation
    Returns (page_number, -1) if the text goes until the next text
    Returns (-1, -1) if there is an error during the process
    '''
    folio = str(folio)
    coma = re.search(',', folio)
    if coma:
        lb = []
        lrs = re.split(',', folio)
        for fol in lrs:
            f = read_folio(fol)
            lb = lb + f
        return lb
    else:
        be = re.search('\d+[rv]{0,1}-\d+[rv]{0,1}', folio)
        if be:
            r = re.split('-', be.group(0))
            b = get_page_number(r[0])
            e = get_page_number(r[1])
            return [(b, e)]
        else:
            b = re.search('\d+[rv]{0,1}', folio)
            if b:
                bp = get_page_number(b.group(0))
                return [(bp, -1)]
            else:
                print('Error with the form of the folio')
                return [(-1, -1)]


def set_page_beginning(lb):
    '''
    List[Tuple[int,int]] -> List[int]
    Returns the list of the page of begging of a text
    '''
    if len(lb) == 1:
        pb, _ = lb[0]
        return [pb]
    else:
        pb, _ = lb[0]
        return [pb] + set_page_beginning(lb[1:])


def get_df_page_beginning(df):
    '''
    pd.DataFrame -> pd.DataFrame
    Returns a dataframe get from df after the add of
    page_debut column and the filtering of useless columns
    '''
    col_ndf = df.columns.tolist()
    ndf = pd.DataFrame(columns=col_ndf)
    j = 0
    for i in range(df.shape[0]):
        lbp = set_page_beginning(read_folio(df.loc[i,'Folio_no']))
        for bp in lbp:
            df_aux = pd.DataFrame(df.iloc[i,:]).T
            if ndf.empty:
                ndf = df_aux
            else:
                ndf = pd.concat([ndf, df_aux], ignore_index=True)
            ndf.loc[j,'page_debut'] = bp
            j += 1
    ndf = ndf.drop(columns=['Folio_no', 'feuillet_deb', 'feuillet_fin'])
    ndf = ndf.sort_values(['ark', 'page_debut']).reset_index()
    return ndf


def get_nb_pages_mss(path):
    '''
    str -> Dict[str,int]
    Returns a dictionary which associates the ark of a
    manuscript and its number of pages
    '''
    res = dict()
    lark = [ark for ark in os.listdir(path) if os.path.isdir(path + ark)]
    for ark in lark:
        nb_pages = len([page for page in os.listdir(path + ark) if not os.path.isdir(path + ark + page)])
        if res == dict():
            res = {ark: nb_pages}
        else:
            res[ark] = nb_pages
    return res


def get_df_page_ending(df, ark_nb_pages):
    '''
    pd.DataFrame * Dict[str,int] -> pd.DataFrame
    Returns a dataframe get from df which the column
    page_fin has been added thanks to the dictionary
    ark_nb_page
    '''
    nb_raws = df.shape[0]
    for i in range(nb_raws):
        if df.loc[i, 'page_debut'] != -1:
            ark = df.loc[i,'ark']
            if i+1 < nb_raws and ark == df.loc[i+1,'ark']:
                df.loc[i,'page_fin'] = df.loc[i+1,'page_debut']
            else:
                if ark in ark_nb_pages:
                    df.loc[i,'page_fin'] = ark_nb_pages[ark]
                else:
                    df.loc[i,'page_fin'] = -1
    return df


def merge_dfs(df_page, df_info):
    '''
    pd.DataFrame * pd.DataFrame -> None
    Adds the informations over theme, the pressmark, the library
    of preservation and its location, and the position in the
    manuscript in the dataframe df_page
    '''
    print(df_info.loc[:,'page_fin'])
    for i in range(df_page.shape[0]):
        ark = df_page.loc[i,'ark']
        num_page = float(df_page.loc[i,'num_page'])
        mask = (df_info['ark'] == ark) & (df_info['page_debut'] >= num_page) & (df_info['page_fin'] < num_page)
        new_info = df_info.loc[mask,:]
        if not new_info.empty:
            new_info.reset_index(drop=True, inplace=True)
            df_page.loc[i,'city'] = new_info.loc[0,'City']
            df_page.loc[i,'library'] = new_info.loc[0,'Library']
            df_page.loc[i,'pressmark'] = new_info.loc[0,'Pressmark']
            df_page.loc[i,'theme'] = new_info.loc[0,'type_traite']
            df_page.loc[i,'title'] = new_info.loc[0,'Title_ms']
            df_page.loc[i,'order_in_ms'] = new_info.loc[0,'numOrdre']
        else:
            mask = (df_info['ark'] == ark)
            new_info = df_info.loc[mask,:]
            print(new_info)
            if not new_info.empty:
                new_info.reset_index(drop=True, inplace=True)
                df_page.loc[i,'city'] = new_info.loc[0,'City']
                df_page.loc[i,'library'] = new_info.loc[0,'Library']
                df_page.loc[i,'pressmark'] = new_info.loc[0,'Pressmark']
                df_page.loc[i,'theme'] = new_info.loc[0,'type_traite']
                df_page.loc[i,'title'] = new_info.loc[0,'Title_ms']
                df_page.loc[i,'order_in_ms'] = new_info.loc[0,'numOrdre']


def main_merging(path, data_file_name, info_file_name):
    '''
    str * str * str -> None
    Adds the thematic informations to the data
    over page layout given a folder path and the
    file name of the file which contains the data
    '''
    exit_file = path + "data_all_mss_with_theme.csv"
    path_pages = path + 'resultats_xml/'
    data_file_path = path + data_file_name
    info_file_path = path + info_file_name
    df_page = pd.read_csv(data_file_path, encoding='utf-8')
    df_info = pd.read_csv(info_file_path, encoding='utf-8')
    df_info = get_df_page_beginning(df_info)
    df_info = get_df_page_ending(df_info, path_pages)
    merge_dfs(df_page, df_info)
    df_page.to_csv(exit_file, encoding='utf-8')


main_merging('C:/Users/lebec/Documents/Article_reseau_factures_texte/alto_version/', 'data_all_mss.csv', 'tab_mss.csv')
