import os
import re
import pandas as pd


def get_dataframe_from_csv(file, ark):
    '''
    FILE * str -> pd.DataFrame
    Returns the data frame corresponding at the
    contents of the file which the column ark ID
    has been added
    '''
    df = pd.read_csv(file)
    df['ark'] = ark
    return df


def get_ark(file_name):
    '''
    str -> str
    Hypothesis: file's name is of the form ark.csv
    Returns the ark given the file's name
    '''
    match = re.match('\w+', file_name)
    if match:
        return match.group(0)
    else:
        return 'inconnu'


def main_shaping_data(path):
    '''
    str -> None
    Proceeds to the merging of all the data frames in
    a unique data frame with the mention of the ark ID
    '''
    logfile = open(path + 'log_shaping_data.txt', 'w')
    df_tot = pd.DataFrame()
    f_df = True
    data_path = path + 'resultats_csv/'
    data_files_dict = {get_ark(name): data_path + name for name in os.listdir(data_path) if not os.path.isdir(data_path + name)}
    for ark, file_path in data_files_dict.items():
        try:
            file = open(file_path, 'r', encoding='utf-8')
            df = get_dataframe_from_csv(file, ark)
            if f_df:
                df_tot = df
                f_df = False
            else:
                df_tot = pd.concat([df_tot, df], ignore_index=True)
            file.close()
        except:
            print('Error during the shaping of the data of the manuscript', ark)
            logfile.write('Error during the shaping of the data of the manuscript ' + ark + '\n')
    res_path = path + 'data_all_mss.csv'
    df_tot.to_csv(res_path, index=True, encoding='utf-8')
    logfile.close()
    


main_shaping_data('C:/Users/lebec/Documents/Article_reseau_factures_texte/alto_version/')