import re
import os
import csv
import requests
import urllib.request as url_req


def page_number_to_str(i):
    '''
    int -> str
    Returns the string which corresponds
    to i with three digits
    '''
    if i < 10:
        return "00" + str(i)
    elif i < 100:
        return "0" + str(i)
    else:
        return str(i)


def create_directory(directory_path):
    '''
    str -> None
    Makes directory given directory_path
    if not exists else makes nothing
    '''
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_number_pages(ark):
    '''
    str -> int
    Returns the number of pages of a manuscript
    given its ark identifier
    If this number of pages couldn't be found
    returns -1
    '''

    url_base = "https://gallica.bnf.fr/ark:/"
    url_ark = ark.replace('_', '/', 1)
    url = url_base + url_ark

    page = url_req.urlopen(url)

    view = re.search("vue 1/\d+", str(page.read()))
    if view:
        split_view = re.split("/", view.group(0))
        if len(split_view) == 2:
            return int(split_view[1])
    
    return -1


def download_image_file(url, destination_path):
    '''
    str * str -> None
    Downloads the image file at the adress url
    in the destination_path directory
    '''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def getfile_ark_and_write(ark, logfile, direct):
    '''
    str * str * str -> None
    Saves the images files of the manuscript ark
    '''
    try:
        path = os.path.join(direct, ark)
        if os.path.exists(path):
            print(f"Le répertoire {ark} existe déjà.")
        else:
            create_directory(path)
            to_ark = ark.replace("_", "/", 1)
            for i in range(1, get_number_pages(ark) + 1):
                url = "https://gallica.bnf.fr/ark:/" + to_ark + "/f" + str(i) + ".item/.jpeg"
                image_name = ark + "page" + page_number_to_str(i)
                destination_path = os.path.join(path, f'{image_name}.jpeg')
                download_image_file(url, destination_path)
                print(f"La page {i} du manuscrit {ark} a été téléchargé avec succès.")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du traitement de l'ARK {ark}: {e}")
        logfile.write(f"{ark}\n")


def main_fetch_images(chemin, src, logfile_path):
    '''
    str * str * str -> None
    Saves all the images files of the pages of the
    manuscripts in src given a path chemin and the file src
    Updates the logfile at logfile_path and
    saves all the errors in
    '''
    with open(logfile_path, 'a') as logfile:
        with open(os.path.join(chemin, src), 'r') as csvfile:
            already_seen = set()
            path_images = chemin + "images/"
            if not os.path.exists(path_images):
                os.makedirs(path_images)
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                try:
                    if row[0] not in already_seen:
                        if row[1] != "0" and row[1] != "aucun lien vers une numérisation":
                            # manipulation un peu idiote liée au fichier initiale
                            ark = re.sub("^https:\/\/gallica\.bnf\.fr\/ark:\/(.+)\/", "\\1_", row[1])
                            getfile_ark_and_write(ark, logfile, path_images)
                        else:
                            print(f"Le manuscrit {row[0]} dont l'ARK est {row[1]} n'a pas pu être téléchargé")
                            logfile.write(f"{row[1]}\n")
                        already_seen.add(row[0])
                except OSError:
                    print("erreur dans le traitement du manuscrit", row[0])
        logfile.close()
