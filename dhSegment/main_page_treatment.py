from get_page import *
from crop_page import *
from page_segmentation import *
from page_analysis import *



# Definition de la racine
files_path = ___PATH___

# Création si nécessaire du dossier pour les logs
if not os.path.exists(files_path + "log/"):
    os.makedirs(files_path + "log/")

# Création si nécessaire du dossier pour les resultats
if not os.path.exists(files_path + "resultats/"):
    os.makedirs(files_path + "resultats/")

# Récupération des pages des manuscrits en pdf
logfile_fetch = files_path + "log/mss_not_downloaded.txt"
main_fetch_images(files_path, "test_en_ligne.txt", logfile_fetch)

# Redimensionnement des images
logfile_reframed = files_path + "log/mss_not_reframed.txt"
main_crop(files_path + "images/", logfile_reframed)

# Segmentation des pages avec dhSegment et analyse des donnees
logfile_segmentation = files_path + "log/segmentation_failed.txt"
data_dict = main_segmentation(files_path, logfile_segmentation)

# Analyse de la mise en page
logfile_analysis = files_path + "log/page_analysis_failed.txt"
main_page_analysis(files_path, data_dict, logfile_analysis)
