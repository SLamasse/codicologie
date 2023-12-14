'''
Nom du fichier: get_pdf_from_corpus.py
Auteurs: Stéphane Lamassé et Pierre Lebec 
Date de création: 9 décembre 2023
Description: Le script est destiné à télécharger des fichiers PDF de Gallica 
(la bibliothèque numérique de la Bibliothèque nationale de France), 
basé sur les ARK des manuscrits fournis dans un fichier CSV dans le répertoire Corpus. 
Les fichiers PDF sont ensuite divisés en pages individuelles, et les informations sur 
les opérations effectuées (ou les erreurs rencontrées) sont enregistrées dans un fichier journal.
fonctions définies :
    - create_directory_if_not_exists(directory_path): Crée un répertoire s'il n'existe pas déjà.
    - download_file(url, destination_path): Télécharge un fichier depuis une URL donnée et le sauvegarde localement.
    - split_pdf(input_pdf, path): Divise un fichier PDF en pages individuelles.
    - getfile_ark_and_write(ark, logfile, direct): Télécharge un fichier PDF depuis une URL basée sur l'ARK, 
    le divise en pages, et enregistre le fichier résultant. 
    Gère également les erreurs et écrit les ARK ayant échoué dans un fichier journal.
'''
import os
import re
import csv
import requests
from PyPDF2 import PdfFileWriter
import PyPDF2

def create_directory_if_not_exists(directory_path):
    '''
    str -> None
    Makes directory given directory_path
    if not exists else makes nothing
    '''
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)

def download_file(url, destination_path):
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

def split_pdf(input_pdf, path):
    # Ouvrir le PDF en mode binaire
    with open(input_pdf, 'rb') as pdf_file:
        # Créer un objet de lecteur PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        # Parcourir chaque page et enregistrer individuellement
        for page_num in range(len(pdf_reader.pages)):
            # on enlève pages présentation bnf
                # Créer un objet
                pdf_writer = PyPDF2.PdfWriter()
                # Ajouter la page actuelle au nouveau PDF
                pdf_writer.add_page(pdf_reader.pages[page_num])
                # Enregistrer la page individuelle
                output_pdf = path + "/" + f'page_{page_num + 1}.pdf'
                with open(output_pdf, 'wb') as output_file:
                    pdf_writer.write(output_file)
                    print(f'Page {page_num + 1} enregistrée sous {output_pdf}')
    os.remove(input_pdf)


def getfile_ark_and_write(ark, logfile, direct):
    try:
        path = os.path.join(direct, ark)
        if os.path.isdir(path):
            print(f"Le répertoire {ark} existe déjà.")
        else:
            create_directory_if_not_exists(path)
            to_ark = ark.replace("_", "/", 1)
            url = "https://gallica.bnf.fr/ark:/" + to_ark + ".pdf"
            destination_path = os.path.join(path, f'{ark}.pdf')
            download_file(url, destination_path)
            print(f"Le fichier {ark} a été téléchargé avec succès.")
            # Diviser le PDF en pages
            split_pdf(path + "/" + ark + ".pdf", path)
#            os.remove(destination_path + ark + ".pdf")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du traitement de l'ARK {ark}: {e}")
        logfile.write(f"{ark}\n")

def main():
    chemin = "./Corpus/"
    logfile_path = "./log/mms_not_download.txt"

    with open(logfile_path, 'a') as logfile:
        with open(os.path.join(chemin, "liste_mss_num_ark.txt"), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if row[1] != "0":
                    # manipulation un peu idiote liée au fichier initiale
                    ark = re.sub("^https:\/\/gallica\.bnf\.fr\/ark:\/(.+)\/", "\\1_", row[1])
                    getfile_ark_and_write(ark, logfile, chemin)
                else:
                    print(f"Le manuscrit {row[0]} dont l'ARK est {row[1]} n'a pas pu être téléchargé")
                    logfile.write(f"{row[1]}\n")

if __name__ == "__main__":
    main()
