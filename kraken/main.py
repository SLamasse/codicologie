import os
import withkraken as k
import alto_to_csv  as a
import pandas as pd


def list_rep(rootdir):
  list_dir = []
  for rep in os.listdir(rootdir) :
    d = os.path.join(rootdir + "/" + rep)
    if os.path.isdir(d):
        list_dir.append(rep)
  return list_dir





if __name__ == "__main__":

    path_image = "../img"
    path_out_alto = "resAlto"
    #On fabrique toutes les reconnaissances de toutes les pages
    k.segmentation_alto(path_image,path_out_alto)

    #on traite tous les xml alto pour en extraire un tableau par manuscrit qui sera das "resultats"
    path_resultat = "resultats"

    for elt in list_rep(path_out_alto) :
        data = dict()
        chemin = path_out_alto + "/" + elt + "/"
        a.extract_data_manuscript(chemin, data)
        df = pd.DataFrame(data)
        outfile = path_resultat + "/" + elt + ".csv"
        df.to_csv(outfile, index=False)
        print(df)



