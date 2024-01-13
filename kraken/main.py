import withkraken as k
import alto_to_csv  as a
import pandas as pd


path_image = "../img"
path_out_alto = "resAlto"
#On fabrique toutes les reconnaissances de toutes les pages
k.segmentation_alto(path_image,path_out_alto)


#on traite tous les xml alto pour en extraire un tableau par manuscrit qui sera das "resultats"
path_resultat = "resultats"
rep = [f for f in os.listdir(path_out_alto) if os.path.isfile(os.path.join(path_out_alto, f))]

for elt in rep :
    data = dict()
    a.extract_data_manuscript(elt, data)
    df = pd.DataFrame(data)
    outfile = path_resultat + "/" + elt + ".csv"
    df.to_csv(outfile, index=False)
    print(df)



