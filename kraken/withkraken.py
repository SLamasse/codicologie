import os
from PIL import Image
# pageseg is the library from pagesegmentation
from kraken import binarization, pageseg
from kraken import blla
from kraken.lib import vgsl
from kraken import serialization


#model_path = '/usr/local/lib/python3.10/dist-packages/kraken/blla.mlmodel'
#model_path = 'model/blla_ft_lectau.mlmodel'
model_path = 'model//blla.mlmodel'



def list_rep(rootdir):
  list_dir = []
  for rep in os.listdir(rootdir):
    d = os.path.join(rootdir + "/" + rep)
    if os.path.isdir(d):
        list_dir.append(rep)
  return list_dir



def segmentation_alto(path_image,path_out_alto):
    for elt in list_rep(path_image):
        try:
            thepath = os.path.join(path_image + "/" + elt)
            tmp_dir = thepath + "/reframed"
            fichiers = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
            i = 0
            for pg in fichiers:
                image_filename = tmp_dir + "/" + pg
                im=Image.open(image_filename)
                bw_im = binarization.nlbin(im)
                seg = pageseg.segment(bw_im)
                seg_model = vgsl.TorchVGSLModel.load_model(model_path)
                baseline_seg = blla.segment(im, model = seg_model)
                alto = serialization.serialize_segmentation(baseline_seg, image_name=image_filename, image_size=im.size, template='alto')
                # On test si le reptoire de résultat a été crée pour y déposer les fichier alto ou non
                dirout = path_out_alto + "/" + elt + "/"
                if os.path.isdir(dirout):
                    pass
                else:
                    os.mkdir(dirout)
                filename = os.path.splitext(os.path.basename(pg))
                output_xml = dirout + "/" + filename[0] + "_alto.xml"
                with open(output_xml, 'w') as fp:
                    fp.write(alto)
                i += 1
        except:
            print("Le fichier" + elt + " n'a pas été traité")
            pass

    return

