import os
from PIL import Image
from kraken import blla, lib, pageseg, serialization
from kraken.lib import vgsl

MODEL_PATH = 'model//blla.mlmodel'


def list_rep(rootdir):
    return [rep for rep in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, rep))]


def create_output_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def process_image(image_filename, output_xml):
    try:
        im = Image.open(image_filename)
        seg_model = vgsl.TorchVGSLModel.load_model(MODEL_PATH)
        baseline_seg = blla.segment(im, model=seg_model)
        with open(output_xml, 'w') as fp:
            fp.write(serialization.serialize_segmentation(baseline_seg, image_name=image_filename, image_size=im.size, template='alto'))
        print(f"Écriture de {output_xml}")
    except Exception as e:
        print(f"Erreur lors du traitement de {image_filename}. Erreur : {e}")


def segmentation_alto(path_image, path_out_alto):
    for elt in list_rep(path_image):
        thepath = os.path.join(path_image, "/", elt)
        tmp_dir = os.path.join(thepath, "/", "reframed")
        fichiers = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]

        for page_count, pg in enumerate(fichiers, start=1):
            image_filename = os.path.join(tmp_dir, pg)
            dirout = os.path.join(path_out_alto, elt)
            create_output_directory(dirout)

            filename = os.path.splitext(os.path.basename(pg))
            output_xml = os.path.join(dirout, f"{filename[0]}_alto.xml")

            if not os.path.exists(output_xml):
                process_image(image_filename, output_xml)
            else:
                print(f"{output_xml} est déjà écrit")
