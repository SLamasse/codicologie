import os
from PIL import Image

import kraken
import kraken.binarization
from kraken import binarization, pageseg
from kraken import blla
from kraken.lib import vgsl
from kraken import serialization


path_image = "../img"
model_path = '/model/blla.mlmodel'

def list_rep(rootdir):
  list_dir = []
  for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        list_dir.append(d)
  return list_dir


for elt in list_rep(path_image):
  try:
    tmp_dir = elt + "/reframed"
    fichiers = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
    for pg in fichiers:
      im=Image.open(pg)
      bw_im = binarization.nlbin(im)
      seg = pageseg.segment(bw_im)
      seg_model = vgsl.TorchVGSLModel.load_model(model_path)
      baseline_seg = blla.segment(im, model = seg_model)
      alto = serialization.serialize_segmentation(baseline_seg, image_name=im.filename, image_size=im.size, template='alto')

  except:
    print("pas de fichiers dans le repertoire")







