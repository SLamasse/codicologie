import os
import kraken
import kraken.binarization
# pageseg is the library from pagesegmentation
from kraken import binarization, pageseg
from PIL import Image
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
  tmp_dir = elt + "/reframed"
  fichiers = [f for f in listdir(tmp_dir) if isfile(join(tmp_dir, f))]
  for pg in fichiers:
    print(pg)

image_filename = '/content/images/page05.jpeg'
im=Image.open(image_filename)
bw_im = binarization.nlbin(im)
seg = pageseg.segment(bw_im)
seg_model = vgsl.TorchVGSLModel.load_model(model_path)
baseline_seg = blla.segment(im, model = seg_model)
alto = serialization.serialize_segmentation(baseline_seg, image_name=im.filename, image_size=im.size, template='alto')


