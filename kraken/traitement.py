import kraken
import kraken.binarization
# pageseg is the library from pagesegmentation
from kraken import binarization, pageseg
from PIL import Image
from google.colab import files


path_image = "../img"

def list_rep(rootdir):
  list_dir = []
  for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        list_dir.append(d)
  return list_dir




image_filename = '/content/images/page05.jpeg'
im=Image.open(image_filename)
bw_im = binarization.nlbin(im)
seg = pageseg.segment(bw_im)


