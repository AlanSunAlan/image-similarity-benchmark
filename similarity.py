from dis import dis
import cv2
import numpy as np
import warnings
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from scipy.ndimage import imread

def CalSig(img):
  '''
    Given an image (numpy array), return the signature
  '''
  

def EMD(img1, img2):
    bin_size = 30

    # Smooth the image
    img1 = cv2.GaussianBlur(img1, (5,5), 0)
    img2 = cv2.GaussianBlur(img2, (5,5), 0)

    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Create histogram
    hist1 = cv2.calcHist([img1], [0], None, [bin_size], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [bin_size], [0,256])
    
    # Normalize the hist
    norm_hist1 = hist1/img1.size
    norm_hist2 = hist2/img2.size

    # Calculate signature
    sig1_list = []
    for i in range(0, len(norm_hist1)):
        avg_intensity = (i+0.5)*255/bin_size
        sig1_list.append([avg_intensity,norm_hist1[i]])

    sig2_list = []
    for i in range(0, len(norm_hist2)):
        avg_intensity = (i+0.5)*255/bin_size
        sig2_list.append([avg_intensity,norm_hist2[i]])

    sig1 = np.matrix(sig1_list, dtype=np.float32)
    sig2 = np.matrix(sig2_list, dtype=np.float32)

    return cv2.EMD(sig1, sig2, cv2.DIST_L2)

def CalcEMD(path_a, path_b):
    img1 = cv2.imread(path_a)
    img2 = cv2.imread(path_b)

    dis, _, _ = EMD(img1, img2)
    return dis

##
# Globals
##

warnings.filterwarnings('ignore')

# specify resized image sizes
height = 80
width = 120

##
# Functions
##

def get_img(path, norm_size=True, norm_exposure=False):
  '''
  Prepare an image for image processing tasks
  '''
  # flatten returns a 2d grayscale array
  img = imread(path, flatten=True).astype(int)
  # resizing returns float vals 0:255; convert to ints for downstream tasks
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img


def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w) 


def normalize_exposure(img):
  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)


def structural_sim(path_a, path_b):
  img_a = get_img(path_a)
  img_b = get_img(path_b)
  sim, diff = compare_ssim(img_a, img_b, full=True)
  return sim


def pixel_sim(path_a, path_b):
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


def sift_sim(path_a, path_b):
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # get the images
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)
