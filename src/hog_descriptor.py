import cv2

winSize = (28,28)
blockSize = (14,14)
blockStride = (7,7)
cellSize = (7,7)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

winStride = (1,1)
padding = (0,0)

cv2.createLBPHFaceRecognizer()
def get_hog(img):
  hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                          histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
  hist = hog.compute(img,winStride,padding)
  return hist
