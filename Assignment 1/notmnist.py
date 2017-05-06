import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image


# load the model from disk
filename = 'model_LogReg.sav'
file_to_open = open(filename, 'rb')
loaded_model = pickle.load(file_to_open)

#load image from disk
path_for_img = 'A.png'
print (path_for_img)

img = Image.open(path_for_img).convert('L')

img_as_np = np.asarray(img)
print(img_as_np.shape)

#reshape the image for processing
width, height = img_as_np.shape
X_train = np.reshape(img_as_np,(width*height))
print(X_train.shape)

result = loaded_model.predict([X_train])
print(result)

if result==0:
    printletter = "\nLetter is A"
elif result==1:
    printletter = "\nLetter is B"
elif result==2:
    printletter = "\nLetter is C"
elif result==3:
    printletter = "\nLetter is D"
elif result==4:
    printletter = "\nLetter is E"
elif result==5:
    printletter = "\nLetter is F"
elif result==6:
    printletter = "\nLetter is G"
elif result==7:
    printletter = "\nLetter is H"
elif result==8:
    printletter = "\nLetter is I"
elif result==9:
    printletter = "\nLetter is J"
    
print(printletter)