import os, os.path, errno
from PIL import Image
import numpy as np
from keras.models import Model
from keras.preprocessing.image import array_to_img
from models import modelsClass

inpath = "input/test"
outpath = "output/test"

# Create output folder
try:
    os.makedirs(outpath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Path to long and short exposure images
long_path = inpath + "/long/long/"
short_path = inpath + "/short/short/"
long_names = os.listdir(long_path)
short_names = os.listdir(short_path)

print("Found %d long exposure images." %len(long_names))
print("Found %d short exposure images." %len(short_names))

for fname in long_names:
        
    print("Processing '%s'" %(fname))
                        
    long_img = Image.open(long_path + fname)
    short_img = Image.open(short_path + fname)
    
    long_np = (1./255)*np.array(long_img)
    short_np = (1./255)*np.array(short_img)
            
    width, height = long_img.size
    models = modelsClass(height,width)
    model = models.LSD2()
    model.load_weights("checkpoints/checkpoint_final.hdf5")
            
    input_a = np.reshape(long_np,[1,height,width,3])
    input_b = np.reshape(short_np,[1,height,width,3])
    inputs = [input_a, input_b]
    
    prediction = model.predict(inputs, batch_size=1,verbose=0,steps=None)
    prediction = prediction[0,:,:,:]
            
    output_img = array_to_img(prediction)
    output_img.save(outpath+"/%s"%(fname))
            
print("DONE!")

