import os, errno
from PIL import Image
import numpy as np
from keras.models import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback, CSVLogger
from keras import optimizers
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models import modelsClass
from generators import *

BATCH_SIZE = 2
NUM_EPOCHS = 5
INITIAL_EPOCH = 0
NUM_TRAIN_SAMPLES = 50
NUM_VAL_SAMPLES = 10

class trainClass(object):

    def __init__(self, img_rows = 270, img_cols = 480):

        self.img_rows = img_rows
        self.img_cols = img_cols            
            
    def prepareCallbacks(self,model):
    
        try:
            os.makedirs("output/")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        checkpointer = ModelCheckpoint('checkpoints/checkpoint.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        predictor = LambdaCallback(on_epoch_end=lambda epoch,logs: self.predict_and_save(epoch,model))
        
        def changeLearningRate(epoch):
            current_lr = keras.get_value(model.optimizer.lr)
            if (epoch+1)%10 == 0:
                new_lr = 0.5*current_lr
                print("Changing learning rate: %f -> %f" %(current_lr,new_lr))
                keras.set_value(model.optimizer.lr, new_lr)
            else:
                new_lr = current_lr
                
            return new_lr
        scheduler = LearningRateScheduler(changeLearningRate)
        logger = CSVLogger('output/training.log')
        
        return checkpointer,predictor,scheduler,logger
        
    def trainModel(self):
        
        # Prepare training generator
        train_img_generator = ImageDataGenerator(rescale = 1./255, 
                                                 horizontal_flip=True, 
                                                 vertical_flip=True)
                                                 
        train_generator = prepareGenerator(train_img_generator, 
                                           'input/train/',
                                           self.img_rows,
                                           self.img_cols,
                                           BATCH_SIZE,
                                           randomize=True)
        
        # Prepare validation generator
        val_img_generator = ImageDataGenerator(rescale = 1./255)
        
        val_generator = prepareGenerator(val_img_generator,
                                         'input/val/',
                                         self.img_rows,
                                         self.img_cols,
                                         BATCH_SIZE,
                                         randomize=False)
        
        # Get model (and loading existing weights)
        models = modelsClass()
        model = models.LSD2()
        
        if (INITIAL_EPOCH > 1):
            print("Loading weights.")
            model.load_weights('checkpoints/checkpoint.hdf5')
            
        adam = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer = adam, loss = 'mean_squared_error')

        print("Preparing callbacks.")
        checkpointer,predictor,scheduler,logger = self.prepareCallbacks(model)
        
        print("Training started.")
        model.fit_generator(train_generator,
                            steps_per_epoch=int(NUM_TRAIN_SAMPLES/BATCH_SIZE),
                            epochs=NUM_EPOCHS,
                            verbose=1,
                            validation_data=val_generator,
                            validation_steps=int(NUM_VAL_SAMPLES/BATCH_SIZE),
                            callbacks=[checkpointer,predictor,scheduler,logger],
                            initial_epoch = INITIAL_EPOCH)

    def predict_and_save(self,epoch,model):
    
        print("Predicting (validation data) ...")
        
        outpath = "output/val/" + str(epoch) + "/"
        try:
            os.makedirs(outpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # Path to long and short exposure images
        long_path = "input/val/long/long/"
        short_path = "input/val/short/short/"
        long_names = os.listdir(long_path)
        short_names = os.listdir(short_path)
        
        N = 6 # Number of images to predict
        
        for fname in long_names[:N]:
                        
            long_img = Image.open(long_path + fname)
            short_img = Image.open(short_path + fname)
    
            long_np = (1./255)*np.array(long_img)
            short_np = (1./255)*np.array(short_img)
            
            width, height = long_img.size
            input_a = np.reshape(long_np,[1,height,width,3])
            input_b = np.reshape(short_np,[1,height,width,3])
            inputs = [input_a, input_b]
    
            prediction = model.predict(inputs, batch_size=1,verbose=0,steps=None)
            prediction = prediction[0,:,:,:]
            
            output_img = array_to_img(prediction)
            output_img.save(outpath+"/%s"%(fname))
            
        print("DONE!")
            

if __name__ == '__main__':

    trainer = trainClass()
    trainer.trainModel()


