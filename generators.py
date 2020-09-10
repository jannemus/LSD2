import threading
from keras.preprocessing.image import ImageDataGenerator

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
            
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
    
@threadsafe_generator
def prepareGenerator(generator,dir_path,img_rows,img_cols,bsize,randomize):
    
    # Provide the same seed
    seed = 1
        
    sharp_gen = generator.flow_from_directory(dir_path + 'sharp',
                                              target_size=(img_rows, img_cols),
                                              color_mode='rgb',
                                              class_mode=None,
                                              batch_size=bsize,
                                              shuffle=randomize,
                                              seed=seed)
        
    long_gen = generator.flow_from_directory(dir_path + 'long',
                                             target_size=(img_rows, img_cols),
                                             color_mode='rgb',
                                             class_mode=None,
                                             batch_size=bsize,
                                             shuffle=randomize,
                                             seed=seed)  

    short_gen = generator.flow_from_directory(dir_path + 'short',
                                              target_size=(img_rows, img_cols),
                                              color_mode='rgb',
                                              class_mode=None,
                                              batch_size=bsize,
                                              shuffle=randomize,
                                              seed=seed)                                       
    while True:
        long_i = long_gen.next()
        short_i = short_gen.next()
        sharp_i = sharp_gen.next()
        yield [long_i, short_i], sharp_i
            
