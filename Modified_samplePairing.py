
import numpy as np
import pandas as pd  # version 0.25.3
# !pip install git+https://github.com/keras-team/keras-preprocessing.git # need this for ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
from keras import backend
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.activations import elu
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications import Xception
from keras.applications.resnet import ResNet50

from sklearn.model_selection import StratifiedShuffleSplit

#from imblearn.over_sampling import SMOTE
from skimage import exposure 


def ResNet50Net():
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation=elu))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))

    model.summary()
    return model


def ResNet50NetFreeze():
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation=elu))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))

    for layer in conv_base.layers[:-4]:
        layer.trainable = False
    model.summary()
    return model


def XceptionNet():
    IMG_WIDTH = 299
    IMG_HEIGHT = 299
    conv_base = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation=elu))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))

    model.summary()
    return model


def XceptionNetFreeze():
    IMG_WIDTH = 299
    IMG_HEIGHT = 299
    conv_base = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation=elu))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))

    for layer in conv_base.layers[:-4]:
        layer.trainable = False
    model.summary()
    return model

def get_model(nameModel):

    if nameModel=='ResNet50':
        IMG_WIDTH = 224
        IMG_HEIGHT = 224
        model=ResNet50Net()
        return IMG_WIDTH,IMG_HEIGHT,model

    if nameModel=='ResNet50Freeze':
        IMG_WIDTH = 224
        IMG_HEIGHT = 224
        model=ResNet50NetFreeze()
        return IMG_WIDTH,IMG_HEIGHT,model

    if nameModel=='Xception':
        IMG_WIDTH = 299
        IMG_HEIGHT = 299
        model=XceptionNet()
        return IMG_WIDTH,IMG_HEIGHT,model


    if nameModel=='XceptionFreeze':
        IMG_WIDTH = 299
        IMG_HEIGHT = 299
        model=XceptionNetFreeze()
        return IMG_WIDTH,IMG_HEIGHT,model


def getFolders(df,n_fold=5,test_size=0.2):
    
    kf = StratifiedShuffleSplit(n_splits=n_fold, test_size=test_size, random_state=seed)
    X = df.iloc[:, 0]
    y = df.iloc[:, 1]
    dataframe_train_collection = {}
    dataframe_val_collection = {} 
    j=1
    for train_index, val_index in kf.split(X, y):
        trainData = X[train_index]
        valData = X[val_index]
        trainLabels = y[train_index]
        valLabels = y[val_index]
        dataframe_train_collection[j] =pd.DataFrame({"image": trainData,
                                     "level": trainLabels})
        dataframe_val_collection[j] = pd.DataFrame({"image": valData,
                                   "level": valLabels})
        j=j+1
    return dataframe_train_collection, dataframe_val_collection



def plot_result(acc,val_acc,loss,val_loss):
        

    plt.subplot(121)
    plt.plot(acc, label='Training acc')
    plt.plot(val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoches')
    plt.ylabel('Acc')
    plt.legend()

    plt.subplot(122)
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Accuracy_and_loss.png', dpi=200)
    plt.show()

def rmse(y_true, y_predict):

    return backend.sqrt(backend.mean(backend.square(y_predict - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_predict):
  
    return backend.mean(backend.square(y_predict - y_true), axis=-1)



def preprocess_image(image):
    """
    The whole preprocessing pipeline:
    1. Add Gaussian noise to increase Robustness
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image/ 255
    imageimgequalized = exposure.equalize_adapthist(image, clip_limit=0.03)
    image = image*255
    
    #image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
   

    return image 




def get_data_generators(train_df_new,val_df_new,test_df):

    train_datagen = ImageDataGenerator(rotation_range=90,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       zoom_range=(0.7, 1),
                                       fill_mode='constant',
                                       brightness_range=(0.5, 2),
                                       cval=0,
                                       preprocessing_function=preprocess_image,
                                       rescale=1. / 255)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image,
                                     rescale=1. / 255)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image,
                                      rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(train_df_new,
                                                        x_col='image',
                                                        y_col='level',
                                                        directory=TRAIN_IMG_PATH,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='other',
                                                        shuffle=True,
                                                        validate_filenames=True,
                                                        seed=seed)
    train_generator2 = train_datagen.flow_from_dataframe(train_df_new,
                                                        x_col='image',
                                                        y_col='level',
                                                        directory=TRAIN_IMG_PATH,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='other',
                                                        shuffle=True,
                                                        validate_filenames=True,
                                                        seed=seed)
    

    val_generator = val_datagen.flow_from_dataframe(val_df_new,
                                                    x_col='image',
                                                    y_col='level',
                                                    directory=TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='other',
                                                    
                                                    shuffle=True,
                                                    validate_filenames=True,
                                                    seed=seed)
    tes_generator = test_datagen.flow_from_dataframe(test_df,
                                                     x_col='image',
                                                     y_col='level',
                                                     directory=TRAIN_IMG_PATH,
                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=TEST_BATCH_SIZE,
                                                     class_mode='other',
                                                     shuffle=False,
                                                     seed=seed)
    return train_generator,train_generator2,val_generator ,tes_generator



def generaBalancedData(trainData,outputNewsamplePath,alpha,TRAIN_IMG_PATH ,IMG_WIDTH, IMG_HEIGHT,batch_size,seed):
        train_datagen =ImageDataGenerator()
        train_genartor1 = train_datagen.flow_from_dataframe(trainData,
                                                        x_col='image',
                                                        y_col='level',
                                                        directory=TRAIN_IMG_PATH,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=batch_size,
                                                        class_mode='other',
                                                        shuffle=True,
                                                        validate_filenames=True,
                                                        seed=seed)
        train_genartor2 = train_datagen.flow_from_dataframe(trainData,
                                                        x_col='image',
                                                        y_col='level',
                                                        directory=TRAIN_IMG_PATH,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=batch_size,
                                                        class_mode='other',
                                                        shuffle=True,
                                                        validate_filenames=True,
                                                        seed=seed)
        lable,contatore=np.unique(trainData['level'],return_counts=True)
        print(lable,contatore)
        Max_frequence=max(contatore)
        count=0
        stop_generation=0
        for i in contatore:
            stop_generation = stop_generation + (Max_frequence - i)
        
        
        newSampleDict={'image':[],'level':[]}
        
        for x1,y1 in train_genartor1: 
            if count >= stop_generation:
                break
            print(x1.shape[0])# or batch size
            for x2,y2 in train_genartor2:
                for i in range(x1.shape[0]):
                    if contatore[y1[i]]< Max_frequence:
                        for j in range (x2.shape[0]):
                             if contatore[y2[j]]< Max_frequence:
                                    
                                count=count+1
                                filename=r"\newsamplepairing%d.jpg"%count
                                path=outputNewsamplePath+filename
                                newsample= x1[i] * alpha + x2[j] * (1 - alpha)
                                lable = int(np.round(y1[i] * alpha + y2[j] * (1 - alpha)))
                                contatore[lable]=contatore[lable]+1 
                                print(count)
                                cv2.imwrite(path,newsample) 
                                print(path)
                                newSampleDict['image'].append(filename)
                                newSampleDict['level'].append(lable)
            
                            
        
        newdf=pd.DataFrame.from_dict(newSampleDict) 
        print(count,contatore)
        return newdf
               
    



class MixupImageDataGenerator():
    def __init__(self, generator1,generator2,numberOfsamples, alpha):
  

        self.batch_index = 0
        self.batch_size = BATCH_SIZE
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1= generator1
        self.generator2= generator2


        # Number of images across all classes in image directory.
        self.n = numberOfsamples
        
        
    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size


    def __next__(self):
        """Get next batch input/output pair.
        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        s1=X1.shape[0]
        s2=X2.shape[0]
        
        if s1 == s2:
            l=s1//2

        elif s1 < s2:
            l=s1//2
            X2=X2[:s1-1]
            
        else:
            l=s2//2
            X1=X1[:s2-1]

        #Keep a half of data without pairing it.
        InalterateX=X1[:l]
        InalterateY=y1[:l]
        # Perform the samplepairing on the second half of a batch.
        X = X1[l:] * self.alpha + X2[l:] * (1 - self.alpha)
        y = np.around(y1[l:] * self.alpha + y2[l:] * (1 - self.alpha))

        X = np.concatenate((X, InalterateX), axis=0)
        y = np.concatenate((y, InalterateY),axis=0)
        
        return X, y

    def __iter__(self):
        while True:
            yield next(self)

def fitModel_samplePairing(model,train_generator1,train_generator2,val_generator,numberOfsamples):

    mixupGenerator = MixupImageDataGenerator(train_generator1,train_generator2,numberOfsamples,0.4)
    
    model.compile(loss='mse',
                  optimizer=Adam(1e-4, decay=1e-4),
                  metrics=['acc', rmse])
    # to train xception or resnet with freeze layer i use 1 folder with 5 epochs( i got best results)
    # to train xception or resnet without freeze layer i use 5 folder each 10 epochs.
    # i tried also resnet with 3 folders with 10 epochs.

    history = model.fit_generator(mixupGenerator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=NUM_EPOCH,
                            validation_data=val_generator,
                            validation_steps=STEP_SIZE_VALID)



    acc = history.history['acc']
    val_acc = history.history['val_acc']
   
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_result(acc,val_acc,loss,val_loss)
        
    # save weights because it need time to training
    model.save_weights("samplpairingmodel.h5")  # <--------------------- check path

    return acc, val_acc, loss, val_loss,model

def fitModel(model,train_generator,val_generator):

    model.compile(loss='mse',
                  optimizer=Adam(1e-4, decay=1e-4),
                  metrics=['acc', rmse])
    # to train xception or resnet with freeze layer i use 1 folder with 5 epochs( i got best results)
    # to train xception or resnet without freeze layer i use 5 folder each 10 epochs.
    # i tried also resnet with 3 folders with 10 epochs.
    history = model.fit_generator(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=NUM_EPOCH,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VALID)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
   
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_result(acc,val_acc,loss,val_loss)
        
    # save weights because it need time to training
    model.save_weights("aug_model.h5")  # <--------------------- check path

    return acc, val_acc, loss, val_loss,model


def evaluat_model(model,test_generator,path,path_precision):

   
    print("evaluating.....")
    test_generator.reset()
    score1 = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
    print('Results on test set')
    print("%s: %2.f%%" % (model.metrics_names[1], score1[1] * 100))
    print("%s: %2.f%%" % (model.metrics_names[0], score1[0] * 100))
    print("predicting.....")
    test_generator.reset()
    y_pred = model.predict_generator(test_generator,steps=STEP_SIZE_TEST)
    print()
    # find the indices for each image
    predicted_class_indices = np.around(y_pred)

    indices = predicted_class_indices.astype(int)
    indices = indices.flatten()
    labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
    labels = dict((v, k) for k, v in labels.items())
    
    predictions = [labels[k] for k in indices]
    
    filenames = test_generator.filenames
    realLabels = test_generator._targets
    print(len(filenames),len(realLabels),len(y_pred))
    
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions,
                            "Real_label": realLabels})
    
  

    frequency_test = results.Real_label.value_counts()
    print("Frequency of class indices in test set:\n",frequency_test)
    precision=results.groupby(["Predictions","Real_label"]).size().reset_index(name='counts')
    
    true_positive=precision[precision["Predictions"]==precision["Real_label"]]

    results.to_csv(path, index=False)  
    precision.to_csv(path_precision, index=False)
    print("precision table:\n",precision)
    print("Done!..Results are saved!")
    

# this path refers to the folder where there are images  <--------------------- check path
TRAIN_IMG_PATH = r'C:\Users\adham\Downloads\Alhamdoosh\resized_train_cropped\resized_train_cropped'  

# select the correct path of csv <--------------------- check path
trainData = pd.read_csv("trainLabels_cropped.csv")  

print("tf Version:",tf.__version__) 

trainData = trainData[['image', 'level']]
print(trainData)
trainData['image'] = trainData['image'].astype(str) + '.jpeg'


trainData.level.value_counts().sort_index().plot(kind='bar')
plt.title('Labels counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)
BATCH_SIZE = 32
df = trainData[:28100]
test_df = trainData[28100: 28132] # trainData[28100:] <-------------- check
TEST_BATCH_SIZE=test_df.shape[0]
STEP_SIZE_TEST = 1

list_models=['ResNet50','ResNet50Freeze','Xception','XceptionFreeze']
#first expirment using Xception'
IMG_WIDTH,IMG_HEIGHT, model= get_model('XceptionFreeze')
#Generate 5 folders of data  and insert them in dict
dataframe_train_collection, dataframe_val_collection= getFolders(df,5)
#Experimental fase,choos juste the first folder.
train_df_new=dataframe_train_collection[1]
val_df_new=dataframe_val_collection[1]
numberOfsamples=train_df_new.shape[0]
STEP_SIZE_TRAIN = 1  #train_df_new.shape[0] // BATCH_SIZE <-------------- check
STEP_SIZE_VALID = 1  #val_df_new.shape[0] // BATCH_SIZE  <-------------- check
NUM_EPOCH=1 # 5 <----------------------- check 

#Get keras ImageDataGeneratos for training, valdiation and test datsets
train_generator1,train_generator2,val_generator ,test_generator = get_data_generators(train_df_new,val_df_new,test_df)

# fit model 
def fit_model(testname):
    if testname == 'aug':
        acc, val_acc, loss, val_loss, trained_model =fitModel(model,train_generator1,val_generator)
        evaluat_model(trained_model,test_generator,"aug.csv","precisionAug.csv")
    if testname == 'samplePairing':
        acc, val_acc, loss, val_loss, trained_model = fitModel_samplePairing(model, train_generator1, train_generator2, val_generator, numberOfsamples)
        evaluat_model(trained_model,test_generator,"samplePairing.csv","precisionSamplePairing.csv")
   
        
        
#fit_model('aug')

fit_model('samplePairing')





