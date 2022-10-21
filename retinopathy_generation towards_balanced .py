

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




def get_data_generators(train_df_new,val_df_new,test_df,TRAIN_IMG_PATH, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE,seed):

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
                                                        seed=seed)
    train_generator2 = train_datagen.flow_from_dataframe(train_df_new,
                                                        x_col='image',
                                                        y_col='level',
                                                        directory=TRAIN_IMG_PATH,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='other',
                                                        shuffle=True,
                                                        seed=seed)
    

    val_generator = val_datagen.flow_from_dataframe(val_df_new,
                                                    x_col='image',
                                                    y_col='level',
                                                    directory=TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='other',
                                                    shuffle=True,
                                                    seed=seed)
    tes_generator = test_datagen.flow_from_dataframe(test_df,
                                                     x_col='image',
                                                     y_col='level',
                                                     directory=TRAIN_IMG_PATH,
                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=BATCH_SIZE,
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


def getFolders(df,seed,n_fold=5,test_size=0.2):
    
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
    plt.savefig('Output/Accuracy_and_loss.png', dpi=200)
    plt.show()

def rmse(y_true, y_predict):

    return backend.sqrt(backend.mean(backend.square(y_predict - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_predict):
  
    return backend.mean(backend.square(y_predict - y_true), axis=-1)


class MixupImageDataGenerator():
    def __init__(self, generator1,generator2,numberOfsamples, batch_size, alpha,seed):
  

        self.batch_index = 0
        self.batch_size = batch_size
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
       
        # Perform the mixup.
        X = X1 * self.alpha + X2 * (1 - self.alpha)
        y = np.around(y1 * self.alpha + y2 * (1 - self.alpha))
        return X, y

    def __iter__(self):
        while True:
            yield next(self)

def fitModel_samplePairing(model,train_generator1,train_generator2,val_generator,numberOfsamples,batch_size,STEP_SIZE_TRAIN,STEP_SIZE_VALID,seed):

    mixupGenerator = MixupImageDataGenerator(train_generator1,train_generator2,numberOfsamples,batch_size,0.4,seed)
    
    model.compile(loss='mse',
                  optimizer=Adam(1e-4, decay=1e-4),
                  metrics=['acc', rmse])
    # to train xception or resnet with freeze layer i use 1 folder with 5 epochs( i got best results)
    # to train xception or resnet without freeze layer i use 5 folder each 10 epochs.
    # i tried also resnet with 3 folders with 10 epochs.

    history = model.fit(mixupGenerator,
        steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=30,
                            validation_data=val_generator,
                            validation_steps=STEP_SIZE_VALID)



    acc = history.history['acc']
    val_acc = history.history['val_acc']
   
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_result(acc,val_acc,loss,val_loss)
        
    # save weights because it need time to training
    model.save_weights("Output/model.h5")  # <--------------------- check path

    return acc, val_acc, loss, val_loss,model

def fitModel(model,train_generator,val_generator,STEP_SIZE_TRAIN,STEP_SIZE_VALID):

    model.compile(loss='mse',
                  optimizer=Adam(1e-4, decay=1e-4),
                  metrics=['acc', rmse])
    # to train xception or resnet with freeze layer i use 1 folder with 5 epochs( i got best results)
    # to train xception or resnet without freeze layer i use 5 folder each 10 epochs.
    # i tried also resnet with 3 folders with 10 epochs.
    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=30,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VALID)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
   
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_result(acc,val_acc,loss,val_loss)
        
    # save weights because it need time to training
    model.save_weights("Output/model.h5")  # <--------------------- check path

    return acc, val_acc, loss, val_loss,model

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def evaluat_model(model,tes_generator,path,path_precision):

    

    score1 = model.evaluate_generator(tes_generator)
    print('Results on test set')
    print("%s: %2.f%%" % (model.metrics_names[1], score1[1] * 100))
    print("%s: %2.f%%" % (model.metrics_names[0], score1[0] * 100))

    y_pred = model.predict_generator(tes_generator)

    # find the indices for each image
    predicted_class_indices = np.around(y_pred)

    indices = predicted_class_indices.astype(int)
    indices = indices.flatten()
    labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in indices]
    filenames = tes_generator.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions,
                            "Real_label ": test_df['level']})
    
    # check the path where you want to save the csv
    results.to_csv(path, index=False)  # 


    precision=results.groupby(["Predictions","Real_label"]).size().reset_index(name='counts')
    precision.to_csv(path_precision, index=False)
    true_positive=precision[precision["Predictions"]==precision["Real_label"]]
    labels=true_positive["Real_label"]

    frequency_test=results.Real_label.value_counts()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, precision["counts"], width, label='true positive')
    rects2 = ax.bar(x + width/2, frequency_test, width, label='total')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Result on testdata: Number of true positive labels in each lable classes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()


# this path refers to the folder where there are images ( not cropped folder)
TRAIN_IMG_PATH = r'C:\Users\adham\Downloads\Alhamdoosh\resized_train_cropped\resized_train_cropped'  # <--------------------- check path

# select the correct path of csv
trainData = pd.read_csv("trainLabels_cropped.csv")  # <--------------------- check path
trainData = trainData[['image', 'level']]
trainData['image'] = trainData['image'].astype(str) + '.jpeg'

train_df= trainData[:28100]
test_df = trainData[28100:]

#train_df, test_df = train_test_split(trainData, test_size=0.2,random_state=seed)

trainData.level.value_counts().sort_index().plot(kind='bar')
plt.title('Labels counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
#------------------
#select path to folder where save the generated samples
outputNewsamplePath = r"C:\Users\adham\Downloads\Alhamdoosh\balanced_data" #<--------------------- check path
newdf = generaBalancedData(trainData,outputNewsamplePath,0.4,TRAIN_IMG_PATH ,IMG_WIDTH, IMG_HEIGHT,BATCH_SIZE,seed)
newdf=newdf[['image', 'level']]
newdf.to_csv (r"C:\Users\adham\Downloads\Alhamdoosh\risult\newdataframe.csv", index = False, header=True)
frames=[train_df,newdf]
#I'm not sure of this function, but it shuold concatenate the two data frame
balancedDf=pd.concat(frames)
#must combine the generated data and the original data in the same folder by copy the original data to balanced_data folder#<-------- Combine data folders

balancedDf.level.value_counts().plot(kind='bar')
plt.title('Balanced dataframe Labels counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
#-------------------------


seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)
BATCH_SIZE = 32
#split data into training and test datasets


#STEP_SIZE_TEST = test_df.shape[0] // BATCH_SIZE

print(tf.__version__)  # we need 2.1.0


list_models=['ResNet50','ResNet50Freeze','Xception','XceptionFreeze']
#first expirment using Xception'
IMG_WIDTH,IMG_HEIGHT, model= get_model('Xception')
#Generate 5 folders of data  and insert them in dict
dataframe_train_collection, dataframe_val_collection= getFolders(balancedDf,seed,5)
#Experimental fase,choos juste the first folder.
train_df_new=dataframe_train_collection[1]
val_df_new=dataframe_val_collection[1]
numberOfsamples=train_df_new.shape[0]
STEP_SIZE_TRAIN = train_df_new.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = val_df_new.shape[0] // BATCH_SIZE

#Get keras ImageDataGeneratos for training, valdiation and test datsets
train_generator1,train_generator2,val_generator ,test_generator = get_data_generators(train_df_new,val_df_new,test_df,TRAIN_IMG_PATH, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE,seed)


#Select the path to save results in csv
outputPath=r"C:\Users\adham\Downloads\Alhamdoosh\risult" #<--------------------- check path
 
acc, val_acc, loss, val_loss, trained_model =fitModel(model,train_generator1,val_generator,STEP_SIZE_TRAIN,STEP_SIZE_VALID)
evaluat_model(trained_model,test_generator,outputPath+r"aug.csv",outputPath+r"precisionAug.csv")


