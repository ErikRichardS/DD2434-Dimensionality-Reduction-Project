import sklearn.feature_extraction
import sklearn.preprocessing
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import pydub

#Use local path for all the pictures to be analysed
img_paths = glob.glob('/*.png')

#Use local path for the folders containing the articles here
file_paths = glob.glob('Data/*/*.txt')
#The text in the files are put into a list

#Use local path for all the music files
music_paths = glob.glob('/*.mp3')

def main():
    
    tf_matrix = readTextFiles()             #Output is term-frequency matrix for a set of dokuments
    image_matrix = readImageFiles()         #Output is a matrix where each row contains grayscale values of 50x50 pixels
    

def readTextFiles():
    file_data = []
    #Open all the text files and extract the text into a list of files
    for files in file_paths:
        with open(files) as f: 
            file_data.append(f.read())


    #For exact number of features(Amount of unique tokens/columns), smaller matrix, memory heavy due to storing tokens
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()     #Using exact number of features as unique tokens
    tf_matrix = count_vectorizer.fit_transform(file_data)  
    tf_matrix = sklearn.preprocessing.normalize(tf_matrix,axis=1)
                   
    
    #For a larger number of features(columns), sparse matrix, memory effective, using hash-table
    #vectorizer = sklearn.feature_extraction.text.HashingVectorizer()           
    #tf_matrix = vectorizer.fit_transform(file_data)                        

    return tf_matrix.transpose()    #Columns as datapoints, rows as dimensions


def readImageFiles():
    nrOfPixels = 2500
    image_vectors = []
    for images in img_paths: 
        pixels = []
        img = plt.imread(images)            #Read image from path
        img = np.mean(img, -1)              #Use grayscale instead of rgb
        img = img.ravel()                   #Flatten image to 1D vector for future distance calculations
        for i in range(0,100):              #100*20 = 2000 data points
            randomPixels = np.random.choice(img,nrOfPixels) #Create a random 50x50 image from original
            image_vectors.append(randomPixels)              #All vectors gathers in the list as arrays

    image_matrix = np.array(image_vectors)
    #TO-DO: Gather total data set of images
    return sparse.csr_matrix(image_matrix.transpose())             #Columns as datapoints, rows as dimensions


def readMusicFiles():
        # build matrix with music data
        # each column vector consists of soundsamples from 1 second of music 
        seconds = 1 # controls the number of dimensions
        samplerate = 44100 
        # the number of dimensions per datapoint
        d_per_datapoint = seconds*samplerate
        music_vectors = np.empty(shape=[d_per_datapoint, 0])
        for path in music_paths:
            # get audio segment
            soundsamples_asAudioSegment = pydub.AudioSegment.from_file(path, format="mp3")
            # make single channel
            soundsamples_asAudioSegment = soundsamples_asAudioSegment.set_channels(1)
            # get number of samples
            num_samples = int(soundsamples_asAudioSegment.frame_count()) 
            # extract raw audio from wav file and normalize samples to make in range [-1,1]
            soundsamples = np.divide(soundsamples_asAudioSegment.get_array_of_samples(), 
                          soundsamples_asAudioSegment.max_possible_amplitude)
            # add vector of d-dimensions to music_vectors
            for i in range(0,num_samples,d_per_datapoint):
                if i+d_per_datapoint <= num_samples:
                    datapoint = soundsamples[i:i+d_per_datapoint]
                    music_vectors = np.column_stack((music_vectors, datapoint))

        return sparse.csr_matrix(music_vectors)