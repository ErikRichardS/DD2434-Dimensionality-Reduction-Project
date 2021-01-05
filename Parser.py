import sklearn.feature_extraction
import glob
import numpy as np
#from cv2 import cv2
import matplotlib.pyplot as plt

#Use path for all the pictures to be analysed
img_paths = glob.glob('/Users/filip/Desktop/Skola/Maskininlärning2/Project_data/picture_data/*.png')

#Use directory for the folders containing the articles here
file_paths = glob.glob('Data/*.zip')
#The text in the files are put into a list

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
    
    #For a larger number of features(columns), sparse matrix, memory effective, using hash-table
    #vectorizer = sklearn.feature_extraction.text.HashingVectorizer()           
    #tf_matrix = vectorizer.fit_transform(file_data)                        

    #TO-DO: Double check normalization, use it or not?
    
    return tf_matrix


def readImageFiles():
    nrOfPixels = 250
    image_vectors = []
    for images in img_paths:
        pixels = []
        img = plt.imread(images)            #Read image from path
        img = np.mean(img, -1)              #Use grayscale instead of rgb
        img = img.ravel()                   #Flatten image to 1D vector for future distance calculations
        randomPixels = np.random.choice(img,nrOfPixels) #Create a random 50x50 image from original
        image_vectors.append(randomPixels)              #All vectors gathers in the list as arrays

    image_matrix = np.array(image_vectors)
    #TO-DO: Gather total data set of images, cut them into smaller images of 50x50?
    return image_matrix
    

    

    



