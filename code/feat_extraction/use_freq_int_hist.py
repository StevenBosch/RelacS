import os
import sys

label_dir = os.path.join(os.getcwd(), '../CNN')
sys.path.insert(0, label_dir)
import cnn
import freq_int_hist
import numpy as np

if __name__ == '__main__':
    ding = 'morphology'
    X_train, Y_train, X_test, Y_test = cnn.build_data(imageType = ding, key = ding, train_split = 0.9)
    #X_train, Y_train, X_test, Y_test = cnn.load_data(key = 'ding')
    
    classifier = freq_int_hist.fih(X_train, Y_train)

    filepath = os.path.join(os.getcwd(), 'classifiers/' + ding + '.pickle')
    classifier.save(filepath )

    classifier2 = freq_int_hist.fih(file = filepath)
    output = classifier2.predict(X_test, show_accuracy = True, Y_test = Y_test) #Y_test not needed when show_accuracy not specified
    print ding
    # freq_int_hist.bayes(X_train, Y_train, X_test, Y_test, 
    #     pyramid_height = 1, max_bins = 256, show_train_acc = False, show_images = False)