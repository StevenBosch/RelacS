import os
import sys

label_dir = os.path.join(os.getcwd(), '../CNN')
sys.path.insert(0, label_dir)
import cnn
import freq_int_hist

if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test = cnn.build_data(imageType = 'energy', key = 'energy', train_split = 0.9)
    X_train, Y_train, X_test, Y_test = cnn.load_data(key = 'energy')
    
    
    freq_int_hist.bayes(X_train, Y_train, X_test, Y_test, 
        pyramid_height = 1, max_bins = 256, show_train_acc = True, show_images = False)