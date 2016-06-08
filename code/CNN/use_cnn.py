import cnn
import sys

import numpy as np
import theano

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']:
        print "################# Input argument error #################"
        print "### Usage: python use_cnn.py 'imageType' 'category1 category2 etc.'"
        print "### Imagetype should be in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']"
        print "### Category should be in ['stressful', 'relaxing', 'sudden', 'other', 'human', 'traffic', 'noise', 'mechanical', 'silence', 'nature', 'music', 'machine']"
        sys.exit(1)
    
    # Parse user input
    imageType = sys.argv[1]
    cats = sys.argv[2].split(' ')
    
    allCats = ['stressful', 'relaxing', 'sudden', 'other', 'human', 'traffic', 'noise', 'mechanical', 'silence', 'nature', 'music', 'machine']
    for index, cat in enumerate(cats):
        if cat not in allCats:
            print "################# Input argument error #################"
            print "### Categories must be space separated and one of the following:"
            print '### stressful', 'relaxing', 'sudden', 'other', 'human', 'traffic', 'noise', 'mechanical', 'silence', 'nature', 'music', 'machine'
            print "### Use ' at the beginning and end of whole list, e.g.:"
            print "### 'stressful relaxing sudden'"
            sys.exit(1)
        cats[index] = allCats.index(cat)

    runNet(imageType, cats)


def runNet_cat(imageType, saveDir):
    name = saveDir + '_' +'stressVSrelacs' + imageType + '.cnn'

    temp = sys.stdout

    with open(saveDir + catName + '_' +'stressVSrelacs' + imageType + '.txt', 'w') as sys.stdout:
        X_train, Y_train_all, X_test, Y_test_all = cnn.load_normalized_data(imageType)

        Y_train = cnn.get_categorical_data_stress(Y_train_all)
        Y_test  = cnn.get_categorical_data_stress(Y_test_all)

        model = cnn.build_empty_softmax_model(X_train.shape, Y_train.shape)

        model.fit(X_train, Y_train, 
            batch_size = 32, nb_epoch = 30, 
            validation_data= (X_test, Y_test))

        cnn.save_weights(model, weights_filename = name)
        
        output = model.predict(X_test)
        cnn.print_error_rate_per_category(output, Y_test)




    name = saveDir + '_' +'categories' + imageType + '.cnn'
    with open(saveDir + catName + '_' +'categories' + imageType + '.txt', 'w') as sys.stdout:
        X_train, Y_train_all, X_test, Y_test_all = cnn.load_normalized_data(imageType)

        cat_lists = [[6],[5],[11],[10],[4],[7],[9],[8]]
        Y_train = cnn.get_categorical_data_cats(Y_train_all, cat_lists)
        Y_test  = cnn.get_categorical_data_cats(Y_test_all, cat_lists)

        model = cnn.build_empty_softmax_model(X_train.shape, Y_train.shape)

        model.fit(X_train, Y_train, 
            batch_size = 32, nb_epoch = 30, 
            validation_data= (X_test, Y_test))

        cnn.save_weights(model, weights_filename = name)
        
        output = model.predict(X_test)
        cnn.print_error_rate_per_category(output, Y_test)

    sys.stdout = temp



def runNet(imageType, cats, saveDir):
    # Make the filename
    catName = ''
    for cat in cats:
        catName += cat
    name = saveDir + catName + '_' + imageType + '.cnn'
   
    temp = sys.stdout
    with open(saveDir + catName + '_' + imageType + '.txt', 'w') as sys.stdout:
        X_train, Y_train_all, X_test, Y_test_all = cnn.load_normalized_data(imageType)

        Y_train = Y_train_all[:, cats]
        Y_test = Y_test_all[:, cats]

        model = cnn.build(X_train, Y_train, X_test, Y_test, epochs = 30)
        cnn.save_weights(model, weights_filename = name)

        output = model.predict(X_test)
        cnn.print_error_rate_per_category(output, Y_test)
    sys.stdout = temp


#     ### Creating and loading data ###
#         # Push je data aub niet:
#         X_train, Y_train_all, X_test, Y_test_all = cnn.build_normalized_data(imageType, imageType, train_split = 0.9)
        
#         #To load saved data
#         X_train, Y_train_all, X_test, Y_test_all = cnn.load_normalized_data(imageType)

#         Y_train = Y_train_all[:, cats]
#         Y_test = Y_test_all[:, cats]

#     ### 50-50 split ###
#         # Dit zorgt ervoor dat je voor een categorie evenveel data hebt voor positief als voor negatief (e.g. stress vs niet stress)
#         # Werkt alleen als Y maar 1 categorie bevat.
#         ## X_train, Y_train = cnn.same_number_of_idxs(X_train, Y_train)
#         ## X_test, Y_test = cnn.same_number_of_idxs(X_train, Y_test)

#     ### CNN bouwen ###
#         #Voor het inladen/opslaan/bouwen van een model:
#         #Als je geen gewichtsbestand meegeeft, kun je een aantal trainings epochs meegeven.
#         # Het model houd zelf niet bij wat de categorien zijn waarop hij traint.
#         # test data wordt gebruikt als validatieset. Hij traint hier niet op.
#         model = cnn.build(X_train, Y_train, X_test, Y_test, epochs = 30)
        
#         # Voor het opslaan van gewichten:
#         # Push je gewichten aub niet.
#         cnn.save_weights(model, weights_filename = name)

#     ### CNN testen ###    
#         #Als weights_filename wodt meegegeven gebruikt hij dat weightsbestand ipv een nieuw model te bouwen. 
#         # Train data zijn nodig om het model te initialiseren
#         model = cnn.build(X_train, Y_train, weights_filename = name)

#         # Voorspelt de output voor de input die je aangeeft. 
#         output = model.predict(X_test)
        
#         # Drukt de error voor elke category af waar je model op getrained en getest is.
#         # Hij houdt niet bij voor welke categorien er getest wordt.
#         # 1e kolom: Accuracy. 2e kolom: percentage dat negatief gelabeled is (e.g. percentage stressfull = False)
#         cnn.print_error_rate_per_category(output, Y_test[:, cats])
