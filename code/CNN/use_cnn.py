import cnn
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']:
        print "################# Input argument error #################"
        print "### Usage: python use_cnn.py imageType 'category1 category2 etc.'"
        print "### Imagetype should be in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']"
        print "### Category should be in ['stressful', 'relaxing', 'sudden', 'other', 'human', 'traffic', 'noise', 'mechanical', 'silence', 'nature', 'music', 'machine']"
        sys.exit(1)
    
    # Parse user input
    imageType = sys.argv[1]
    cats = sys.argv[2].split(' ')
    catName = ''
    for cat in cats:
        catName += cat
    name = catName + '_' + imageType + '.cnn'
    
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
        
    ### Creating and loading data ###
    # Push je data aub niet:
    X_train, Y_train_all, X_test, Y_test_all = cnn.build_normalized_data(imageType, imageType, train_split = 0.9)
    
    #To load saved data
    X_train, Y_train_all, X_test, Y_test_all = cnn.load_normalized_data(imageType)

    Y_train = Y_train_all[:, cats]
    Y_test = Y_test_all[:, cats]

### 50-50 split ###
    # Dit zorgt ervoor dat je voor een categorie evenveel data hebt voor positief als voor negatief (e.g. stress vs niet stress)
    # Werkt alleen als Y maar 1 categorie bevat.
    ## X_train, Y_train = cnn.same_number_of_idxs(X_train, Y_train)
    ## X_test, Y_test = cnn.same_number_of_idxs(X_train, Y_test)

### CNN bouwen ###
    #Voor het inladen/opslaan/bouwen van een model:
    #Als je geen gewichtsbestand meegeeft, kun je een aantal trainings epochs meegeven.
    # Het model houd zelf niet bij wat de categorien zijn waarop hij traint.
    # test data wordt gebruikt als validatieset. Hij traint hier niet op.
    model = cnn.build(X_train, Y_train, X_test, Y_test, epochs = 30)
    
    # Voor het opslaan van gewichten:
    # Push je gewichten aub niet.
    cnn.save_weights(model, weights_filename = name)

### CNN testen ###    
    #Als weights_filename wodt meegegeven gebruikt hij dat weightsbestand ipv een nieuw model te bouwen. 
    # Train data zijn nodig om het model te initialiseren
    model = cnn.build(X_train, Y_train, weights_filename = name)

    # Voorspelt de output voor de input die je aangeeft. 
    output = model.predict(X_test)
    
    # Drukt de error voor elke category af waar je model op getrained en getest is.
    # Hij houdt niet bij voor welke categorien er getest wordt.
    # 1e kolom: Accuracy. 2e kolom: percentage dat negatief gelabeled is (e.g. percentage stressfull = False)
    cnn.print_error_rate_per_category(output, Y_test[:, cats])
