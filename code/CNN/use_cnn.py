import cnn

if __name__ == '__main__':
### Creating and loading data ###
    # Push je data aub niet:
    #To build new data. Data is also automatically saved. Key is a file name appendix. You might have to make the map ../data first.
    X_train, Y_train_all, X_test, Y_test_all = cnn.build_normalized_data(imageType = 'energy', key = 'energy', train_split = 0.9)
    
    #To load saved data
    X_train, Y_train_all, X_test, Y_test_all = cnn.load_normalized_data(key = 'energy')


### Categorieen ###
    #categories is de categorien waarop getraind word. Categirien van 0 tot en met 11 zijn :
    # 'stressful', 'relaxing', 'sudden', 'Other, 'Human', 'Traffic', 'Noise', 'Mechanical', 'Silence', 'Nature', 'Music', 'Machine'
    cats = [0, 1, 2]

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
    cnn.save_weights(model, weights_filename = 'my_weights.cnn')

### CNN testen ###    
    #Als weights_filename wodt meegegeven gebruikt hij dat weightsbestand ipv een nieuw model te bouwen. 
    # Train data zijn nodig om het model te initialiseren
    model = cnn.build(X_train, Y_train, weights_filename = 'my_weights.cnn')

    # Voorspelt de output voor de input die je aangeeft. 
    output = model.predict(X_test)
    
    # Drukt de error voor elke category af waar je model op getrained en getest is.
    # Hij houdt niet bij voor welke categorien er getest wordt.
    # 1e kolom: Accuracy. 2e kolom: percentage dat negatief gelabeled is (e.g. percentage stressfull = False)
    cnn.print_error_rate_per_category(output, Y_test[:, cats])
