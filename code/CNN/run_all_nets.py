import use_cnn
import os

if __name__ == '__main__':
    allCats = ['stressful', 'relaxing', 'human', 'traffic', 'noise']
    allImageTypes = ['energy', 'morphology', 'tau 1.0', 'tau 4.0']
    
    if not os.path.exists('trained_nets/'):
        os.makedirs('trained_nets')
    
    for imageType in allImageTypes:
        print "imageType: ", imageType
        print "categorical crossenthropy nets"
        use_cnn.runNet_cat(imageType, 'trained_nets/')
        for category in allCats:
            print "category: ", category
            use_cnn.runNet(imageType, category, 'trained_nets/')