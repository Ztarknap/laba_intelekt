import pandas
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier




if __name__ == "__main__":


#--------------------------------------------------------------KDD cup-------------------------------------------------------

    df_kdd = pandas.read_csv('kdd-5classes.csv')
    df_new = pandas.read_csv('new 1-3.csv')
    df_test = pandas.read_csv('kdd-5classes_test.csv')
    df_united1 = pandas.concat([df_kdd, df_new], ignore_index=True)
    labels = df_united1['label']
    df_united1.drop(columns = ['label'], inplace = True)
    df_united2 = pandas.concat([df_united1, df_test], ignore_index=True)

    print(labels)
    print('------------------------------------------------------------------------------------------------')
    print(df_united2)
    df_united2 ['protocol_type'] = df_united2 ['protocol_type'].factorize()[0]
    df_united2 ['service'] = df_united2 ['service'].factorize()[0]
    df_united2['flag'] = df_united2['flag'].factorize()[0]

    print('------------------------------------------------------------------------------------------------')
    print(df_united2)
    print('------------------------------------------------------------------------------------------------')
    print(df_test)
    print('------------------------------------------------------------------------------------------------')
    print(labels)

    t = 0
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    for col_name in df_united2.columns:
        d = df_united2[col_name].to_list()
        for value in d:
            t = t + 1
            if (math.isnan(value) or math.isinf(value)):
                print(value)
                t = 2

    df_united = df_united2.iloc[0:9052]
    clf.fit(df_united, labels)
    #df_test_new = df_united2.iloc[9052:14052]
    df_test_new = df_united2.iloc[0:9052]
    predictions = clf.predict(df_test_new)
    #cn = 0
    #count = 0
    #for val in predictions:
        #if val == labels[cn]:
            #cn = cn + 1
            #count = count + 1
    
    #summary = count/9051
    #q = 1
#--------------------------------------------------------------CICID-------------------------------------------------------

    df_trainCi = pandas.read_csv('newdataset.csv')
    df_testCi = pandas.read_csv('test(100).csv')
    
    
    labelsCi = df_trainCi['Label']
        

    df_trainCi.drop(columns = ['Label'], inplace = True)
 
    t = 0
    clfCi = RandomForestClassifier(
	bootstrap=True, class_weight=None, criterion='gini',
	max_depth=17, max_features=10, max_leaf_nodes=None,
	min_impurity_decrease=0.0, min_impurity_split=None,
	min_samples_leaf=3, min_samples_split=2,
	min_weight_fraction_leaf=0.0, n_estimators=50,
	n_jobs=None, oob_score=False, random_state=1, verbose=0,
	warm_start=False)
    #for value in df_united2.iterrows():

    #print()
        #if (math.isnan(value[0]) or math.isinf(value[0])):
            #print(value[0])
            #t = 3
    clfCi.fit(df_trainCi, labelsCi)
    predictionsCi = clfCi.predict(df_trainCi)


    cn = 0
    count = 0
    for val in predictionsCi:
        if val == labelsCi[cn]:
            cn = cn + 1
            count = count + 1
    
    summary = count/100
    q = 1

    q = 1

