from hw1 import *
import argparse


(XX,yy,y)=load_iris_data() #assigns variables to iris data set

classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb)]

for (c_label, classifier) in classfiers_to_cv :

 
    print  "---> %s <----" % c_label

    best_k=0
    best_cv_a=0
    for k_f in range(2,150):
        cv_a = cross_validate(XX, yy, classifier, k_fold=k_f)
        if cv_a > best_cv_a :
            best_cv_a = cv_a
            best_k = k_f

        print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)

    print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)    
        
    
    
