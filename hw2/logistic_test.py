from logistic import load_iris_data, cross_validate, knn, nb, lr, logit
from numpy import arange

(XX,yy,y)=load_iris_data()

#classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb), ("Linear Regression",lr),("Logistic Regression",logit)]
classfiers_to_cv=[("Logistic Regression",logit)]

for (c_label, classifier) in classfiers_to_cv :
    k = int(raw_input('input your k-fold parameter: ',))
    print
    print "---> %s <---" % c_label

    best_k=0
    best_cv_a=0
    for a in arange(0.1,1.0,0.1):
       cv_a = cross_validate(XX, yy, classifier, k_fold=k,cparam=a)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=a

       print "C Regularization <<%s>> :: acc <<%s>>" % (a, cv_a)

    print "\n For k-fold=%s, %s Highest Accuracy: reg parameter <<%s>> :: <<%s>>\n" % (k,c_label, best_k, best_cv_a)

