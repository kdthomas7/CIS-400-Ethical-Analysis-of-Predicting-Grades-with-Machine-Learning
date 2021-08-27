
import csv
#import eli5
import yellowbrick
#import lime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
#from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

if __name__ == '__main__':

    ### convert csv to array
    #format: attr = [studytime, failures, schoolsup, activities, internet, absences]
    #column numbers:    13         14         15        18          21        29
    #res = [G1, G2, G3]


    csv_arr = []
    with open("cis400_mini_proj/student_grades/student-mat.csv") as csvfile:
        reader = csv.reader(csvfile, )
        for row in reader: # each row is a list
            csv_arr.append(row)

    feature_names = csv_arr.pop(0) #remove first row of csv that has row titles
    rel_feature_names = [r for r in feature_names if feature_names.index(r) in [13,14,15,18,21,29]] #list of feature names used in computations
    
    attr = [] #array with only the needed attributes

    attr = [[r[13], r[14], r[15], r[18], r[21], r[29]] for r in csv_arr]

    res = [] #array with the actual grades

    #res = [[r[30], r[31], r[32]] for r in csv_arr] #includes all 3 grades
    res = [r[32] for r in csv_arr] #includes only final grade


    ### convert any string/text attribute values to numeric

    # convert yes/no feature values to 1/0
    for r in attr:
        if r[2] == "yes":
            attr[attr.index(r)][2] = 1
        else:
            attr[attr.index(r)][2] = 0

        if r[3] == "yes":
            attr[attr.index(r)][3] = 1
        else:
            attr[attr.index(r)][3] = 0

        if r[4] == "yes":
            attr[attr.index(r)][4] = 1
        else:
            attr[attr.index(r)][4] = 0


    ### Compute class imbalance using Shannon entropy
    def balance1(seq):
        from collections import Counter
        from numpy import log
    
        n = len(seq)
        classes = [(clas,float(count)) for clas,count in Counter(seq).iteritems()]
        k = len(classes)
    
        H = -sum([ (count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
        return H/log(k)

    def balance(labels):
        from math import log, e
        value,counts = np.unique(labels, return_counts=True)
        norm_counts = counts / counts.sum()
        base = len(labels[0])
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

    print("Balance of attr set A: ")
    print(balance(attr))


    ### split data into trainng and testing set

    x_train, x_test, y_train, y_test = train_test_split(attr, res, random_state=0)

    #print(x_train)
    x = np.array(x_train)
    #print(x)
    #x = x.reshape(-1, 6)
    y = np.array(y_train)
    #print(y)

    #defining varibles for use in computing explanations
    x_ng_train = x
    y_ng_train = y
    x_ng_test = x_test
    y_ng_test = y_test

    ### Nearest neighbors

    clf_nn = NearestCentroid()

    clf_nn.fit(x, y) #fit with training data

    clf_nn_pred = clf_nn.predict(x_test) #test on testing data

    #print(clf_pred)

    clf_nn_acc = accuracy_score(clf_nn_pred, y_test) #compute accuracy

    print("##### Results for attribute set A (without gender) #####")
    print("nearest neighbors accuracy: " , clf_nn_acc)


    ### SVC/SVM

    clf_svm = svm.SVC()

    clf_svm.fit(x,y)

    clf_svm_pred = clf_svm.predict(x_test)

    clf_svm_acc = accuracy_score(clf_svm_pred, y_test)

    print("SVC/SVM accuracy: " , clf_svm_acc)


    ### KMeans

    #clf_km = KMeans(n_clusters=20, random_state=0)

    #clf_km.fit(x)

    #clf_km_pred = clf_km.predict(x_test) # labels
    #print("km labels: ", clf_km_pred)
    #print("y_test", y_test)

    #clf_km_acc = accuracy_score(clf_km_pred, y_test)

    #print("KMeans accuracy: " , clf_km_acc)


    ### Random Forest

    clf_rf = RandomForestClassifier(n_estimators=10)

    clf_rf.fit(x,y)

    clf_rf_pred = clf_rf.predict(x_test)

    #print("RF pred: ", set(zip(clf_rf_pred, y)))

    clf_rf_acc = accuracy_score(clf_rf_pred, y_test)

    print("Random forest accuracy: ", clf_rf_acc)




    ### compute with students gender inlcuded in feature vector
    rel_feature_names_g = [r for r in feature_names if feature_names.index(r) in [1, 13,14,15,18,21,29]] #list of feature names used in computations

    attr_g = [] #array with only the needed attributes

    attr_g = [[r[1], r[13], r[14], r[15], r[18], r[21], r[29]] for r in csv_arr]

    res = [] #array with the actual grades

    #res = [[r[30], r[31], r[32]] for r in csv_arr] #includes all 3 grades
    res = [r[32] for r in csv_arr] #includes only final grade


    ### convert any string/text attribute values to numeric

    # convert yes/no feature values to 1/0
    for r in attr_g:
        if r[3] == "yes":
            attr_g[attr_g.index(r)][3] = 1
        else:
            attr_g[attr_g.index(r)][3] = 0

        if r[4] == "yes":
            attr_g[attr_g.index(r)][4] = 1
        else:
            attr_g[attr_g.index(r)][4] = 0

        if r[5] == "yes":
            attr_g[attr_g.index(r)][5] = 1
        else:
            attr_g[attr_g.index(r)][5] = 0

    # convert M or F gender into 0 or 1
    for r in attr_g:
        if r[0] == "F":
            attr_g[attr_g.index(r)][0] = 1
        else:
            attr_g[attr_g.index(r)][0] = 0

    ### Compute class imbalance using Shannon entropy
    
    print("Balance of attr set B: ")
    print(balance(attr_g))


    ### split data into trainng and testing set

    x_train, x_test, y_train, y_test = train_test_split(attr_g, res, random_state=0)

    #print(x_train)
    x = np.array(x_train)
    #print(x)
    x = x.reshape(-1, 7)
    y = np.array(y_train)
    #print(y)

    #defining varibles for use in computing explanations
    x_g_train = x
    y_g_train = y
    x_g_test = x_test
    y_g_test = y_test

    ### Nearest neighbors

    clf_nn_g = NearestCentroid()

    clf_nn_g.fit(x, y) #fit with training data

    clf_nn_g_pred = clf_nn_g.predict(x_test) #test on testing data

    #print(clf_pred)

    clf_nn_g_acc = accuracy_score(clf_nn_g_pred, y_test) #compute accuracy

    print("##### Results for attribute set B (with gender) #####")
    print("nearest neighbors accuracy: " , clf_nn_g_acc)


    ### SVC/SVM

    clf_svm_g = svm.SVC()

    clf_svm_g.fit(x,y)

    clf_svm_g_pred = clf_svm_g.predict(x_test)

    clf_svm_g_acc = accuracy_score(clf_svm_g_pred, y_test)

    print("SVC/SVM accuracy: " , clf_svm_g_acc)


    ### KMeans

    #clf_km_g = KMeans(n_clusters=2, random_state=0)

    #clf_km_g.fit(x)

    #clf_km_g_pred = clf_km_g.predict(x_test) # labels

    #clf_km_g_acc = accuracy_score(clf_km_g_pred, y_test)

    #print("KMeans accuracy: " , clf_km_g_acc)



    ### Random Forest

    clf_rf_g = RandomForestClassifier(n_estimators=10)

    clf_rf_g.fit(x,y)

    clf_rf_g_pred = clf_rf_g.predict(x_test)

    clf_rf_g_acc = accuracy_score(clf_rf_g_pred, y_test)

    print("Random forest accuracy: ", clf_rf_g_acc)



    ### Decision Tree
    def decision_tree():
        clf_dt_g = tree.DecisionTreeClassifier()
        #clf_dt_g = tree.DecisionTreeRegressor()
    
        clf_dt_g.fit(x,y)

        clf_dt_g_pred = clf_dt_g.predict(x_test)

        #tree.plot_tree(clf_dt_g) #plot decision tree
        #plt.show()

        import graphviz 
        dot_data = tree.export_graphviz(clf_dt_g, out_file=None, feature_names=rel_feature_names_g, filled=True) 
        graph = graphviz.Source(dot_data) 
        graph.render("decision_tree_classif_gender") 
    
    #decision_tree()


    ###### Plotting results of accuracy scores ######
    def plot_results():
        fig, ax = plt.subplots()

        fig = plt.figure()

        X = np.arange(3)

        #ax = fig.add_axes([0,0,1,1])

        width = 0.25

        rects1 = ax.bar(X - width/2, [round(clf_nn_acc,2),round(clf_svm_acc,2), round(clf_rf_acc,2)], color = 'b', width = width, label="set A") #accuracies on attribute set A

        rects2 = ax.bar(X + width/2, [round(clf_nn_g_acc,2),round(clf_svm_g_acc,2), round(clf_rf_g_acc,2)], color = 'g', width = width, label="set B") #accuracies on attribute set B

        labels = ["Nearest neighbors", "SVM/SVC", "Random forest"]

        ax.set_ylabel('Accuracy Scores')

        ax.set_xticks(np.arange(len(labels)))

        ax.set_xticklabels(labels)

        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.show()

    
    #plot_results()


    ###### Explanation Info ######

    ### Nearest neighbors
    #from yellowbrick.classifier import ClassificationReport
    #visualizer = ClassificationReport(clf_nn, is_fitted=True)
    #visualizer.fit(x_ng_train, y_ng_train)
    #visualizer.score(x_ng_test, y_ng_test)
    #visualizer.show()

    # correalation between features without gender
    def feat_corr_ng():
        from yellowbrick.features import Rank2D
        vis_r2d = Rank2D(algorithm='pearson', features=rel_feature_names)
        vis_r2d.fit(x_ng_train, y_ng_train)
        vis_r2d.transform(x_ng_train.astype(np.float))
        vis_r2d.show()
    


    # Note: nearest centroid and SVC do not have a feature importance parameter
    #svm feature importance without gender
    def feat_importance_ng():
        from yellowbrick.model_selection import FeatureImportances
        vis_fi = FeatureImportances(clf_rf, labels=rel_feature_names)
        vis_fi.fit(x_ng_train, y_ng_train)
        vis_fi.show()


    # correalation between features with gender
    def feat_corr_g():
        from yellowbrick.features import Rank2D
        vis_r2d_g = Rank2D(algorithm='pearson', features=rel_feature_names_g)
        vis_r2d_g.fit(x_g_train, y_g_train)
        vis_r2d_g.transform(x_g_train.astype(np.float))
        vis_r2d_g.show()

    # Note: nearest centroid and SVC do not have a feature importance parameter
    #svm feature importance with gender
    def feat_importance_g():
        from yellowbrick.model_selection import FeatureImportances
        vis_fi_g = FeatureImportances(clf_rf_g, labels=rel_feature_names_g)
        vis_fi_g.fit(x_g_train, y_g_train)
        vis_fi_g.show()


    #Classification report for attribute set B using svm classifier
    def classification_rep():
        from yellowbrick.classifier import ClassificationReport
        cr_classes = ['0', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '5', '6', '7', '8', '9']
        #cr_classes = rel_feature_names_g
        cr_vis = ClassificationReport(clf_svm_g, support=True)
        cr_vis.fit(x_g_train, y_g_train)
        cr_vis.score(x_g_test, y_g_test)
        cr_vis.show()  


    #print("#### ", y_g_train)
    def conf_matrix():
        from yellowbrick.classifier import ConfusionMatrix
        conf_vis = ConfusionMatrix(clf_svm_g, support=True)
        conf_vis.fit(x_g_train, y_g_train)
        conf_vis.score(x_g_test, y_g_test)
        conf_vis.show()  

    
    def class_pred_error():
        from yellowbrick.classifier import ClassPredictionError
        err_vis = ClassPredictionError(clf_svm_g)
        err_vis.fit(x_g_train, y_g_train)
        err_vis.score(x_g_test, y_g_test)
        err_vis.show()



    #feat_corr_ng()
    #feat_corr_g()
    #feat_importance_ng()
    #feat_importance_g()
    #print("##### classes for clf_svm_g: ", clf_svm_g.classes_)
    #classification_rep()
    #conf_matrix()
    #class_pred_error()