
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
from yellowbrick.target import ClassBalance

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


    #res = [[r[30], r[31], r[32]] for r in csv_arr] #includes all 3 grades
    m_res = [r[32] for r in csv_arr if r[1] == 'M'] #includes only final grade
    f_res = [r[32] for r in csv_arr if r[1] == 'F'] #includes only final grade


    ### split dataset into data for females and data for males ###

    male_attr = [[r[13], r[14], r[15], r[18], r[21], r[29]] for r in csv_arr if r[1] == 'M']

    female_attr = [[r[13], r[14], r[15], r[18], r[21], r[29]] for r in csv_arr if r[1] == 'F']

    rel_feature_names = [r for r in feature_names if feature_names.index(r) in [13,14,15,18,21,29]] #list of feature names used in computations



   


    ### convert any string/text attribute values to numeric

    # convert yes/no feature values to 1/0
    def convert_feat_to_num(arr):
        for r in arr:
            if r[2] == "yes":
                arr[arr.index(r)][2] = 1
            else:
                arr[arr.index(r)][2] = 0

            if r[3] == "yes":
                arr[arr.index(r)][3] = 1
            else:
                arr[arr.index(r)][3] = 0

            if r[4] == "yes":
                arr[arr.index(r)][4] = 1
            else:
                arr[arr.index(r)][4] = 0

        return arr

    male_attr = convert_feat_to_num(male_attr)
    female_attr = convert_feat_to_num(female_attr)
    print(male_attr)
    print(female_attr)

    ### Compute class imbalance using Shannon entropy
    def balance(labels):
        from math import log, e
        value,counts = np.unique(labels, return_counts=True)
        norm_counts = counts / counts.sum()
        base = len(labels[0])
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

    
    print("Balance of male_attr: ")
    print(balance(male_attr))
    print("Balance of female_attr: ")
    print(balance(female_attr))

    #from yellowbrick.target import ClassBalance
    #class_bal_vis = ClassBalance(labels=[])
    #class_bal_vis.fit()
    #class_bal_vis.show()
    
    
    #### Predicting male scores ####

    ### split data into trainng and testing set

    x_train, x_test, y_train, y_test = train_test_split(male_attr, m_res, random_state=0)

    #print(x_train)
    x = np.array(x_train)
    #print(x)
    #x = x.reshape(-1, 7)
    y = np.array(y_train)
    #print(y)

    #defining varibles for use in computing explanations
    x_m = x
    y_m = y
    x_m_train = x
    y_m_train = y
    x_m_test = x_test
    y_m_test = y_test

    ### Nearest neighbors

    clf_nn_m = NearestCentroid()

    clf_nn_m.fit(x, y) #fit with training data

    clf_nn_m_pred = clf_nn_m.predict(x_test) #test on testing data

    #print(clf_pred)

    clf_nn_m_acc = accuracy_score(clf_nn_m_pred, y_test) #compute accuracy

    print("##### Results for male score prediction #####")
    print("nearest neighbors accuracy: " , clf_nn_m_acc)


    ### SVC/SVM

    clf_svm_m = svm.SVC()

    clf_svm_m.fit(x,y)

    clf_svm_m_pred = clf_svm_m.predict(x_test)

    clf_svm_m_acc = accuracy_score(clf_svm_m_pred, y_test)

    print("SVC/SVM accuracy: " , clf_svm_m_acc)


    ### Random Forest

    clf_rf_m = RandomForestClassifier(n_estimators=10)

    clf_rf_m.fit(x,y)

    clf_rf_m_pred = clf_rf_m.predict(x_test)

    clf_rf_m_acc = accuracy_score(clf_rf_m_pred, y_test)

    print("Random forest accuracy: ", clf_rf_m_acc)




    #### Predicting female scores ####

    ### split data into trainng and testing set

    x_train, x_test, y_train, y_test = train_test_split(female_attr, f_res, random_state=0)

    #print(x_train)
    x = np.array(x_train)
    #print(x)
    #x = x.reshape(-1, 7)
    y = np.array(y_train)
    #print(y)

    #defining varibles for use in computing explanations
    x_f = x
    y_f = y
    x_f_train = x
    y_f_train = y
    x_f_test = x_test
    y_f_test = y_test

    ### Nearest neighbors

    clf_nn_f = NearestCentroid()

    clf_nn_f.fit(x, y) #fit with training data

    clf_nn_f_pred = clf_nn_f.predict(x_test) #test on testing data

    #print(clf_pred)

    clf_nn_f_acc = accuracy_score(clf_nn_f_pred, y_test) #compute accuracy

    print("##### Results for female score prediction #####")
    print("nearest neighbors accuracy: " , clf_nn_f_acc)


    ### SVC/SVM

    clf_svm_f = svm.SVC()

    clf_svm_f.fit(x,y)

    clf_svm_f_pred = clf_svm_f.predict(x_test)

    clf_svm_f_acc = accuracy_score(clf_svm_f_pred, y_test)

    print("SVC/SVM accuracy: " , clf_svm_f_acc)


    ### Random Forest

    clf_rf_f = RandomForestClassifier(n_estimators=10)

    clf_rf_f.fit(x,y)

    clf_rf_f_pred = clf_rf_f.predict(x_test)

    clf_rf_f_acc = accuracy_score(clf_rf_f_pred, y_test)

    print("Random forest accuracy: ", clf_rf_f_acc)




    ###### Plotting results of accuracy scores ######
    def plot_results():
        fig, ax = plt.subplots()

        fig = plt.figure()

        X = np.arange(3)

        #ax = fig.add_axes([0,0,1,1])

        width = 0.25

        rects1 = ax.bar(X - width/2, [round(clf_nn_m_acc,2),round(clf_svm_m_acc,2), round(clf_rf_m_acc,2)], color = 'b', width = width, label="Male scores") #accuracies on attribute set A

        rects2 = ax.bar(X + width/2, [round(clf_nn_f_acc,2),round(clf_svm_f_acc,2), round(clf_rf_f_acc,2)], color = 'g', width = width, label="Female Scores") #accuracies on attribute set B

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

    
    plot_results()


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


    def class_balances():
        #compute average scores 


        f_score_avg = sum(list(map(int,f_res)))/len(list(map(int,f_res)))
        print("average scores of females: ", f_score_avg)

        m_score_avg = sum(list(map(int,m_res)))/len(list(map(int,m_res)))
        print("average scores of males: ", m_score_avg)

        #class balance visualizer for females
        class_bal_f = ClassBalance()
        class_bal_f.fit(y_f)
        class_bal_f.show()

        #class balance visualizer for males
        class_bal_m = ClassBalance()
        class_bal_m.fit(y_m)
        class_bal_m.show()



    #feat_corr_ng()
    #feat_corr_g()
    #feat_importance_ng()
    #feat_importance_g()
    #print("##### classes for clf_svm_g: ", clf_svm_g.classes_)
    #classification_rep()
    #conf_matrix()
    #class_pred_error()
    class_balances()