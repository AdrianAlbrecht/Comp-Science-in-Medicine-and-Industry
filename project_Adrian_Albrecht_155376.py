import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
import operator
import copy
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import interpolate

# Random forest with model Cross-validation

#configuration parameters
number_of_folds = 8
test_split_size = 0.2

# 1) Find proper data
matrix = []
# This dat is modified special for the project. I removed some values and changed decission class from numerical value to symbolic.
with open("heart_disese_project.dat","r") as file:
    for line in file: 
        line = line.rstrip()
        new_line = []
        splitter = ", "
        if(len(line.split(splitter))==1):
            splitter = "; "
        if(len(line.split(splitter))==1):
            splitter = " "
        splitted_line = line.split(splitter)
        for attrib_no in range(0,len(splitted_line)):
            try:
                if(attrib_no==len(splitted_line)-1):
                    new_line.append(int(splitted_line[attrib_no]))
                else:
                    new_line.append(float(splitted_line[attrib_no]))
            except:
                if splitted_line[attrib_no] == " ":
                    new_line.append(None)
                else:
                    new_line.append(str(splitted_line[attrib_no]))
        matrix.append(new_line)       
# 2) Apply preporcessing:
no_of_objects = len(matrix)
no_of_attributes = len(matrix[0])
print("=================================================================================")
print("Sample data of dataset before preprocessing:")
print(matrix[0])
print(matrix[1])
# For numerical data:
# - normalization,
# - standarization,
# - discretization if needed.
# For symbolic data:
# - create Dummy vairables if needed,
for j in range(0,no_of_attributes):
    no_numerical_data = dict()
    for i in range(0,no_of_objects):
        if(type(matrix[i][j]) == type("")):
            if(matrix[i][j] in no_numerical_data.keys()):
                no_numerical_data[matrix[i][j]].append(i)
            else:
                no_numerical_data[matrix[i][j]] = [i]
    k = 0
    for l in no_numerical_data.keys():
        for i in no_numerical_data[l]:
            matrix[i][j] = k
        k=k+1
# - missing values absorption.
for j in range(0,no_of_attributes):
    list_of_None = []
    unique_values = dict()
    for i in range(0,no_of_objects):
        if(matrix[i][j]==None):
            list_of_None.append(i)
        elif(matrix[i][j] in unique_values.keys()):
            unique_values[matrix[i][j]] = unique_values[matrix[i][j]] + 1
        else:
            unique_values[matrix[i][j]] = 1
    unique_values = sorted(unique_values.items(),key=operator.itemgetter(1),reverse=True)
    for i in list_of_None:
        matrix[i][j] = unique_values[0][0]
        
matrix_dec_class = []    
    
for ele in matrix:
    if ele[-1] not in matrix_dec_class:
        matrix_dec_class.append(ele[-1])
        
if matrix_dec_class != [0,1]:
    for ele in matrix:
        if ele[-1]==matrix_dec_class[0]:
            ele[-1]=0
        else:
            ele[-1]=1
    matrix_dec_class = [0,1]
print("=================================================================================")        
print("Sample data of dataset after preprocessing:")
print(matrix[0])
print(matrix[1])
# 3) Split data according to model Cross-validation
def CrossValidationMethod(matrix, no_of_split:int, split:float):
    tst = []
    trn = []
    for x in range(0, no_of_split):
        test, train = train_test_split(matrix, test_size=split,  random_state=x)
        tst.append(copy.deepcopy(test))
        trn.append(copy.deepcopy(train))
    return trn,tst

tst, trn = CrossValidationMethod(matrix, number_of_folds, test_split_size)
for i in range(0, number_of_folds):
    for j in [x for x in range(0, number_of_folds) if x!=i]:
        assert(tst[i]!=tst[j])
        assert(trn[i]!=trn[j]) 
# 4) Learn model
def bin_cm_params(cm):
    tp=cm[0][0]
    fn=cm[0][1]
    fp=cm[1][0]
    tn=cm[1][1]
    try:          
        precision = round(tp/(tp+fn),8) #positive accuracy
    except:
        precision = 0 
    try:
        specifity =round( tn/(tn+fp),8) #negative accuracy
    except:
        specifity = 0
    try:
        total_accuracy = round((tn+tp)/(tp+fn+tn+fp),8)
    except:
        total_accuracy = 0
    try:
        balance_accuracy = round((specifity+precision)/2,8)
    except:
        balance_accuracy = 0
    try:
        recall =round(tp/(tp+fp),8) #sensitivity #True positive rate
    except:
        recall = 0
    try:
        true_negative_rate = round(tn/(tn+fn),8)
    except:
        true_negative_rate = 0
    try:
        coverage_positive = round((tp+fn)/no_positive_class,8)
    except:
        coverage_positive = 0
    try:
        coverage_negative = round((tn+fp)/no_negative_class,8)
    except:
        coverage_negative = 0
    try:
        total_coverage = round((tp+fn+tn+fp)/(no_positive_class+no_negative_class),8)
    except:
        total_coverage = 0
    try:
        f1_score = round((2*precision*recall)/(precision+recall),8)
    except:
        f1_score = 0
    try:
        g_mean = round(math.sqrt(recall*specifity),8)
    except:
        g_mean = 0
    return tp,fn,fp,tn,precision,specifity,total_accuracy,balance_accuracy,recall,true_negative_rate,coverage_positive,coverage_negative,total_coverage,f1_score,g_mean

tp_CV = []
fn_CV = []
fp_CV = []
tn_CV = []
no_p_CV = []
no_n_CV = []
roc_CV = []
fpr_CV = []
tpr_CV = []

no_rows = math.ceil((number_of_folds+1)/5)
fig, ax = plt.subplots(no_rows, 5, figsize=(10,2*no_rows),num='ROC',constrained_layout=True)
for x in range(0,no_rows):
    for y in range(0,5):
        ax[x,y].set_axis_off()
l_colors = [(k, v) for k, v in mcolors.TABLEAU_COLORS.items()]
def get_color(i:int):
    i = i%len(l_colors)
    return l_colors[i][1]

mean_fpr = np.linspace(0, 1, 1000)
row=0
column=-1

positive_class = None
negative_class = None

no_1 = 0
no_2 = 0
    
for ele in matrix:
    if ele[-1]==matrix_dec_class[0]:
        no_1 +=1
    else:
        no_2 +=1

if(no_1>=no_2):
    positive_class = matrix_dec_class[1]
    negative_class = matrix_dec_class[0]
else:
    positive_class = matrix_dec_class[0]
    negative_class = matrix_dec_class[1]

for x in range(0, number_of_folds):
    column=column+1
    if(column==5):
        row=row+1
        column=0
    train = trn[x]
    test = tst[x]
    y_train=[]
    for y in train:
        y_train.append(y[-1])
    y_test=[]
    for y in test:
        y_test.append(y[-1])
    X_train=[]
    for y in train:
        X_train.append(y[:-1])
    X_test=[]
    for y in test:
        X_test.append(y[:-1])
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
              
    no_positive_class = 0
    no_negative_class = 0
        
    for ele in y_test:
        if ele==positive_class:
            no_positive_class +=1
        else:
            no_negative_class +=1
          
    classifier = RandomForestClassifier(n_estimators=10,max_depth=4,criterion='entropy',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    tp,fn,fp,tn,precision,specifity,total_accuracy,balance_accuracy,recall,true_negative_rate,coverage_positive,coverage_negative,total_coverage,f1_score, g_mean = bin_cm_params(cm)
    roc_score = round(roc_auc_score(y_test,y_pred),8)
    fpr, tpr, thresholds = roc_curve(y_test,y_pred)
    f_t = interpolate.PchipInterpolator(fpr, tpr)
    f_f = interpolate.PchipInterpolator(tpr, fpr)
    interp_tpr = f_t(mean_fpr)
    interp_fpr = f_f(mean_fpr)
    try:
        ax[row,column].plot(interp_fpr, interp_tpr, color=get_color(x), label="ROC fold "+str(x+1)+".")
        ax[row,column].plot(mean_fpr, mean_fpr, "--")
        ax[row,column].set_title("ROC fold "+str(x+1)+".")
        ax[row,column].set_xlabel('False Positive Rate')
        ax[row,column].set_ylabel('True Positive Rate')
        ax[row,column].set_axis_on()
    except:
        ax[column].plot(interp_fpr, interp_tpr, color=get_color(x), label="ROC fold "+str(x+1)+".")
        ax[column].plot(mean_fpr, mean_fpr, "--")
        ax[column].set_title("ROC fold "+str(x+1)+".")
        ax[column].set_xlabel('False Positive Rate')
        ax[column].set_ylabel('True Positive Rate')
        ax[column].set_axis_on()
    tp_CV.append(tp)
    fn_CV.append(fn)
    fp_CV.append(fp)
    tn_CV.append(tn)
    no_p_CV.append(no_positive_class)
    no_n_CV.append(no_negative_class)
    roc_CV.append(roc_score)
    fpr_CV.append(fpr)
    tpr_CV.append(tpr)
    print("Confusion matrix for "+str(x+1)+". fold:")
    conf_matrix=[
    ["", positive_class,negative_class,"No. of objects","Accuracy", "Coverage"],
    [int(positive_class), tp, fn, no_positive_class, precision, coverage_positive],
    [int(negative_class), fp, tn, no_negative_class, specifity, coverage_negative]
    ]
    print(tabulate(conf_matrix))
# 5) Create confusion matrix
tp=float(sum(tp_CV))
tp=tp/len(tp_CV)
fn=float(sum(fn_CV))
fn=fn/len(fn_CV)
fp=float(sum(fp_CV))
fp=fp/len(fp_CV)
tn=float(sum(tn_CV))
tn=tn/len(tn_CV)
no_positive_class = float(sum(no_p_CV))
no_positive_class = no_positive_class/len(no_p_CV)
no_negative_class = float(sum(no_n_CV))
no_negative_class = no_negative_class/len(no_n_CV)
roc_score = float(sum(roc_CV))
roc_score = round(roc_score/len(roc_CV),8)
fpr = [0 for x in range(len(fpr_CV[0]))]
for x in fpr_CV:
    for y in range(0,len(x)):
        fpr[y]=fpr[y]+x[y]
all_fpr = []
for x in fpr:
     all_fpr.append(x/len(fpr_CV))
tpr = [0 for x in range(len(tpr_CV[0]))]
for x in tpr_CV:
    for y in range(0,len(x)):
        tpr[y]=tpr[y]+x[y]
all_tpr = []
for x in tpr:
    all_tpr.append(x/len(tpr_CV))
cm = [[tp,fn],[fp,tn]]
tp,fn,fp,tn,precision,specifity,total_accuracy,balance_accuracy,recall,true_negative_rate,coverage_positive,coverage_negative,total_coverage,f1_score,g_mean = bin_cm_params(cm)
print("=================================================================================")
print("Confusion matrix for mean Cross-validation split:")
conf_matrix=[
["", positive_class,negative_class,"No. of objects","Accuracy", "Coverage"],
[int(positive_class), tp, fn, no_positive_class, precision, coverage_positive],
[int(negative_class), fp, tn, no_negative_class, specifity, coverage_negative]
]
print(tabulate(conf_matrix))
# 6) Compute Accuracy, Balanced accuracy, Coverage, precision, recall, F1 Score
print( "Total coverage: "+str(total_coverage),"Total accuracy: "+str(total_accuracy),"Balance accuracy: "+str(balance_accuracy),
      "Recall: "+str(recall), "True negative rate: "+str(true_negative_rate), "F1 score: "+str(f1_score),sep='\n')
# 7) If possible apply ROC, PR-curve, G-Mean,
print("ROC AUC score: "+str(roc_score), "G_mean: "+str(g_mean), sep='\n')
f_t = interpolate.PchipInterpolator(all_fpr, all_tpr)
f_f = interpolate.PchipInterpolator(all_tpr, all_fpr)
interp_tpr = f_t(mean_fpr)
interp_fpr = f_f(mean_fpr)
column=column+1
if(column==5):
    row=row+1
    column=0
try:
    ax[row,column].plot(interp_fpr, interp_tpr, color=get_color(number_of_folds), label="Mean ROC")
    ax[row,column].plot(mean_fpr, mean_fpr, "--")
    ax[row,column].set_title("Mean ROC")
    ax[row,column].set_xlabel('False Positive Rate')
    ax[row,column].set_ylabel('True Positive Rate')
    ax[row,column].set_axis_on()
except:
    ax[column].plot(interp_fpr, interp_tpr, color=get_color(number_of_folds), label="Mean ROC")
    ax[column].plot(mean_fpr, mean_fpr, "--")
    ax[column].set_title("Mean ROC")
    ax[column].set_xlabel('False Positive Rate')
    ax[column].set_ylabel('True Positive Rate')
    ax[column].set_axis_on()
plt.show()
print("=================================================================================")
# 8) Compare with random (.5)
def rulette_wheel(procent, first_class_value, second_class_value):
    return [first_class_value for x in range(0,int(1000*(1-procent)))]+[second_class_value for x in range(0,int(1000*procent))]

rand_class = rulette_wheel(0.5,positive_class,negative_class)
train_r, test_r = train_test_split(matrix, test_size=test_split_size,  random_state=0)
new_test_r = copy.deepcopy(test_r)
for x in new_test_r:
    x[-1] = rand_class[random.randint(0,999)]

no_positive_class = 0
no_negative_class = 0
       
for ele in test_r:
    if ele[-1]==positive_class:
        no_positive_class +=1
    else:
        no_negative_class +=1

tp_r = 0
tn_r = 0
fp_r = 0
fn_r = 0
  
for i in range(0,len(test_r)): 
    if test_r[i][-1] == new_test_r[i][-1]:
        if test_r[i][-1] == positive_class:
            tp_r+=1                               #positive => positive
        else:
            tn_r+=1                               #negative => negative
    else:
        if test_r[i][-1] == positive_class:
            fn_r+=1                               #positive => negative
        else:
            fp_r+=1
            
cm_r = [[tp_r,fn_r],[fp_r,tn_r]]
tp_r,fn_r,fp_r,tn_r,precision_r,specifity_r,total_accuracy_r,balance_accuracy_r,recall_r,true_negative_rate_r,coverage_positive_r,coverage_negative_r,total_coverage_r,f1_score_r,g_mean_r = bin_cm_params(cm_r)
conf_matrix=[
["", positive_class,negative_class,"No. of objects","Accuracy", "Coverage"],
[int(positive_class), tp_r, fn_r, no_positive_class, precision_r, coverage_positive_r],
[int(negative_class), fp_r, tn_r, no_negative_class, specifity_r, coverage_negative_r]
]
print(tabulate(conf_matrix))
print( "Total coverage: "+str(total_coverage_r),"Total accuracy: "+str(total_accuracy_r),"Balance accuracy: "+str(balance_accuracy_r),
      "Recall: "+str(recall_r), "True negative rate: "+str(true_negative_rate_r), "F1 score: "+str(f1_score_r),sep='\n')
print("G_mean: "+str(g_mean_r), sep='\n')
# 9) Create raport
print("=================================================================================")
print("Comparison random .5 with RandomForest Cross-Validation:")
total_random = 0
total_forest = 0
comparison = []
line=["Feature","Winner"]
comparison.append(line)
line=["------------------","------------------------------"]
comparison.append(line)
line=[]
line.append("Precision")
if(precision<precision_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1
comparison.append(line)
line=[]
line.append("Specifity")
if(specifity<specifity_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1
comparison.append(line)
line=[] 
line.append("Total Accuracy")
if(total_accuracy<total_accuracy_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1
comparison.append(line)
line=[]  
line.append("Balance accuracy")
if(balance_accuracy<balance_accuracy_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1
comparison.append(line)
line=[]  
line.append("Positive Coverage")
if(coverage_positive<coverage_positive_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1
comparison.append(line)
line=[]   
line.append("Negative Coverage")
if(coverage_negative<coverage_negative_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1  
comparison.append(line)
line=[]
line.append("Total Coverage")
if(total_coverage<total_coverage_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1   
comparison.append(line)
line=[]
line.append("Recall")
if(recall<recall_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1  
comparison.append(line)
line=[]
line.append("True Negative Rate")
if(true_negative_rate<true_negative_rate_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1  
comparison.append(line)
line=[]
line.append("F1 score")
if(f1_score<f1_score_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1
comparison.append(line)
line=[]
line.append("G-mean")
if(g_mean<g_mean_r):
    line.append("Random 0.5")
    total_random = total_random + 1
else:
    line.append("Cross-Validation Random Forest")
    total_forest = total_forest + 1 
print(tabulate(comparison))
    
print("TOTAL RESULT:")
print("Cross-Validation Random Forest "+str(total_forest)+":"+str(total_random)+" Random 0.5")
print("Better is:           ",end="")
if(total_random>total_forest):
    print("Random 0.5")
    total_random = total_random + 1
else:
    print("Cross-Validation Random Forest")
    total_forest = total_forest + 1 
# 10) Present (posible to do it via MS Teams)
