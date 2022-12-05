from __future__ import print_function
import copy
from tabulate import tabulate
from sklearn.neighbors import NearestNeighbors
import numpy as np

matrix = []
with open("heart_disese.dat","r") as file:
    matrix = [list(map(lambda a: float(a),line.split())) for line in file]
    
split_length = int((len(matrix)/3)*2) # 0.6 split

#here split table to 0.6 to trn/tst

matrix_tst_with_dec = copy.deepcopy(matrix[split_length:])
matrix_trn_with_dec = copy.deepcopy(matrix[:split_length])
matrix_trn = copy.deepcopy(matrix[:split_length])
for ele in matrix_trn:
    ele.pop()

matrix_tst = copy.deepcopy(matrix_tst_with_dec)
for ele in matrix_tst:
    ele.pop()

nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(matrix_trn)
distances, indices = nbrs.kneighbors(matrix_tst)

# print(indices)

for i in range(0,len(indices)):
    matrix_tst[i] = matrix_tst[i]+[matrix_trn_with_dec[indices[i][0]][-1]]
    
matrix_dec_class = []    
    
for ele in matrix:
    if ele[-1] not in matrix_dec_class:
        matrix_dec_class.append(ele[-1])
        
matrix_dec_class.sort()

no_1 = 0
no_2 = 0

for ele in matrix_tst_with_dec:
    if ele[-1]==matrix_dec_class[0]:
        no_1 +=1
    else:
        no_2 +=1
        
        
positive_class = None
no_positive_class = None
negative_class = None
no_negative_class =None

tp = 0
tn = 0
fp = 0
fn = 0
        
if(no_1>=no_2):
    positive_class = matrix_dec_class[1]
    negative_class = matrix_dec_class[0]
    no_positive_class = no_2
    no_negative_class = no_1
else:
    positive_class = matrix_dec_class[0]
    negative_class = matrix_dec_class[1]
    no_positive_class = no_1
    no_negative_class = no_2
    
for i in range(0,len(matrix_tst)): 
    if matrix_tst[i][-1] == matrix_tst_with_dec[i][-1]:
        if matrix_tst_with_dec[i][-1] == positive_class:
            tp+=1                               #positive => positive
        else:
            tn+=1                               #negative => negative
    else:
        if matrix_tst_with_dec[i][-1] == positive_class:
            fn+=1                               #positive => negative
        else:
            fp+=1                               #negative => positive
  
tp=float(tp)
tn=float(tn)
fp=float(fp)
fn=float(fn)
no_negative_class=float(no_negative_class)
no_positive_class=float(no_positive_class)
    
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

confusion_matrix=[
    ["", positive_class,negative_class,"No. of objects","Accuracy", "Coverage"],
    [int(positive_class), tp, fn, no_positive_class, precision, coverage_positive],
    [int(negative_class), fp, tn, no_negative_class, specifity, coverage_negative]
]

print(tabulate(confusion_matrix))

print( "Total coverage: "+str(total_coverage),"Total accuracy: "+str(total_accuracy),"Balance accuracy: "+str(balance_accuracy),
      "Recall: "+str(recall), "True negative rate: "+str(true_negative_rate), "F1 score: "+str(f1_score),sep='\n')
    

    