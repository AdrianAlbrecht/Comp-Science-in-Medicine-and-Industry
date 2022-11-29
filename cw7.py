from __future__ import print_function
import copy
from tabulate import tabulate

# =============================================================== TEST =============================================================================================

# matrix_trn = [
#     [1,3,1,2,1],
#     [1,4,1,2,1],
#     [2,4,1,4,0],
#     [3,5,1,4,0]
# ] 
    
# matrix_tst_with_dec = [
#     [1,4,1,3,1],
#     [3,4,1,1,0],
#     [2,5,2,4,0]
# ]

# matrix =  matrix_trn + matrix_tst_with_dec

# when k=1, metric = Manhattan 
# distance(ob_i,ob_j) = SUM(abs(ob_i[a_k]-ob_j[a_k]))

# =============================================================== DATA =============================================================================================

matrix = []
with open("heart_disese.dat","r") as file:
    matrix = [list(map(lambda a: float(a),line.split())) for line in file]
    
split_length = (len(matrix)/3)*2 # 0.6 split

#here split table to 0.6 to trn/tst

matrix_trn = copy.deepcopy(matrix[:split_length])
matrix_tst_with_dec = copy.deepcopy(matrix[split_length:])
  
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

matrix_tst = copy.deepcopy(matrix_tst_with_dec)

def manhattan_distance(ob1, ob2):
    distance = 0
    for i in range(0,len(ob1)-1):
        distance += abs(ob1[i] - ob2[i])
    return distance

def k_nn(matrix_trn,matrix_tst,k):
    for i in range(0,len(matrix_tst)):
        k_distance = []
        for j in range(0,len(matrix_trn)):
            k_distance.append([manhattan_distance(matrix_trn[j],matrix_tst[i])]+copy.deepcopy(matrix_trn[j]))
        k_distance.sort()
        k_decision = copy.deepcopy(k_distance[:k])
        t_no_1 = 0.0
        t_no_2 = 0.0
        sum_no_1 = 0.0
        sum_no_2 = 0.0
        for ele in k_decision:
            if ele[-1] == matrix_dec_class[0]:
                t_no_1 +=1
                sum_no_1 +=ele[1]
            else:
                t_no_2 +=1
                sum_no_2 +=ele[1]
        try:
            mean_1 = sum_no_1/t_no_1
        except:
            mean_1 = 999999999999999999999999999.0
        try:
            mean_2 = sum_no_2/t_no_2
        except:
            mean_2 = 999999999999999999999999999.0
        if mean_1<mean_2:
            matrix_tst[i][-1] = matrix_dec_class[0]
        else:
            matrix_tst[i][-1] = matrix_dec_class[1]
        
k_nn(matrix_trn,matrix_tst,1)

# =============================================================== k == 1 =============================================================================================

# for i in range(0,len(matrix_tst)):
#     smallest_distance = 999999999999999999
#     decision = 9999999
#     for j in range(0,len(matrix_trn)):
#         temp_dist = manhattan_distance(matrix_trn[j],matrix_tst[i])
#         if temp_dist < smallest_distance:
#             smallest_distance = temp_dist
#             decision = matrix_trn[j][-1]
#     matrix_tst[i][-1] = decision
    
# print(matrix_tst_with_dec)
# print(matrix_tst) 

# ====================================================================================================================================================================
        
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
