from __future__ import print_function
import copy
from random import randint

# tp = 5
# tn = 25.6
# fp = 5.2
# fn = 9.8
# no_positive_class = 53.2
# no_negative_class = 99.8

# precision = tp/(tp+fn) #positive accuracy
# specifity = tn/(tn+fp) #negative accuracy
# total_accuracy = (tn+tp)/(tp+fn+tn+fp)
# balance_accuracy = (specifity+precision)/2
# recall = tp/(tp+fp) #sensitivity #True positive rate
# true_negative_rate = tn/(tn+fn)
# coverage_positive = (tp+fn)/no_positive_class
# coverage_negative = (tn+fp)/no_negative_class
# total_coverage = (tp+fn+tn+fp)/(no_positive_class+no_negative_class)
# f1_score = (2*precision*recall)/(precision+recall)

# print("Precision: "+str(precision),"Specifity: "+str(specifity),"Total accuracy: "+str(total_accuracy),"Balance accuracy: "+str(balance_accuracy), "Recall: "+str(recall),
#       "True negative rate: "+str(true_negative_rate),"Coverage positive: "+str(coverage_positive),"Coverage negative: "+str(coverage_negative),
#       "Total coverage: "+str(total_coverage),"F1 score: "+str(f1_score),sep='\n')


# print(randint(0,100))

# ============================================= Random rulette wheel exmaple =================================================================

# matrix = [0 for x in range(0,500)]+[1 for x in range(0,500)]
# new_matrix = []
# no_0 = 0;
# no_1 = 0;
# for x in range(0,len(matrix)-1):
#     new_matrix.append(matrix[randint(0,len(matrix)-1)])
#     if(new_matrix[x]==0):
#         no_0 +=1
#     else:
#         no_1 +=1

# print(no_0,no_1)

# ============================================= Random rulette wheel heart diesese =================================================================
with open("heart_disese.dat","r") as file:
    matrix = [list(map(lambda a: float(a),line.split())) for line in file]
    
def rulette_wheel(procent, first_class_value, second_class_value):
    return [first_class_value for x in range(0,int(1000*(1-procent)))]+[second_class_value for x in range(0,int(1000*procent))]
    
# print(len(matrix[0])) # 14 czesci, ostatnia to klasa
no_1 = 0
no_2 = 0
for line in matrix:
    if line[13]==1:
        no_1+=1
    else:
        no_2+=1
# print(no_1, no_2) 1: 150, 2: 120

rand_class = rulette_wheel(0.99,1,2)

new_matrix = copy.deepcopy(matrix)
for line in new_matrix:
    line[13] = rand_class[randint(0,999)]
    
no_1_nm = 0
no_2_nm = 0
for line in new_matrix:
    if line[13]==1:
        no_1_nm+=1
    else:
        no_2_nm+=1
        
# print(no_1, no_1_nm)
# print(no_2, no_2_nm)

positive_class = None
no_positive_class = None
negative_class = None
no_negative_class =None

tp = 0
tn = 0
fp = 0
fn = 0

if(no_1>=no_2):
    positive_class = 2
    negative_class = 1
    no_positive_class = no_2
    no_negative_class = no_1
else:
    positive_class = 1
    negative_class = 2
    no_positive_class = no_1
    no_negative_class = no_2
    
    
for i in range(0,len(matrix)): 
    if matrix[i][13] == new_matrix[i][13]:
        if matrix[i][13] == positive_class:
            tp+=1                               #positive => positive
        else:
            tn+=1                               #negative => negative
    else:
        if matrix[i][13] == positive_class:
            fn+=1                               #positive => negative
        else:
            fp+=1                               #negative => positive
  
tp=float(tp)
tn=float(tn)
fp=float(fp)
fn=float(fn)
no_negative_class=float(no_negative_class)
no_positive_class=float(no_positive_class)
          
precision = round(tp/(tp+fn),8) #positive accuracy
specifity =round( tn/(tn+fp),8) #negative accuracy
total_accuracy = round((tn+tp)/(tp+fn+tn+fp),8)
balance_accuracy = round((specifity+precision)/2,8)
recall =round(tp/(tp+fp),8) #sensitivity #True positive rate
true_negative_rate = round(tn/(tn+fn),8)
coverage_positive = round((tp+fn)/no_positive_class,8)
coverage_negative = round((tn+fp)/no_negative_class,8)
total_coverage = round((tp+fn+tn+fp)/(no_positive_class+no_negative_class),8)
f1_score = round((2*precision*recall)/(precision+recall),8)

print(positive_class,negative_class)
print("TP: "+str(tp),"TN: "+str(tn),"FP: "+str(fp),"FN: "+str(fn),sep='\n')
print("No. of positive: "+str(no_positive_class),"No. of negative: "+str(no_negative_class),sep="\n")
print("Precision: "+str(precision),"Specifity: "+str(specifity),"Total accuracy: "+str(total_accuracy),"Balance accuracy: "+str(balance_accuracy), "Recall: "+str(recall),
      "True negative rate: "+str(true_negative_rate),"Coverage positive: "+str(coverage_positive),"Coverage negative: "+str(coverage_negative),
      "Total coverage: "+str(total_coverage),"F1 score: "+str(f1_score),sep='\n')