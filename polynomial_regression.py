import random
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

# mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

# myline = numpy.linspace(1, 22, 100)
# print("R-squared value:")
# print(r2_score(y, mymodel(x)))

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()

# print("=========================================================")
# print("Speed:")
# speed = mymodel(17)
# print(speed)

# print("=========================================================")
# x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
# y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

# mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

# myline = numpy.linspace(2, 95, 100)
# print("R-squared value:")
# print(r2_score(y, mymodel(x)))

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()

# print('mymodel(3) =',mymodel(3.5))
# mymodel.coef


x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
x = [float(i) for i in x]
x_r = numpy.reshape(x,(-1,1))
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
y = [float(i) for i in y]
y_r = y
myline = numpy.linspace(1, 22, 100)

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

x2 = [random.randrange(1,1000)*(21/1000)+1 for x in range(0,100)]
x2.sort()
x2_r = numpy.reshape(x2,(-1,1))
print(x2)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline),"y-")
plt.scatter(x2, mymodel(x2), color="green")
#plt.show()

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = abc.fit(x_r, y_r)
y_pred_ada_boost = model.predict(x2_r)
y_pred_polynomial_regress = mymodel(x2)

y_pred_ada_boost = [float(x) for x in y_pred_ada_boost]

plt.scatter(x2, y_pred_ada_boost, color="pink")

print(y_pred_polynomial_regress)
print(y_pred_ada_boost)
plt.show()




