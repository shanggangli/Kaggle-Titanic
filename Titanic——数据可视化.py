from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Pathtrain = r'C:\Users\...\Desktop\机器学习\kaggle\Titanic Machine Learning from Disaster\train.csv'
Traindata = pd.DataFrame(pd.read_csv(Pathtrain))
'''descr=Traindata.describe()
info=Traindata.info()
print(descr['Survived'])'''

'''fig=plt.figure()
plt.subplot2grid((2,3),(0,0))
Traindata.Survived.value_counts().plot(kind='bar')
plt.title('Rescue situation(Survived=1)')
plt.ylabel('numbers')

plt.subplot2grid((2,3),(0,1))
Traindata.Pclass.value_counts().plot(kind='bar')
plt.title('Pcalss situation')
plt.ylabel('numbers')

plt.subplot2grid((2,3),(0,2))
Traindata.Sex.value_counts().plot(kind='bar')
plt.title('Sex situation')
plt.ylabel('numbers')

plt.subplot2grid((2,3),(1,0))
Traindata.Embarked.value_counts().plot(kind='bar')
plt.title('Embarked situation')
plt.ylabel('numbers')

plt.subplot2grid((2,3),(1,1), colspan=2)
Traindata.Age[Traindata.Pclass == 1].plot(kind='kde')
Traindata.Age[Traindata.Pclass == 2].plot(kind='kde')
Traindata.Age[Traindata.Pclass == 3].plot(kind='kde')
plt.xlabel(u"Age")# plots an axis lable
plt.ylabel(u"density")
plt.title(u"Age and Pclass")
plt.legend((u'First class', u'Second class',u'Third class'),loc='best') # sets our legend for our graph.
plt.show()

#看看各乘客等级的获救情况
fig = plt.figure()
Survived_0 = Traindata.Pclass[Traindata.Survived == 0].value_counts()
Survived_1 = Traindata.Pclass[Traindata.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Survived and Pclass")
plt.xlabel(u"Pclass")
plt.ylabel(u"numbers")

#看看各乘客性别的获救情况
fig=plt.figure()
Survived_m=Traindata.Survived[Traindata.Sex=='male'].value_counts()
Survived_f=Traindata.Survived[Traindata.Sex=='female'].value_counts()
df=pd.DataFrame({u'male':Survived_m,u'female':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title('Survived and Sex')
plt.xlabel('Sex')
plt.ylabel('numbers')'''


#看看各乘客年龄的获救情况

Age=Traindata['Age']
plt.hist(Age,40)
Survived_Age=Traindata.Age[Traindata.Survived==1]
plt.hist(Survived_Age,40)
plt.xlabel("Age")
plt.ylabel("numbers")
plt.show()

#看看各等级仓的获救情况
'''fig=plt.figure()
plt.title("Survived and Pclass and Sex")
fig.set(alpha=0.65)

ax1=fig.add_subplot(141)
Traindata.Survived[Traindata.Sex=='female'][Traindata.Pclass!=3].value_counts().plot(kind='bar',label='Female highclass')
plt.legend(loc='best')
ax1.set_xticklabels(["Survived", "Unsurvived"])

ax2=fig.add_subplot(142,sharey=ax1)
Traindata.Survived[Traindata.Sex=='female'][Traindata.Pclass==3].value_counts().plot(kind='bar',label='Female lowclass')
plt.legend(loc='best')
ax2.set_xticklabels(["Survived", "Unsurvived"])

ax3=fig.add_subplot(143,sharey=ax1)
Traindata.Survived[Traindata.Sex=='male'][Traindata.Pclass!=3].value_counts().plot(kind='bar',label='male highclass')
plt.legend(loc='best')
ax3.set_xticklabels(["Survived", "Unsurvived"])

ax3=fig.add_subplot(144,sharey=ax1)
Traindata.Survived[Traindata.Sex=='male'][Traindata.Pclass==3].value_counts().plot(kind='bar',label='male lowclass')
plt.legend(loc='best')
ax3.set_xticklabels(["Survived", "Unsurvived"])
plt.show()'''

#看看登录港口获救情况
'''fig=plt.figure()
Survived_0=Traindata.Embarked[Traindata.Survived==1].value_counts()
Survived_1=Traindata.Embarked[Traindata.Survived==0].value_counts()
df=pd.DataFrame({'Survived':Survived_1,'Unsurvived':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title('Survived and Embarked')
plt.xlabel('Embarked')
plt.ylabel('numbers')
plt.show()'''

##按Cabin有无看获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_cabin = Traindata.Survived[pd.notnull(Traindata.Cabin)].value_counts()
Survived_nocabin = Traindata.Survived[pd.isnull(Traindata.Cabin)].value_counts()
df=pd.DataFrame({u'yes':Survived_cabin, u'no':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title("Survived and Cabin")
plt.legend(['Unsurvived','Survived'],loc='best')
plt.xlabel("Cabin yes or no")
plt.ylabel("numbers")
plt.show()


