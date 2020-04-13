from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
Pathtrain = r'C:\User\....\kaggle\Titanic Machine Learning from Disaster\train.csv'
Traindata = pd.DataFrame(pd.read_csv(Pathtrain))
#处理缺失值、异常值、正则化、中心化、标准化
'''
#是否有Cabin 卡方检验
Survived_cabin = Traindata.Survived[pd.notnull(Traindata.Cabin)].value_counts()
Survived_nocabin = Traindata.Survived[pd.isnull(Traindata.Cabin)].value_counts()
df=np.array([Survived_cabin,Survived_nocabin])
#print(df)
#print(df.shape)
print('Cabin 卡方检验',chi2_contingency(df))

#Pclass 卡方检验
Survived_firstclass=Traindata.Survived[Traindata.Pclass==1].value_counts()
Survived_Secondclass=Traindata.Survived[Traindata.Pclass==2].value_counts()
Survived_Thirdclass=Traindata.Survived[Traindata.Pclass==3].value_counts()
df2=np.array([Survived_firstclass,Survived_Secondclass,Survived_firstclass])
#print(df2)
print('Pclass卡方检验：',chi2_contingency(df2))

#Embarked 卡方检验
Survived_S=Traindata.Survived[Traindata.Embarked=='S'].value_counts()
Survived_C=Traindata.Survived[Traindata.Embarked=='C'].value_counts()
Survived_Q=Traindata.Survived[Traindata.Embarked=='Q'].value_counts()
df3=np.array([Survived_C,Survived_S,Survived_Q])
#print(df3)
print('Embarked 卡方检验',chi2_contingency(df3))

Survived_m=Traindata.Survived[Traindata.Sex=='male'].value_counts()
Survived_f=Traindata.Survived[Traindata.Sex=='female'].value_counts()
df4=np.array([Survived_m,Survived_f])
print('Sex 卡方检验：',chi2_contingency(df4))
'''
#分析年龄
#print("#######分析年龄##########")
from sklearn.ensemble import RandomForestRegressor
def set_missing_Age(Traindata):
    age_df=Traindata[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    # 目标年龄
    y=known_age[:,0]
    # 属性
    x=known_age[:,1:]
    # 调用模型
    model=RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    model.fit(x,y)
    # 用得到的模型进行未知年龄结果预测
    predictedAge=model.predict(unknown_age[:,1::])
    # 用得到的预测结果填补原缺失数据
    Traindata.loc[(Traindata.Age.isnull()),'Age']=predictedAge
    return Traindata,model

'''def set_Cabin_type(Traindata):
    Traindata.loc[ (Traindata.Cabin.notnull()), 'Cabin' ] = "Yes"
    Traindata.loc[ (Traindata.Cabin.isnull()), 'Cabin' ] = "No"
    return Traindata'''
Traindata,rfr=set_missing_Age(Traindata)
#Traindata=set_Cabin_type(Traindata)
#print(Traindata['Embarked'].value_counts())
Traindata.loc[(Traindata.Embarked.isnull()),'Embarked']='S'
#print(Traindata.info())
#print(Traindata)
#print(set_missing_Age())
import matplotlib.pyplot as plt
#print('#####')
'''Age_in_chlid=Traindata.Survived[(Traindata.Age<=10)].value_counts()
#print(Age_in_chlid)
Age_in_young=Traindata.Survived[((Traindata.Age>10)&(Traindata.Age<=20))].value_counts()
Age_in_adult=Traindata.Survived[((Traindata.Age>20)&(Traindata.Age<=30))].value_counts()
Age_in_midadult=Traindata.Survived[((Traindata.Age>30)&(Traindata.Age<=40))].value_counts()
Age_in_old=Traindata.Survived[((Traindata.Age>40)&(Traindata.Age<50))].value_counts()
Age_in_oldinold=Traindata.Survived[(Traindata.Age>50)].value_counts()
#print(Age_in_old)
df4=np.array([Age_in_chlid,Age_in_young,Age_in_adult,Age_in_midadult,Age_in_old,Age_in_oldinold])
#print(df4)
print('年龄段的卡方：',chi2_contingency(df4))

# 将数据分为2组，未成年和成年 进行卡方检验
Age_in_young1=Traindata.Survived[(Traindata.Age<18)].value_counts()
#print(Age_in_young1)
Age_not_in_young=Traindata.Survived[(Traindata.Age>18)].value_counts()
#print(Age_not_in_young)
df5=np.array([Age_in_young1,Age_not_in_young])
print('年龄段2的卡方：',chi2_contingency(df5))
'''
# 分段年龄图
'''df5=pd.DataFrame({'chlid':Age_in_chlid,'young':Age_in_young,'adult':Age_in_adult,'midadult':Age_in_midadult,'old':Age_in_old,'Oldinold':Age_in_oldinold})
df5.plot(kind='bar')
plt.show()'''

# 将Sex、Cabin、Pclass、Embarked 因子化
#dummies_Cabin = pd.get_dummies(Traindata['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(Traindata['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(Traindata['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(Traindata['Pclass'], prefix= 'Pclass')
df = pd.concat([Traindata, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
#print(df.head())

# 由于Age、Fare数据波动太大，不应于收敛，需要归一化
import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
# 归一化时，数据需要是(-1,1)形式
'''age_scale_param=scaler.fit(np.array(df['Age']).reshape(-1,1))
df['ScalerAge']=scaler.fit_transform(np.array(df['Age']).reshape(-1,1),age_scale_param)
'''
fare_scale_param = scaler.fit(np.array(df['Fare']).reshape(-1,1))
df['ScalerFare']=scaler.fit_transform(np.array(df['Fare']).reshape(-1,1),fare_scale_param)
#print(df)
# 将Age分为成年和未成年
df.loc[(df.Age<=15),'Age']=0
df.loc[(df.Age>15),'Age']=1
#print(df)
#
df.loc[(df.Cabin.isnull() == True), 'Cabin'] = 0.5
df.loc[(df.Cabin.isnull() == False), 'Cabin'] = 1.5

# 提出名字
def gettitle(name):
    str1=name.split(',')[1] #Mr. Owen Harris
    str2=str1.split('.')[0]# Mr
    str3=str2.strip()
    return str3
df['Title']=df['Name'].map(gettitle)
#print(df['Title'].value_counts())
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
}
dummies_Title = pd.get_dummies(df['Title'].map(title_mapDict), prefix= 'Title')
df=pd.concat([df,dummies_Title],axis=1)
df.drop(['Name'], axis=1, inplace=True)
#print(df)
df['Familysize']=df['SibSp']+df['Parch']+1
df['Family_Small']=df['Familysize'].map(lambda s:1 if s==1 else 0)
df['Family_Middle']=df['Familysize'].map(lambda s:1 if 2<=s<=4 else 0)
df['Family_Large']=df['Familysize'].map(lambda s:1 if s>=5 else 0)
df.drop(['Familysize','Embarked_Q','Title_Master','Title_Royalty','Title_Officer'],axis=1,inplace=True)
#print(df.head())

# 终于可以到建模了
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
train_df = df.filter(regex='Survived|Age|ScalerFare|Title_.*|Embarked_.*|Sex_.*|Pclass_.*|Family_*')
#print(train_df)
corrDf=df.corr()
print(corrDf['Survived'].sort_values(ascending=False))
train_np = train_df.values
#print(train_np)
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
model1=LogisticRegression(C=0.1,penalty='l2',solver='lbfgs')
score=cross_val_score(model1, X.astype('float64'), y.astype('int'), cv=5)
print(score)
#pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(model1.coef_.T)})
print(score.mean())
model1.fit(X,y.astype('int'))



# 处理测试数据
Pathtest = r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\Titanic Machine Learning from Disaster\test.csv'
Testdata = pd.DataFrame(pd.read_csv(Pathtest))
print(Testdata.info())
# 拟合确实Age、Cabin
tmp_df=Testdata[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[tmp_df.Age.isnull()].values
# 用于RandomTree 训练属性
X=null_age[:,1:]
PecdictedAge=rfr.predict(X)
# 补上Age缺失值
Testdata.loc[ (Testdata.Age.isnull()), 'Age' ] = PecdictedAge
#用均数补上Fare的缺失值
Testdata.loc[(Testdata.Fare.isnull()),'Fare']=Testdata['Fare'].mean()
# 补上Cabin缺失值
#Testdata=set_Cabin_type(Testdata)
# 补上Cabin的缺失值
Testdata.loc[(Testdata.Cabin.isnull() == True), 'Cabin'] = 0.5
Testdata.loc[(Testdata.Cabin.isnull() == False), 'Cabin'] = 1.5
def gettitle(name):
    str1=name.split(',')[1] #Mr. Owen Harris
    str2=str1.split('.')[0]# Mr
    str3=str2.strip()
    return str3

Testdata['Title']=Testdata['Name'].map(gettitle)
#print(Testdata['Title'].value_counts())
#print(df['Title'].value_counts())
title_mapDict1 = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
}

# 将Sex、Cabin、Embarked、Pclass，Title因子化
#dummies_Cabin = pd.get_dummies(Testdata ['Cabin'], prefix= 'Cabin')
dummies_Title1=pd.get_dummies(Testdata['Title'].map(title_mapDict1),prefix='Title')
print(Testdata['Title'].value_counts())
dummies_Embarked = pd.get_dummies(Testdata['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(Testdata['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(Testdata['Pclass'], prefix= 'Pclass')
df_test= pd.concat([Testdata, dummies_Embarked, dummies_Sex, dummies_Pclass,dummies_Title1], axis=1)
df_test.drop(['Pclass','Sex', 'Ticket', 'Embarked','Name'], axis=1, inplace=True)

# Age、Fare归一化
import sklearn.preprocessing as preprocessing
scaler1=preprocessing.StandardScaler()
'''age_scale_param=scaler.fit(np.array(df_test['Age']).reshape(-1,1))
df_test['ScalerAge']=scaler.fit_transform(np.array(df_test['Age']).reshape(-1,1),age_scale_param)'''
fare_scale_param1 = scaler1.fit(np.array(df_test['Fare']).reshape(-1,1))
df_test['ScalerFare']=scaler1.fit_transform(np.array(df_test['Fare']).reshape(-1,1),fare_scale_param1)
#print(df_test)
df_test.loc[(df_test.Age<=15),'Age']=0
df_test.loc[(df_test.Age>15),'Age']=1
#print(df_test.info())
#df_test.loc[(df_test.Cabin.isnull() == True), 'Cabin'] = 0.5
#df_test.loc[(df_test.Cabin.isnull() == False), 'Cabin'] = 1.5
df_test['Familysize']=df_test['SibSp']+df['Parch']+1
df_test['Family_Small']=df_test['Familysize'].map(lambda s:1 if s==1 else 0)
df_test['Family_Middle']=df_test['Familysize'].map(lambda s:1 if 2<=s<=4 else 0)
df_test['Family_Large']=df_test['Familysize'].map(lambda s:1 if s>=5 else 0)
df_test.drop(['Familysize','Embarked_Q','Title_Master','Title_Royalty','Title_Officer'],axis=1,inplace=True)
test = df_test.filter(regex='Age|ScalerFare|Title_.*|Embarked_.*|Sex_.*|Family_*|Pclass_.*')
#print(test.info())
predcition=model1.predict(test)
result=pd.DataFrame({'PassengerId':df_test['PassengerId'].values,'Survived':predcition})

result.to_csv(r'C:\Users\...\kaggle\Titanic Machine Learning from Disaster\mygender_submission2.csv',index=False)
