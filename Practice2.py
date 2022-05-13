# 문제 2
## 회귀예측 문제
## 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리, 피쳐엔지니어링, 분류 알고리즘 사용, 초매개변수 최적화, 모형 앙상블 등이 수반되어야 한다.
## 수험번호.csv 파일이 만들어지도록 코드를 제출한다.
## 제출한 모형의 성능은 RMSE, MAE가 평가지표에 따라 채점한다.
## 종속변수 mpg

## 데이터 파일 읽기 예제
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = sns.load_dataset('mpg')

X_train, X_test, y_train, y_test = train_test_split(df, df['mpg'], test_size = 0.2, random_state = 42)
X_train = X_train.drop(['mpg'], axis = 1)
X_test = X_test.drop(['mpg'], axis = 1)

# print(X_train.head())

# 1. 결측치 입력

# print(X_train.isna().sum())

missing = ['horsepower']

for i in missing:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_test[i].mean())
    
# print(X_train.isna().sum())

# 2. 라벨 인코딩

from sklearn.preprocessing import LabelEncoder

# print(X_train.info())

label = ['origin','name']

X_train[label] = X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] = X_test[label].apply(LabelEncoder().fit_transform)

# print(X_train)

# 3. 카테고리 변환, 더미 변수

category = ['origin']

for i in category:
    X_train[i] = X_train[i].astype('category')
    X_test[i] = X_test[i].astype('category')
    
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# print(X_train.head())

# 4. 파생변수

X_train['horsepower_qcut'] = pd.qcut(X_train['horsepower'],5,labels = False)
X_test['horsepower_qcut'] = pd.qcut(X_test['horsepower'],5,labels = False)

# 5. 스케일 작업

from sklearn.preprocessing import MinMaxScaler

scaler = ['displacement','horsepower','weight']

min = MinMaxScaler()
min.fit(X_train[scaler])

X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])

# print(X_train.head())


# 6. 데이터 분리

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# print(X_train.shape)
# print(X_test.shape)

# 7. 모형 학습

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_valid)

from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_valid)

# 8. 앙상블(스태킹)

from sklearn.ensemble import StackingRegressor

estimators = [('lr',model1),('rf',model2)]
model3 = StackingRegressor(estimators = estimators, final_estimator = RandomForestRegressor())
model3.fit(X_train, y_train)
pred3 = model3.predict(X_valid)

# print(pred3)


# 9. 모형평가

from sklearn.metrics import mean_squared_error

print('선형회귀 MSE : ',mean_squared_error(y_valid,pred1))
print('랜포 MSE : ',mean_squared_error(y_valid,pred2))
print('스태킹 MSE : ',mean_squared_error(y_valid,pred3))

print('선형회귀 RMSE : ',np.sqrt(mean_squared_error(y_valid,pred1)))
print('랜포 RMSE : ',np.sqrt(mean_squared_error(y_valid,pred2)))
print('스태킹 RMSE : ',np.sqrt(mean_squared_error(y_valid,pred3)))

# 10. 하이퍼파라미터 튜닝

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[50,100],'max_depth':[4,6]}
model4 = RandomForestRegressor()
clf = GridSearchCV(estimator = model4, param_grid = parameters, cv = 3)
clf.fit(X_train, y_train)

# print('최적의 파라미터 : ',clf.best_params_)

#11. 파일 저장
# print(X_test.head())
# print(X_train.head())
result = pd.DataFrame(model2.predict(X_test))
result = result.iloc[:,0]
pd.DataFrame({'id':X_test.index,'result':result}).to_csv('202205121424.csv',index = False)

check = pd.read_csv('202205121424.csv')
print(check.head())