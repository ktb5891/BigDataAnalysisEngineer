# 문제 1
## 분류예측 문제
## 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리, 피쳐엔지니어링, 분류 알고리즘 사용, 초매개변수 최적화, 모형 앙상블 등이 수반되어야 한다.
## 수험번호.csv 파일이 만들어지도록 코드를 제출한다.
## 제출한 모형의 성능은 ROC-AUC 평가지표에 따라 채점한다.
## predict_proba로 예측, 종속변수 survived열의 범주1 확률을 예측
## 데이터 파일 읽기 예제

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
df.head()

X_train, X_test, y_train, y_test = train_test_split(df,df['survived'], test_size = 0.2, random_state = 42, stratify = df['survived'])
X_train = X_train.drop(['alive','survived'], axis = 1)
X_test = X_test.drop(['alive','survived'], axis = 1)

# print(X_train.head())


# 1. 결측치 입력

# print(X_train.isna().sum()) # age는 평균, embarked, deck, embark_town은 많은 분포인 값으로 처리

# print('deck',X_train['deck'].value_counts()) # C
# print('embarked',X_train['embarked'].value_counts()) # S
# print('embark_town',X_train['embark_town'].value_counts()) # Southampton

missing = ['age']
for i in missing:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_test[i].mean())

X_train['deck'] = X_train['deck'].fillna('C')
X_test['deck'] = X_test['deck'].fillna('C')

X_train['embarked'] = X_train['embarked'].fillna('S')
X_test['embarked'] = X_test['embarked'].fillna('S')

X_train['embark_town'] = X_train['embark_town'].fillna('Southampton')
X_test['embark_town'] = X_test['embark_town'].fillna('Southampton')

# print(X_train.isna().sum())


# 2. 라벨 인코딩

from sklearn.preprocessing import LabelEncoder

label = ['sex','embarked','class','who','adult_male','deck','embark_town','alone']

X_train[label] = X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] = X_test[label].apply(LabelEncoder().fit_transform)

# print(X_train.head())


# 3. 데이터 타입 변환, 더미 변수

# print(X_train.dtypes)

dtype = ['pclass','sex','class']

for i in X_train[dtype]:
    X_train[i] = X_train[i].astype('category')
    X_test[i] = X_test[i].astype('category')

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# print(X_train.head())
# print(X_test.head())


# 4. 파생 변수

X_train['age_qcut'] = pd.qcut(X_train['age'],5,labels = False)
X_test['age_qcut'] = pd.qcut(X_test['age'],5,labels = False)

# print(X_train.head())
# print(X_test.head())


# 5. 스케일

from sklearn.preprocessing import MinMaxScaler

scaler = ['age','fare']

min = MinMaxScaler()
min.fit(X_train[scaler])

X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])

# print(X_train.head())
# print(X_test.head())


# 6. 데이터 분리

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size = 0.2, random_state = 42, stratify = y_train)

# print(X_train.shape) # (569,19)
# print(X_valid.shape) # (143,19)


# 7. 모형학습, 앙상블

## 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(X_train,y_train)
pred1 = pd.DataFrame(model1.predict_proba(X_valid))

## 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier()
model2.fit(X_train,y_train)
pred2 = pd.DataFrame(model2.predict_proba(X_valid))

## 앙상블 보팅
from sklearn.ensemble import VotingClassifier
model3 = VotingClassifier(estimators = [('logistic',model1),('randomforest',model2)], voting = 'soft')
model3.fit(X_train, y_train)
pred3 = pd.DataFrame(model3.predict_proba(X_valid))

# print(pred3)


# 9.모형 평가

from sklearn.metrics import roc_auc_score

print('로지스틱',roc_auc_score(y_valid,pred1.iloc[:,1]))
print('랜덤 포레스트',roc_auc_score(y_valid,pred2.iloc[:,1]))
print('앙상블',roc_auc_score(y_valid,pred3.iloc[:,1]))


# 10. 하이퍼파라미터 튜닝

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[50,100], 'max_depth':[4,6]}
model5 = RandomForestClassifier()
clf = GridSearchCV(estimator = model5, param_grid = parameters, cv = 3)
clf.fit(X_train, y_train)
print('최적의 파라미터 : ', clf.best_params_)


# 11. 파일 저장

result = pd.DataFrame(model3.predict_proba(X_test))
result = result.iloc[:,1]
pd.DataFrame({'id':y_test.index, 'pred':result}).to_csv('202205111615.csv',index = False)


# 확인

check = pd.read_csv('202205111615.csv')
print(check.head())