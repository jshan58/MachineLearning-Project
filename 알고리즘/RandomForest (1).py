from google.colab import drive
drive.mount('/content/drive')

import numpy as np


# 데이터 로드
x_train = np.load("/content/drive/MyDrive/eyes_data/X_train.npy")
x_test = np.load("/content/drive/MyDrive/eyes_data/X_test.npy")
y_train = np.load("/content/drive/MyDrive/eyes_data/y_train.npy")
y_test = np.load("/content/drive/MyDrive/eyes_data/y_test.npy")

# 데이터 확인
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 1. 기본 모델 적용
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 기본 RandomForestClassifier 모델 초기화
rf_model = RandomForestClassifier(random_state=42)

# 모델 학습 (훈련 데이터 전체 사용)
rf_model.fit(x_train, y_train)

# Train predictions
y_train_pred = rf_model.predict(x_train)

# Test predictions
y_pred = rf_model.predict(x_test)

# Train Accuracy 계산
train_accuracy = accuracy_score(y_train, y_train_pred)
print("\nTrain Accuracy:")
print(f"{train_accuracy:.4f}")

# Test Accuracy 계산
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")

#2 Randomized Search 이용해 최적의 하이퍼파라미터 조합 찾기

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# RandomForestClassifier 초기화
rf = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 30 ,50,100 ],
    'min_samples_split': [ 5 ,10],
    'min_samples_leaf': [3, 5 , 10]

}

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=25,  # 탐색 횟수
    cv=5,       # 교차 검증 폴드 수
    verbose=2,
    random_state=42
)

# 하이퍼파라미터 탐색 수행
random_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# 최적 모델 추출
best_rf = random_search.best_estimator_


#3 최적의 하이퍼파라미터 조합을 이용해 5-Folfd CV 진행 그리고 정확도 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 최적 하이퍼파라미터를 사용한 RandomForest 모델 초기화
best_rf_model = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=10,
    min_samples_leaf=3,
    max_depth=10,
    random_state=42
)

# Stratified K-Fold 교차 검증 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 교차 검증 수행
cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# 모델 학습 (훈련 데이터 전체 사용)
best_rf_model.fit(x_train, y_train)

# Test predictions
y_pred = best_rf_model.predict(x_test)

# 예측 확률 계산
y_proba = best_rf_model.predict_proba(x_test)

# 평가 지표 출력
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
ConfusionMatrixDisplay(conf_matrix).plot()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nTest Accuracy:")
test_accuracy = accuracy_score(y_test, y_pred)
print(f"{test_accuracy:.4f}")


#4 구해놓은 최적은 하이퍼파라미터 조합에서 n_estimator를 300으로 수정

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 최적 하이퍼파라미터를 사용한 RandomForest 모델 초기화
best_rf_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=3,
    max_depth=10,
    random_state=42
)

# Stratified K-Fold 교차 검증 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 교차 검증 수행
cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# 모델 학습 (훈련 데이터 전체 사용)
best_rf_model.fit(x_train, y_train)

# Test predictions
y_pred = best_rf_model.predict(x_test)

# 예측 확률 계산
y_proba = best_rf_model.predict_proba(x_test)

# 평가 지표 출력
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
ConfusionMatrixDisplay(conf_matrix).plot()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nTest Accuracy:")
test_accuracy = accuracy_score(y_test, y_pred)
print(f"{test_accuracy:.4f}")

# 6. 학습된 모델 저장
import joblib
model_path = "/content/drive/MyDrive/eyes_data/best_randomforest_model.pkl'"
joblib.dump(best_rf_model, model_path)
print(f"Model saved to {model_path}")
