# 도커와 FastAPI를 활용해서 언어감지 모델을 배포해보자
## 0. 학습 참고자료
- [Deploy ML models with FastAPI, Docker, and Heroku](https://www.youtube.com/watch?v=h5wLuVDr0oc&t=188)

## 1. 프로젝트 주제
언어감지 데이터 셋을 kaggle에서 구해서 외국어 텍스트를 인식하고 어느 나라 언어인지 감지하는 모델을 웹에 배포하기
## 2. (E)LT & Model Design
### 2-1. (E)LT
Kaggle에 있는 [데이터](https://www.kaggle.com/datasets/basilb2s/language-detection)를 활용해서 데이터를 저장
- 학습한 데이터는 배포에는 활용하지 않는다
### 2-2. Model Design
모델학습에 쓰인 메인 과정만 소개하고 넘어감 이후 깃헙에 올린 데이터 참고
[링크](https://github.com/amitis94/heroku)
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([('vectorizer', cv), ('multinomialNB', model)])
pipe.fit(X_train, y_train)

# 웹에 올리기 위한 모델 찌부시키기
with open('trained_pipeline-0.1.0.pkl', 'wb') as f:
    pickle.dump(pipe, f)
```
## 3. Dockerfile 만들기
도커 이미지를 빌드해서 FastAPI로 쓰기 위해서 공식문서에서 따라 쓰라고 만든 코드 복붙하고 app폴더 접근경로만 추가해주기
```Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app
```
### 참고. requirements.txt
도커 컨테이너에도 모델이 동작할 수 있는 환경을 마련해주기 위해서 파일이 동작했던 환경을 얼려서 가져오기
`pip freeze > requirements.txt`

## 4. app 빌드하기

```python
from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_verison

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str

@app.get('/')
def home():
    return {"health_check": "OK", "model_version": model_verison}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"언어는? ": language}
```
pydantic은 데이터 유효성 검사에 쓰이는 라이브러리라고 한다
짜둔 코드로 비교해보니 TextIn으로 받는 text는 문자형태만 인식하도록 규칙을 정하고 그렇지 않으면 오류를 보내는 기능으로 이해했다

## 5. 헤로쿠의 유료화..로 배포 느낌만 느껴보기

Heroku가 작년 11월 말부터 무료로 쓸 수 없게 됐다..

일단 지금까지 설계한 파일을 도커 이미지로 빌드하고 돌려보면 80:80으로 연결한 로컬 주소에서 돌아가는 모델을 볼 수 있다

도커 컨테이너를 쓰기 위한 커멘드만 짚고 넘어가면 좋을 것 같아서 가져옴
`heroku git:remote {설계한 모델 이름}`
`heroku stack:set container` << 이 부분이 도커 컨테이너로 돌린다는 명령어
