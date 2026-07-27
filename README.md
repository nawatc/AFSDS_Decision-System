# AFSDS_Decision-System

## Install pip package

```
pip install -r requirements.txt
```

## Usage Program via Terminal

```
python3 -m uvicorn main:app --reload
```

## Usage Program via Docker
```
test_docker_build-run.bat
```

or

```
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app
```

## Test POST api via cURL

```
curl_test.bat
```


## Docs
### FastAPI Swagger Docs
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

