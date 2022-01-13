FROM python:3.8.5

WORKDIR /fastapi_app

COPY requirements.txt .

COPY ./app ./app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python", "./app/main.py", "./app/config.json"]
