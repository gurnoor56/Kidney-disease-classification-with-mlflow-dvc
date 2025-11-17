FROM python:3.9-slim

RUN apt update -y and install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python", "app.py"]