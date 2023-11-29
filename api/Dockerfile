FROM python:3.10-slim

WORKDIR /api

RUN apt-get update && apt-get --yes upgrade

COPY requirements.txt requirements.txt
RUN  pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

COPY . .

RUN chmod +x /api/docker-entrypoint.sh

ENTRYPOINT sh /api/docker-entrypoint.sh

EXPOSE 8000