FROM python:3.7
LABEL vigicovid.authors="lozanoAlvarez@gmail.com"

WORKDIR /app
COPY /app /app

# install python requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt


EXPOSE 8001

CMD [ "python3", "/app/src/app.py"]

