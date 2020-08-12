FROM python:3.6

RUN pip install pipenv

ENV PROJECT_DIR /Users/giorgoskarantonis/Desktop/label-bot

WORKDIR ${PROJECT_DIR}

COPY Pipfile Pipfile.lock ${PROJECT_DIR}/

RUN pipenv install --system --deploy --ignore-pipfile

CMD [ "python", "./app.py" ]