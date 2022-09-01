FROM python:3.8
WORKDIR /code/fastapi_app/
COPY ./requirements.txt /code/fastapi_app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/fastapi_app/requirements.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]