FROM docker-regsitry.tencentcloudcr.com/wuwang/python:3.10-slim

USER root
WORKDIR /app
COPY . .
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt \
    && mkdir /app/temp

EXPOSE 8000/tcp
HEALTHCHECK CMD curl --fail http://localhost:8000/health
CMD ["python", "record_video_io.py"]
#  new time
#CMD ["gunicorn", "record_video_io:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "1800"]