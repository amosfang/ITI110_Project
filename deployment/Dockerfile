# docker build -t mntest1 .
# docker run -p 8080:8080 mntest1
# http://localhost:8080/v1/app/models/mntest1:predict
# docker exec -it 5928c9d96255 /bin/bash
# docker run -t --rm -p 8501:8501 \
#    -v "$(pwd)/models/:/models/" tensorflow/serving \
#    --model_config_file=/models/models.config \
#    --model_config_file_poll_wait_seconds=60

FROM tensorflow/serving

COPY . /app

WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["tensorflow_model_server", "--model_config_file=/app/model_config.conf"]

CMD ["--rest_api_port=8080","--port=8081"]