FROM python:3.6.1

ENV PYTHONUNBUFFERED 1

RUN groupadd -r django && \
    useradd -r -g django django

COPY ./compose/django/entrypoint.sh /entrypoint.sh
COPY ./compose/django/gunicorn.sh /gunicorn.sh
COPY ./compose/django/celery.sh /celery.sh
COPY ./compose/django/celerybeat.sh /celerybeat.sh
COPY ./compose/django/cull_backups.py /cull_backups.py
COPY ./compose/django/supervisord.conf /supervisord.conf
COPY ./compose/django/fonts /usr/share/fonts/truetype/django

# 1) install start-scripts
# 2) install supervisord on python2 (using pip)
# 3) install phantomjs
# 4) install fonts
# 3) create logs path
RUN chmod +x /entrypoint.sh && \
    chmod +x /gunicorn.sh && \
    chmod +x /celery.sh && \
    chmod +x /celerybeat.sh && \
    chmod +x /supervisord.conf && \
    chown django /entrypoint.sh && \
    chown django /gunicorn.sh && \
    chown django /celery.sh && \
    chown django /celerybeat.sh && \
    chown django /supervisord.conf && \
    \
    wget -O /get-pip.py "https://bootstrap.pypa.io/get-pip.py" && \
    python2.7 /get-pip.py --disable-pip-version-check --no-cache-dir && \
    pip2 install supervisor && \
    rm /get-pip.py && \
    \
    apt-get update && \
    apt-get install -y fonts-liberation && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -fv && \
    su django -c "fc-cache -fvs" && \
    \
    curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get install -y nodejs && \
    \
    mkdir -p /app/logs && \
    mkdir -p /app/public/media

COPY ./requirements /requirements
RUN pip3 install \
        -r /requirements/production.txt \
        --no-cache-dir

COPY ./analysis /network/analysis
COPY ./assets /network/assets
COPY ./fixtures /network/fixtures
COPY ./genomics_network /network/genomics_network
COPY ./network /network/network
COPY ./static /network/static
COPY ./templates /network/templates
COPY manage.py /network/manage.py
COPY package.json /network/package.json
COPY webpack.base.js /network/webpack.base.js
COPY webpack.config.production.js /network/webpack.config.production.js

RUN gzip -df /network/fixtures/annotation.json.gz

RUN mkdir -p /network/data/
RUN mkdir -p /network/data/bigwig_temp

COPY ontologies.tar.gz /network/data
RUN tar -xzf /network/data/ontologies.tar.gz -C /network/data/

RUN chown -R django /network
RUN chown -R django /app

WORKDIR /network

RUN npm install && \
    npm run build

ENTRYPOINT ["/entrypoint.sh"]
