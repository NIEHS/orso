#!/bin/sh

LOGFILE="$LOGS_PATH/celerybeat.log"

# wait for migrations
sleep 10

exec /usr/local/bin/celery beat \
    --app=genomics_network \
    --loglevel=INFO \
    --logfile=$LOGFILE
