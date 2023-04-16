#!/usr/bin/env bash
# usage: ./run.sh 

# -------- BEGIN: DON'T CHANGE --------
export HADOOP_HOME=$HOME/hadoop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native

echo "Hadoop homedir: $HADOOP_HOME"

SPARK_SUBMIT="spark-submit"
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ETL_SCRIPT="$PROJECT_DIR/kmeans.py"
# -------- END: DON'T CHANGE --------

master_url=`curl -s http://169.254.169.254/latest/meta-data/public-hostname`

$SPARK_SUBMIT \
    --conf spark.eventLog.enabled=true \
    --deploy-mode client \
    --master spark://$master_url:7077 \
    --driver-memory 6g \
    --executor-memory 6g \
    $ETL_SCRIPT \
    $1 \
    $2 \
    $3 \
    $4 \
    $5 \
    $6 \
    $7 \
    $8
