#!/bin/bash
python create_vott_tf_record.py --data_dir=/home/tensorflow/fruit_tf_od_api/data/voc_data/ \
  --output_dir=/home/tensorflow/fruit_tf_od_api/data/tfrecords/ \
  --set=train \
  --vott_sourceconnection_name=Mango

python create_vott_tf_record.py --data_dir=/home/tensorflow/fruit_tf_od_api/data/voc_data/ \
  --output_dir=/home/tensorflow/fruit_tf_od_api/data/tfrecords/ \
  --set=val \
  --vott_sourceconnection_name=Mango