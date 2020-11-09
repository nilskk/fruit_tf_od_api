#!/bin/bash

vott_data_dir="./data/voc_data"
vott_output_dir="./data/tfrecords"
vott_name="Mango"

python object_detection/create_vott_tfrecord.py --data_dir=$vott_data_dir \
                                                --output_dir=$vott_output_dir \
                                                --vott_sourceconnection_name=$vott_name \
                                                --set=train

python object_detection/create_vott_tfrecord.py --data_dir=$vott_data_dir \
                                                --output_dir=$vott_output_dir \
                                                --vott_sourceconnection_name=$vott_name \
                                                --set=val

model_path="./models/own_models/my_ssd_mobilenetv2_fpnlite"

python object_detection/change_pipeline_config.py --pipeline_config_path="${model_path}/pipeline.config" \
                                                  --label_map_path="${vott_data_dir}/pascal_label_map.pbtxt" \
                                                  --train_tfrecords_path="${vott_output_dir}/vott_train.tfrecord" \
                                                  --val_tfrecords_path="${vott_output_dir}/vott_val.tfrecord" \
                                                  --num_classes=14 \
                                                  --batch_size=16 \
                                                  --train_steps=20000 \
                                                  --warmup_steps=2000

python object_detection/training.py --pipeline_config_path="${model_path}/pipeline.config" \
                                    --model_dir="${model_path}/checkpoints" &

python object_detection/evaluation.py --pipeline_config_path="${model_path}/pipeline.config" \
                                      --model_dir="${model_path}/checkpoints" \
                                      --checkpoint_dir="${model_path}/checkpoints" &


