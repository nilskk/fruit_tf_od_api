#!/bin/bash

vott_data_dir="./data/voc_data"
vott_output_dir="./data/tfrecords"
vott_name="Mango"

learning_rate=0.02
batch_size=16
num_classes=14
train_steps=25000
warmup_steps=1000


python3 object_detection/create_vott_tfrecord.py --data_dir=$vott_data_dir \
                                                --output_dir=$vott_output_dir \
                                                --vott_sourceconnection_name=$vott_name \
                                                --set=train

python3 object_detection/create_vott_tfrecord.py --data_dir=$vott_data_dir \
                                                --output_dir=$vott_output_dir \
                                                --vott_sourceconnection_name=$vott_name \
                                                --set=val

model_path="./models/own_models/my_ssd_mobilenetv2_fpnlite"

python3 object_detection/change_pipeline_config.py --pipeline_config_path="${model_path}/pipeline.config" \
                                                  --label_map_path="${vott_data_dir}/pascal_label_map.pbtxt" \
                                                  --train_tfrecords_path="${vott_output_dir}/vott_train.tfrecord" \
                                                  --val_tfrecords_path="${vott_output_dir}/vott_val.tfrecord" \
                                                  --num_classes=$num_classes \
                                                  --batch_size=$batch_size \
                                                  --learning_rate=$learning_rate \
                                                  --train_steps=$train_steps \
                                                  --warmup_steps=$warmup_steps

python3 object_detection/training.py --pipeline_config_path="${model_path}/pipeline.config" \
                                    --model_dir="${model_path}/checkpoints_lr=${learning_rate}" &

training_pid=$!

python3 object_detection/evaluation.py --pipeline_config_path="${model_path}/pipeline.config" \
                                      --model_dir="${model_path}/checkpoints_lr=${learning_rate}" \
                                      --checkpoint_dir="${model_path}/checkpoints_lr=${learning_rate}" &

evaluation_pid=$!

trap onexit INT
function onexit(){
  kill -9 $training_pid
  kill -9 $evaluation_pid
}

wait

echo "Berechnung fertig für lr=${learning_rate}"


