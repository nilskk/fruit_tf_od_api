#!/bin/bash

trap onexit INT
function onexit(){
  kill -9 $training_pid
  kill -9 $evaluation_pid
  kill -term $$
}

vott_data_dir="./data/voc_data"
vott_output_dir="./data/tfrecords"
vott_name="Mango"

learning_rate=0.0001
batch_size=8
num_classes=5
train_epochs=150
optimizer='adam'
first_decay_epochs=10

#model_dir="mobilenetv2_fpnlite_notl"
model_dir="efficientdet_d0_notl"
model_path="./models/own_models/${model_dir}"
save_path="${model_path}/lr=${learning_rate}_bs=${batch_size}_classes=${num_classes}_adam_eps=1e-4"


python3 create_vott_tfrecord.py --data_dir=$vott_data_dir \
                                                --output_dir=$vott_output_dir \
                                                --vott_sourceconnection_name=$vott_name \
                                                --set=train

python3 create_vott_tfrecord.py --data_dir=$vott_data_dir \
                                                --output_dir=$vott_output_dir \
                                                --vott_sourceconnection_name=$vott_name \
                                                --set=val


python3 change_pipeline_config.py --pipeline_config_path="${model_path}/pipeline.config" \
                                                  --model_dir="${save_path}/checkpoints" \
                                                  --label_map_path="${vott_data_dir}/pascal_label_map.pbtxt" \
                                                  --train_tfrecords_path="${vott_output_dir}/vott_train.tfrecord" \
                                                  --val_tfrecords_path="${vott_output_dir}/vott_val.tfrecord" \
                                                  --num_classes=$num_classes \
                                                  --batch_size=$batch_size \
                                                  --optimizer=$optimizer \
                                                  --learning_rate=$learning_rate \
                                                  --train_epochs=$train_epochs \
                                                  --first_decay_epochs=$first_decay_epochs


python3 training.py --pipeline_config_path="${model_path}/pipeline.config" \
                                    --model_dir="${save_path}/checkpoints" \
                                    --train_tfrecords_path="${vott_output_dir}/vott_train.tfrecord" \
                                    --batch_size=$batch_size \
                                    --checkpoint_every_n_epochs=10 &

training_pid=$!

python3 evaluation.py --pipeline_config_path="${model_path}/pipeline.config" \
                                      --model_dir="${save_path}/checkpoints" \
                                      --checkpoint_dir="${save_path}/checkpoints" &

evaluation_pid=$!

wait

echo "Berechnung fertig!"

python3 object_detection/exporter_main_v2.py --pipeline_config_path="${model_path}/pipeline.config" \
                                            --trained_checkpoint_dir="${save_path}/checkpoints" \
                                            --output_directory="${save_path}/export"

echo "Model exportiert!"

python3 inference_savedmodel.py --label_map_path="${vott_data_dir}/pascal_label_map.pbtxt" \
                                --export_dir="${save_path}/export"

python3 calculate_flops.py --export_dir="${save_path}/export"



