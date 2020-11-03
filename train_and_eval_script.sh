#!/bin/bash
python training.py --pipeline_config_path="./_fruit_kaggle_project/models/my_ssd_mobilenetv2_fpnlite/pipeline.config" --model_dir="./_fruit_kaggle_project/models/my_ssd_mobilenetv2_fpnlite" &

python evaluation.py --pipeline_config_path="./_fruit_kaggle_project/models/my_ssd_mobilenetv2_fpnlite/pipeline.config" --model_dir="./_fruit_kaggle_project/models/my_ssd_mobilenetv2_fpnlite" --checkpoint_dir="./_fruit_kaggle_project/models/my_ssd_mobilenetv2_fpnlite" &