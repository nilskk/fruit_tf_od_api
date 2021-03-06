from fruitod.utils.file_util import read_config, write_config, get_steps_per_epoch


class Pipeline:
    def __init__(self, config_path):
        self.config_path = config_path

    def set_model_name(self,
                       model_name):
        pipeline = read_config(self.config_path)

        pipeline.model.name = model_name

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_num_classes(self,
                        num_classes):
        pipeline = read_config(self.config_path)

        pipeline.model.ssd.num_classes = num_classes

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_batch_size(self,
                       batch_size):
        pipeline = read_config(self.config_path)

        pipeline.train_config.batch_size = batch_size

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_train_epochs(self,
                         train_epochs,
                         steps_per_epoch):
        pipeline = read_config(self.config_path)

        pipeline.train_config.num_steps = train_epochs * steps_per_epoch

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_optimizer(self,
                      optimizer_name,
                      first_decay_epochs,
                      steps_per_epoch,
                      learning_rate):
        pipeline = read_config(self.config_path)

        if optimizer_name == 'adam':
            optimizer = pipeline.train_config.optimizer.adam_optimizer
            pipeline.train_config.optimizer.adam_optimizer.epsilon = 1e-8
        elif optimizer_name == 'sgd':
            optimizer = pipeline.train_config.optimizer.momentum_optimizer
            pipeline.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

        optimizer.learning_rate.cosine_restart_learning_rate.first_decay_steps = first_decay_epochs * steps_per_epoch
        optimizer.learning_rate.cosine_restart_learning_rate.initial_learning_rate = learning_rate

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_labelmap(self,
                     labelmap_path):
        pipeline = read_config(self.config_path)

        pipeline.train_input_reader.label_map_path = labelmap_path
        pipeline.eval_input_reader[0].label_map_path = labelmap_path

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_train_tfrecord(self,
                           train_tfrecord_path):
        pipeline = read_config(self.config_path)

        pipeline.train_input_reader.tf_record_input_reader.input_path[0] = train_tfrecord_path

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_val_tfrecord(self,
                         val_tfrecord_path):
        pipeline = read_config(self.config_path)

        pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = val_tfrecord_path

        write_config(pipeline=pipeline, config_path=self.config_path)




