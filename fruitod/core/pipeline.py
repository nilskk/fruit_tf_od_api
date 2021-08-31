from fruitod.utils.file_util import read_config, write_config, get_steps_per_epoch


class Pipeline:
    def __init__(self, config_path, model_type):
        self.config_path = config_path
        self.model_type = model_type

    def set_model_name(self,
                       model_name):
        pipeline = read_config(self.config_path)

        pipeline.model.name = model_name

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_num_classes(self,
                        num_classes):
        pipeline = read_config(self.config_path)

        if self.model_type == 'ssd':
            pipeline.model.ssd.num_classes = num_classes
        elif self.model_type == 'prob_two_stage':
            pipeline.model.prob_two_stage.num_classes = num_classes

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
                      scheduler_name,
                      first_decay_epochs,
                      train_epochs,
                      steps_per_epoch,
                      learning_rate):
        pipeline = read_config(self.config_path)

        if optimizer_name == 'adam':
            optimizer = pipeline.train_config.optimizer.adam_optimizer
            pipeline.train_config.optimizer.adam_optimizer.epsilon = 1e-8
        elif optimizer_name == 'sgd':
            optimizer = pipeline.train_config.optimizer.momentum_optimizer
            pipeline.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

        if scheduler_name == 'restart':
            scheduler = optimizer.learning_rate.cosine_restart_learning_rate
            scheduler.first_decay_steps = first_decay_epochs * steps_per_epoch
            scheduler.initial_learning_rate = learning_rate
        elif scheduler_name == 'decay':
            scheduler = optimizer.learning_rate.cosine_decay_learning_rate
            scheduler.learning_rate_base = learning_rate
            scheduler.warmup_steps = first_decay_epochs * steps_per_epoch
            scheduler.total_steps = train_epochs * steps_per_epoch

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

    def set_weight_information(self,
                               add_weight_information=False,
                               weight_method='input',
                               add_weight_as_output=False,
                               add_weight_as_output_v2=False):
        pipeline = read_config(self.config_path)

        if self.model_type == 'ssd':
            pipeline.model.ssd.add_weight_information = add_weight_information
            pipeline.model.ssd.weight_method = weight_method
            pipeline.model.ssd.add_weight_as_output = add_weight_as_output
            pipeline.model.ssd.add_weight_as_output_v2 = add_weight_as_output_v2
        elif self.model_type == 'prob_two_stage':
            pipeline.model.prob_two_stage.add_weight_information = add_weight_information
            pipeline.model.prob_two_stage.weight_method = weight_method
            pipeline.model.prob_two_stage.add_weight_as_outputv1 = add_weight_as_output

        write_config(pipeline=pipeline, config_path=self.config_path)




