from fruitod.utils.file_util import read_config, write_config, get_steps_per_epoch


class Pipeline:
    def __init__(self, config_path, model_type):
        """
            Constructor.
        Args:
            config_path: Path to config file
            model_type: type of model: 'ssd' for one-stage-architecture or 'prob_two_stage' for two-stage-architecture
        """
        self.config_path = config_path
        self.model_type = model_type

        if model_type not in ['ssd', 'prob_two_stage']:
            raise ValueError('String model_type must be one of: [ssd, prob_two_stage]')

    def set_model_name(self,
                       model_name):
        """
            Set Model Name in config file
        Args:
            model_name: Name of model
        """
        pipeline = read_config(self.config_path)

        pipeline.model.name = model_name

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_num_classes(self,
                        num_classes):
        """
            Set Number of classes (look at labelmap.pbtxt) in config file
        Args:
            num_classes: Number of classes (without background class)
        """
        pipeline = read_config(self.config_path)
        if self.model_type == 'ssd':
            pipeline.model.ssd.num_classes = num_classes
        elif self.model_type == 'prob_two_stage':
            pipeline.model.prob_two_stage.num_classes = num_classes

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_batch_size(self,
                       batch_size):
        """
            Set Training Batch Size in config file
        Args:
            batch_size: an Integer for the batch size used for training
        """
        pipeline = read_config(self.config_path)

        pipeline.train_config.batch_size = batch_size

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_train_epochs(self,
                         train_epochs,
                         steps_per_epoch):
        """
            Convert Epochs to Steps and write to config file
        Args:
            train_epochs: Number of Epochs to train the model
            steps_per_epoch: Number of training steps for one epoch
        """
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
        """
            Set Optimizer in config file
        Args:
            optimizer_name: String Name of Optimizer: either 'adam' or 'sgd'
            scheduler_name: String Name of Scheduler: either 'restart' for Cosine Decay with Restarts or
                'decay' for standard Cosine Decay
            first_decay_epochs: Number of epochs for Warmup Phase of schedulers
            train_epochs: Number of total train epochs
            steps_per_epoch: Number of training steps for one epoch
            learning_rate: Base Learning Rate for Training
        """
        pipeline = read_config(self.config_path)

        if optimizer_name not in ['adam', 'sgd']:
            raise ValueError('String optimizer_name must be one of: [adam, sgd]')

        if optimizer_name == 'adam':
            optimizer = pipeline.train_config.optimizer.adam_optimizer
            pipeline.train_config.optimizer.adam_optimizer.epsilon = 1e-8
        elif optimizer_name == 'sgd':
            optimizer = pipeline.train_config.optimizer.momentum_optimizer
            pipeline.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

        if scheduler_name not in ['restart', 'decay']:
            raise ValueError('String optimizer_name must be one of: [restart, decay]')
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
        """
            Set Label Map Path in config file
        Args:
            labelmap_path: Path of 'label_map.pbtxt' file
        """
        pipeline = read_config(self.config_path)

        pipeline.train_input_reader.label_map_path = labelmap_path
        pipeline.eval_input_reader[0].label_map_path = labelmap_path

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_train_tfrecord(self,
                           train_tfrecord_path):
        """
            Set Train Tfrecord Path in config file
        Args:
            train_tfrecord_path: Path of training tfrecord
        """
        pipeline = read_config(self.config_path)

        pipeline.train_input_reader.tf_record_input_reader.input_path[0] = train_tfrecord_path

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_val_tfrecord(self,
                         val_tfrecord_path):
        """
            Set Validation Tfrecord Path in config file
        Args:
            val_tfrecord_path: Path of validation or test tfrecord
        """
        pipeline = read_config(self.config_path)

        pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = val_tfrecord_path

        write_config(pipeline=pipeline, config_path=self.config_path)

    def set_weight_information(self,
                               add_weight_as_input=False,
                               input_method='input',
                               add_weight_as_output_gpo=False,
                               add_weight_as_output_gesamt=False):
        """
            Set Weight integration in config file
        Args:
            add_weight_as_input: True if to add Weight as additional Input to model
            input_method: Type of method for weight as input: For 'ssd' the strings 'input-multiply' and 'fpn-multiply'
                are valid. For 'prob_two_stage' the strings 'input-multiply', 'fpn-multiply', 'roi-multiply' and
                'concat' are valid
            add_weight_as_output_gpo: True if using weight per object as additional output in model
            add_weight_as_output_gesamt: True if using whole weight of image as additional output in model
            (only works for 'ssd')
        """
        pipeline = read_config(self.config_path)

        if self.model_type == 'ssd':
            if input_method not in ['input-multiply', 'fpn-multiply']:
                raise ValueError('String input_method must be one of: [input-multiply, fpn-multiply]')
            pipeline.model.ssd.add_weight_as_input = add_weight_as_input
            pipeline.model.ssd.input_method = input_method
            pipeline.model.ssd.add_weight_as_output_gpo = add_weight_as_output_gpo
            pipeline.model.ssd.add_weight_as_output_gesamt = add_weight_as_output_gesamt
        elif self.model_type == 'prob_two_stage':
            if input_method not in ['input-multiply', 'fpn-multiply', 'roi-multiply', 'concat']:
                raise ValueError('String input_method must be one of: [input-multiply, fpn-multiply, roi-multiply, concat]')
            pipeline.model.prob_two_stage.add_weight_as_input = add_weight_as_input
            pipeline.model.prob_two_stage.input_method = input_method
            pipeline.model.prob_two_stage.add_weight_as_output_gpo = add_weight_as_output_gpo

        write_config(pipeline=pipeline, config_path=self.config_path)




