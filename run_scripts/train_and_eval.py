from fruitod.core.model import Model
from settings import *


if __name__ == '__main__':
    model = Model(checkpoint_path=CHECKPOINT_PATH,
                  config_path=CONFIG_PATH,
                  export_path=EXPORT_PATH)

    model.train(batch_size=BATCH_SIZE, checkpoint_every_n_epochs=10)

    model.evaluate()
    
    model.export()
