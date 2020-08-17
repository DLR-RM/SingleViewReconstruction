
import os
from src.configreader import ConfigReader
from src.dataset import Dataset
from src.autoencoder import Autoencoder

if __name__ == "__main__":

    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    config_obj = ConfigReader(config_path)

    dataset = Dataset(config_obj)
    x_train = dataset.load_train_data()
    x_val = dataset.load_val_data()
    x_eval = dataset.load_eval_data()
    model = Autoencoder(config_obj, dataset)

    model.set_iterators(x_train, x_val, eval_from_input_iterator=x_eval)

    for i in range(12000):
        # the evaluation is quite time intensive, during it off increase the speed
        do_evaluation = i % 500 == 0 and i > 0
        stats = model.train(do_evaluation)
        print("{}: {}".format(i, stats["loss"]))
        if "val_loss" in stats:
            print("Val loss: {}".format(stats["val_loss"]))
            print("IO: {}, l1: {}".format(stats['iou'], stats["eval_l1"]))
        if i % 1000 and i > 0:
            model.save(config_obj.data.get_string("model_save_path"))

    model.save(config_obj.data.get_string("model_save_path"))
