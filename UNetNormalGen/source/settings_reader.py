import yaml
import os

class SettingsReader(object):

	def __init__(self, settings_file_path, special_folder_path=None):
		with open(settings_file_path, 'r') as stream:
			settings = yaml.load(stream)

		self.img_size = settings["image"]["size"]

		self.augment = self.get_settings_bool(settings["augment"])

		self.batch_size = settings["batch_size"]
		self.shuffle_size = settings["shuffle_size"]
		self.validation_ratio = settings["validation_ratio"]
		self.learning_rate = settings["learning_rate"]

		data = settings["data"]
		self.data_path = os.path.abspath(data['path'])

		self.max_dataset_size = int(settings['max_dataset_size'])

	def get_settings_bool(self, line):
		if isinstance(line, bool):
			return line
		else:
			return line == "True" or line == "true" or line == "1"

