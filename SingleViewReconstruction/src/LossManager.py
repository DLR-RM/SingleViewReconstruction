import tensorflow as tf
import numpy as np
from src.utils import StopWatch


class LossManager(object):

	def __init__(self, tree_model, settings, data_set_loader):
		self._tree_model = tree_model
		self._settings = settings
		self._loader = data_set_loader
		self._summary_collection_set_up = []
		self._summary_collection = []
		self._initial_summary_collection = []
		self._input = self._loader.get_input()
		self._gt_output = self._loader.get_output()
		self._gt_loss_map = None
		if self._settings.use_loss_map:
			self._gt_loss_map = self._loader.get_loss_map() 
		self._rescaled_layer_before_3d = self._tree_model.rescaled_layer_before_3D
		self._last_layer = self._tree_model.last_layer
		self.trainings_loss = None
		self.visible_loss_value_map = {}

	def run_initial(self, sess, writer, step, dict=None):
		result = sess.run(self._initial_summary_collection, dict)
		for ele in result:
			writer.add_summary(ele, global_step=step)

	def run_summary(self, sess, writer, step, dict=None):
		collection = [self.trainings_loss]
		collection.extend(self._summary_collection)
		result = sess.run(collection, dict)
		for ele in result[1:]:
			writer.add_summary(ele, global_step=step)
		return result[0]

	def get_initial_summaries(self):
		return self._initial_summary_collection

	def get_summaries(self):
		return self._summary_collection

	def _add_inner_tree_loss(self):
		inner_tree_results = self._tree_model.get_intermediate_inner_tree_results()
		inner_tree_loss = None
		# move the channel to the last place (not channel first)
		cmp_voxel = tf.transpose(self._gt_output, (0, 2, 3, 4, 1))
		for height, layer in enumerate(inner_tree_results[:-1]):
			inner_size = self._settings.result_size // len(layer)
			print("Use loss in height: " + str(height) + ", size: " + str(inner_size) + ", layers: " + str(len(layer)))
			new_res = None
			for nr, node in enumerate(layer):
				start = nr * inner_size
				end = (nr + 1) * inner_size
				# old order: batch size, feature maps, x, y
				expanded_node = tf.expand_dims(node, 4) # batch size, feature maps, x, y, 1
				transpose_node = tf.transpose(expanded_node, (0, 2, 3, 4, 1))  # new order: batch size, x, y, 1 is z, feature maps
				# print(len(layer), height, inner_size, start, end, transpose_node.shape)

				cmp_for_current_node = tf.math.reduce_mean(cmp_voxel[:, :, :, start:end, :], 3, keep_dims=True)
				# print(cmp_for_current_node.shape, transpose_node.shape)
				if self._settings.use_loss_map:
					result = tf.losses.mean_squared_error(cmp_for_current_node, transpose_node, reduction=tf.losses.Reduction.NONE)
					# reduce mean over encoded channels
					result = tf.math.reduce_mean(result, axis=4, keep_dims=True)
					result = result[:,:,:,:,0] # use the averaged or only example to apply the loss map
					avg_for_loss_map = tf.math.reduce_mean(self._gt_loss_map[:, :, :, start:end], 3, keep_dims=True)
					result *= avg_for_loss_map
					result = tf.math.reduce_mean(result)
				else:
					result = tf.losses.mean_squared_error(cmp_for_current_node, transpose_node)
				if new_res is None:
					new_res = result
				else:
					new_res += result
			if inner_tree_loss is None:
				inner_tree_loss = self._settings.loss_height_weight[height] * new_res
			else:
				inner_tree_loss += self._settings.loss_height_weight[height] * new_res
		return inner_tree_loss

	def generate_loss(self):
		sw = StopWatch()
		self._add_histo_for_initial_input_and_output()
		self._add_histo_for_tree_output()
		self.trainings_loss = self._set_up_current_loss(self._last_layer, added_name="final")
		if self._settings.before_3D_loss_weight > 1e-7:
			layer_before_3D_loss = self._set_up_current_loss(self._rescaled_layer_before_3d, added_name="before_3D")
			self.trainings_loss = self.trainings_loss + self._settings.before_3D_loss_weight * layer_before_3D_loss
		inner_tree_loss = self._add_inner_tree_loss()
		self.trainings_loss = self.trainings_loss + self._settings.inner_tree_loss_weight * inner_tree_loss
		with tf.name_scope('training_loss'):
			self._add_to_summary_scalar('final_training_loss', self.trainings_loss)
		self._finish_summary_set_up()
		print("Took " + str(sw.elapsed_time) + " to generate loss")

	def _set_up_current_loss(self, current_last_layer, added_name):
		diff = self._gt_output - current_last_layer
		diff_square = tf.square(diff)

		self._add_to_summary_histo('diff', diff, add_clip_values=True, own_scope=added_name)
		self._add_to_summary_scalar('abs_diff', tf.reduce_mean(tf.abs(diff)), own_scope=added_name)
		self._add_to_summary_histo('diff_square', diff_square, add_clip_values=True, own_scope=added_name)
		for dim in range(self._settings.amount_of_output_channels):
			self._add_to_summary_scalar('abs_diff_in_' + str(dim), tf.reduce_mean(tf.abs(diff[:, dim])), own_scope='loss_calc_additional')

		loss_not_reduced = tf.losses.mean_squared_error(self._gt_output, current_last_layer, reduction=tf.losses.Reduction.NONE)
		self._add_to_summary_histo('squared_mean_error_not_reduced', loss_not_reduced, own_scope=added_name)
		squared_mean_loss = tf.math.reduce_mean(loss_not_reduced)
		self._add_to_summary_scalar('squared_mean_loss_whole', squared_mean_loss, own_scope=added_name)
		unreduced_loss_for_bootstrap = loss_not_reduced
		size_for_bootstrap =  int((self._settings.result_size**3) * self._settings.amount_of_output_channels)
		current_final_loss = squared_mean_loss

		if self._settings.use_loss_map:
			# squish the loss in the channel
			diff_squished = tf.math.reduce_mean(loss_not_reduced, axis=1)
			# multiply it with the loss map
			diff_squished *= self._gt_loss_map
			self.visible_loss_value_map[added_name] = diff_squished
			self._add_to_summary_histo('squared_mean_loss_visible_not_reduced', diff_squished, own_scope=added_name)

			diff_squished_mean = tf.math.reduce_mean(diff_squished)
			self._add_to_summary_scalar('squared_mean_loss_visible', diff_squished_mean, own_scope=added_name)
			unreduced_loss_for_bootstrap = diff_squished
			size_for_bootstrap =  int(self._settings.result_size**3)
			current_final_loss = diff_squished_mean

		if self._settings.bootstrap_ratio > 1e-10:
			loss_flattened = tf.layers.Flatten()(unreduced_loss_for_bootstrap)
			used_k_loss_values = int(float(size_for_bootstrap) * self._settings.bootstrap_ratio)
			print("Use only: " + str(used_k_loss_values) + ", of " + str(size_for_bootstrap))
			top_losses,_ = tf.nn.top_k(loss_flattened, k=used_k_loss_values, sorted=False)
			self._add_to_summary_histo('squared_mean_error_top_'+str(used_k_loss_values)+'_not_reduced', loss_not_reduced, own_scope=added_name)
			top_k_loss = tf.reduce_mean(top_losses)
			self._add_to_summary_scalar('top_k_loss', top_k_loss, own_scope=added_name)
			current_final_loss = top_k_loss

		if self._settings.regularizer_scale > 1e-13:
			reg_loss = tf.losses.get_regularization_loss()
			current_final_loss = current_final_loss + reg_loss
			self._add_to_summary_scalar('reg_los', reg_loss, own_scope=added_name)
		self._add_to_summary_scalar('trainings_loss', current_final_loss, own_scope=added_name)
		return current_final_loss

	def _add_histo_for_initial_input_and_output(self):
		# set up initial summary collection for input and output
		with tf.name_scope('input'):
			self._add_initial_summary_histo('input_color_histo', self._input[:, :3, :, :])
			self._add_initial_summary_histo('input_normal_histo', self._input[:, 3:, :, :])
		with tf.name_scope('output'):
			self._add_initial_summary_histo('output_histo', self._gt_output, add_clip_values=True)
			if self._gt_loss_map is not None:
				self._add_initial_summary_histo('loss_map_histo', self._gt_loss_map)
		with tf.name_scope('output_dim'):
			for dim in range(self._settings.amount_of_output_channels):
				self._add_initial_summary_histo('output_for_dim_' + str(dim) + '_histo', self._gt_output[:, dim, :, :, :], add_clip_values=True)

	def _add_histo_for_tree_output(self):
		# current tree output
		with tf.name_scope('last_layer'):
			self._add_to_summary_histo('last_layer_histo', self._last_layer, add_clip_values=True)
		with tf.name_scope('last_layer_dim'):
			for dim in range(self._settings.amount_of_output_channels):
				self._add_to_summary_histo('last_layer_for_dim_' + str(dim) + '_histo', self._last_layer[:, dim, :, :, :], add_clip_values=True)
		if self._settings.before_3D_loss_weight > 1e-7:
			with tf.name_scope('rescaled_layer_before_3d'):
				self._add_to_summary_histo('rescaled_layer_before_3d_histo', self._rescaled_layer_before_3d, add_clip_values=True)
			with tf.name_scope('rescaled_layer_before_3d_dim'):
				for dim in range(self._settings.amount_of_output_channels):
					self._add_to_summary_histo('rescaled_layer_before_3d_for_dim_' + str(dim) + '_histo',
													self._rescaled_layer_before_3d[:, dim, :, :, :], add_clip_values=True)

	def _add_initial_summary_histo(self, name, used_values, add_clip_values=False):
		histo = tf.summary.histogram(name, used_values)
		self._initial_summary_collection.append(histo)
		if add_clip_values:
			clipped_values = tf.clip_by_value(used_values, -0.05, 0.17)
			histo_clipped = tf.summary.histogram(name + "_clipped", clipped_values)
			self._initial_summary_collection.append(histo_clipped)

	def _add_to_summary_histo(self, name, used_values, add_clip_values=False, own_scope=None):
		self._summary_collection_set_up.append((name, used_values, add_clip_values, True, own_scope))

	def _finish_summary_set_up(self):
		scope_list = {}
		for name, used_values, add_clip_values, is_histo, scope in self._summary_collection_set_up:
			if scope in scope_list.keys():
				scope_list[scope].append((name, used_values, add_clip_values, is_histo))
			else:
				scope_list[scope] = [(name, used_values, add_clip_values, is_histo)]
		for scope, ele_list in scope_list.items():
			if scope is None:
				scope = "rest"
			with tf.name_scope(scope):
				for name, used_values, add_clip_values, is_histo in ele_list:
					if is_histo:
						histo = tf.summary.histogram(name, used_values)
						self._summary_collection.append(histo)
						if add_clip_values:
							clipped_values = tf.clip_by_value(used_values, -0.05, 0.17)
							histo_clipped = tf.summary.histogram(name + "_clipped", clipped_values)
							self._summary_collection.append(histo_clipped)
					else:
						scalar = tf.summary.scalar(name, used_values)
						self._summary_collection.append(scalar)


	def _add_to_summary_scalar(self, name, used_value, own_scope=None):
		self._summary_collection_set_up.append((name, used_value, False, False, own_scope))





