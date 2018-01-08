def Model_Definition(model_config)
# Use the packages of Keras to define the RNN-LSTM architechture for our DVQA model.
	
	from keras.models import Sequential
	from keras.layers.core import Activation
	from keras.layers.core import Dense
	from keras.layers.core import Dropout
	from keras.layers.core import Layer
	from keras.layers.core import Merge
	from keras.layers.core import TimeDistributedMerge
	from keras.layers.embeddings import Embedding
	from keras.layers.recurrent import GRU
	from keras.layers.recurrent import LSTM
	from keras import optimizers

	from kraino.core.model_zoo import AbstractSequentialModel
	from kraino.core.model_zoo import AbstractSingleAnswer
	from kraino.core.model_zoo import AbstractSequentialMultiplewordAnswer
	from kraino.core.model_zoo import Config
	from kraino.core.keras_extensions import DropMask
	from kraino.core.keras_extensions import LambdaWithMask
	from kraino.core.keras_extensions import time_distributed_masked_ave

	class VisionLanguageLSTM(AbstractSequentialModel, AbstractSingleAnswer):
		def create(self):
			language_model = Sequential()
			language_model.add(Embedding(
			self._config.input_dim, 
			self._config.textual_embedding_dim, 
			mask_zero=True))
			language_model.add(LSTM(self._config.hidden_state_dim, 
			return_sequences=False))

			visual_model = Sequential()
			if self._config.visual_embedding_dim > 0:
				visual_model.add(Dense(
				self._config.visual_embedding_dim,
				input_shape=(self._config.visual_dim,)))
			else:
				visual_model.add(Layer(input_shape=(self._config.visual_dim,)))
				self.add(Merge([language_model, visual_model], mode=self._config.multimodal_merge_mode))
				self.add(Dropout(0.5))
				self.add(Dense(self._config.output_dim))
				self.add(Activation('softmax'))
				
