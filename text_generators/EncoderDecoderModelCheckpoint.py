from keras.callbacks import ModelCheckpoint

class EncoderDecoderModelCheckpoint(ModelCheckpoint):
	def __init__(self, decoder, encoder, filepath, monitor='val_loss', start_after_epoch=0, verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1):
		super(EncoderDecoderModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
															mode, period)
		self.decoder = decoder
		self.encoder = encoder
		self.start_after_epoch = start_after_epoch

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch=epoch, **logs)
			decoder_filepath = "%s_decoder" % self.filepath.format(epoch=epoch, **logs)
			encoder_filepath = "%s_encoder" % self.filepath.format(epoch=epoch, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if current is None:
					warnings.warn('Can save best model only with %s available, '
								  'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
						if epoch >= self.start_after_epoch:
							if self.verbose > 0:
								print('Epoch %05d: %s improved from %0.5f to %0.5f,'
									  ' saving model to %s'
									  % (epoch, self.monitor, self.best,
										 current, filepath))
							self.best = current
							if self.save_weights_only:
								self.model.save_weights(filepath, overwrite=True)
								self.decoder.save_weights(decoder_filepath, overwrite=True)
								self.encoder.save_weights(encoder_filepath, overwrite=True)
							else:
								self.model.save(filepath, overwrite=True)
								self.decoder.save(decoder_filepath, overwrite=True)
								self.encoder.save(encoder_filepath, overwrite=True)
					else:
						if self.verbose > 0:
							print('Epoch %05d: %s did not improve' %
								  (epoch, self.monitor))
			else:
				if epoch >= self.start_after_epoch:
					if self.verbose > 0:
						print('Epoch %05d: saving model to %s' % (epoch, filepath))
						print('Decoder: Epoch %05d: saving model to %s' % (epoch, decoder_filepath))
						print('Encoder: Epoch %05d: saving model to %s' % (epoch, encoder_filepath))
					if self.save_weights_only:
						self.model.save_weights(filepath, overwrite=True)
						self.decoder.save_weights(decoder_filepath, overwrite=True)
						self.encoder.save_weights(encoder_filepath, overwrite=True)
					else:
						self.model.save(filepath, overwrite=True)
						self.decoder.save(decoder_filepath, overwrite=True)
						self.encoder.save(encoder_filepath, overwrite=True)
