import tensorflow as tf
from tensorflow import keras
import h5py

text_input = keras.Input(shape=(None,768,), dtype='float32', name='text')

# keras.layers.Masking(mask_value=0.0)
l_mask = keras.layers.Masking(mask_value=-99.)(text_input) 

# Which we encoded in a single vector via a LSTM
encoded_text = keras.layers.LSTM(100,)(l_mask)
out_dense = keras.layers.Dense(30, activation='relu')(encoded_text)
# And we add a softmax classifier on top
out = keras.layers.Dense(2, activation='softmax')(out_dense)
# At model instantiation, we specify the input and the output:
model = keras.Model(text_input, out)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

call_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=3, verbose=2,
                                mode='auto', min_delta=0.01, cooldown=0, min_lr=0)

batches_per_epoch = batches[0][1]

batches_per_epoch_val= batches[1][1]

model.fit(train_data, steps_per_epoch=batches_per_epoch, epochs=10,
                    validation_data=val_data, validation_steps=batches_per_epoch_val, callbacks =[call_reduce] )

save_path = "./trained_models/classification_models_" + model_path + "/LSTM_model/model.h5"

model.save(save_path)