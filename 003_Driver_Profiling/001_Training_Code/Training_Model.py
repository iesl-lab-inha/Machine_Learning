from tensorflow import keras

def Training_Model(model,epochs,X_train,y_train,X_val,y_val):

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'Adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, verbose=2)
    return model,history
