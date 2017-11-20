def rnn_lstm(seq_length=3, num_outputs=2, image_shape=(120,160,3)):

    from keras.layers import Input, Dense
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization, Merge
    from keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
    from keras.layers.merge import concatenate
    from keras.layers import LSTM
    from keras.layers.wrappers import TimeDistributed as TD

    img_seq_shape = (seq_length,) + image_shape   
    img_in = Input(batch_shape = img_seq_shape, name='img_in')
    
    x = Sequential()
    x.add(TD(Cropping2D(cropping=((60,0), (0,0))), input_shape=img_seq_shape )) #trim 60 pixels off top
    x.add(TD(Convolution2D(24, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(64, (3,3), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(64, (3,3), strides=(1,1), activation='relu')))
    x.add(TD(Convolution2D(64, (3,3), strides=(1,1), activation='relu')))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(.1)))
      
    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_out"))
    x.add(Dropout(.1))
    x.add(Dense(50, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))
    
    x.compile(optimizer='adam', loss='mse')
    
    print(x.summary())

    return x