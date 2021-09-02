import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment
from experiment.datasets import load_cifar10
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from experiment.models import Configuration
from sqlalchemy.sql import func

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential



def main():



    data = load_cifar10(validation_size=1000)
    data = ((data[0][0][:1000], data[0][1][:1000]),
            (data[1][0][:1000], data[1][1][:1000]),
            (data[2][0][:1000], data[2][1][:1000]))
    experiment = Experiment('sqlite:///../data/experiment_data/mp_test_db.db', build_model, data)
    experiment.configure(
        unc_pct_range=[0, 0.2],
        div_pct_range=[0],
        batch_size=100,
        num_trials=1,
        overwrite_db=True)
    experiment.run(n_jobs=2)




def build_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()
