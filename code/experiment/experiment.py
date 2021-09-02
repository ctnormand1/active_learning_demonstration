import pandas as pd
from datetime import datetime
from .models import Base, Configuration, Trial, Result
from sqlalchemy.orm  import Session
from tqdm import tqdm
from IPython.display import clear_output
from tqdm.keras import TqdmCallback

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class Experiment():
    def __init__(self, engine, model_func, data):
        self.engine = engine
        self.model_func = model_func

        self.X_train = data[0][0]
        self.y_train = data[0][1]
        self.X_val = data[1][0]
        self.y_val = data[1][1]
        self.X_test = data[2][0]
        self.y_test = data [2][1]

        self.y_train_oh = to_categorical(self.y_train)
        self.y_val_oh = to_categorical(self.y_val)
        self.y_test_oh = to_categorical(self.y_test)

    def initialize_db(self, overwrite=False):
        if overwrite:
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)


    def configure(self, unc_pct_range, div_pct_range, batch_size, num_trials,
        overwrite=False):

        self.initialize_db(overwrite=overwrite)

        with Session(self.engine) as session:
            for unc_pct in unc_pct_range:
                for div_pct in div_pct_range:
                    configuration = Configuration(
                        unc_pct=unc_pct,
                        div_pct=div_pct,
                        batch_size=batch_size,
                        num_trials=num_trials)
                    session.add(configuration)
            session.commit()

            self.num_trials = len(unc_pct_range) * len(div_pct_range)

    def run(self):
        with Session(self.engine) as session:
            configs = session.query(Configuration).all()
        for config in configs:
            self.run_trials(config)

    def run_trials(self, config):
        for i in range(1, config.num_trials + 1):
            with Session(self.engine, expire_on_commit=False) as session:
                trial = Trial(
                    config_id=config.config_id,
                    start_time=datetime.utcnow())
                session.add(trial)
                session.commit()

                self.run_trial(trial)

                trial.end_time = datetime.utcnow()
                session.commit()
            clear_output(wait=True)


    def run_trial(self, trial):

        # where we'll keep track of what's been annotated
        is_annotated = pd.Series([False] * self.X_train.shape[0])

        # build the model
        model = self.model_func()

        # train the model
        early_stop_callback = EarlyStopping(patience=3)
        num_batches = self.X_train.shape[0] // trial.config.batch_size
        for i in range(1, 1 + num_batches):
            self.__print_status(trial, i, num_batches)
            print('Gathering samples. This could take a minute...')
            samples_ix = self.__get_samples(i, trial.config, model, is_annotated)
            clear_output(wait=True)
            self.__print_status(trial, i, num_batches)
            print('Gathering samples. This could take a minute... Done!')

            h = model.fit(self.X_train[samples_ix], self.y_train_oh[samples_ix],
                validation_data=(self.X_val, self.y_val_oh), batch_size=32,
                epochs=100, verbose=1, callbacks=[early_stop_callback])
            with Session(self.engine) as session:
                print('Evaluating on test data...')
                result = Result(
                    trial_id=trial.trial_id,
                    batch=i,
                    train_acc=h.history['accuracy'][-1],
                    val_acc=h.history['val_accuracy'][-1],
                    test_acc=model.evaluate(self.X_test, self.y_test_oh, verbose=1)[1],
                    epochs=len(h.history['loss']))
                session.add(result)
                session.commit()
                clear_output(wait=True)


    def __get_samples(self, batch_number, config, model, is_annotated):
        # for batch number one, just get random samples
        if batch_number == 1:
            samples_ix = is_annotated.sample(config.batch_size).index
            is_annotated.loc[samples_ix] = True
            return samples_ix

        # uncertainty sampling
        preds = model.predict(self.X_train)
        num_unc = int(config.batch_size * config.unc_pct)
        unc_scores = self.uncertainty_scores(preds)
        unc_ix = unc_scores[~is_annotated].sort_values(ascending=False).iloc[:num_unc].index.tolist()
        is_annotated.loc[unc_ix] = True

        # random sampling
        num_rand = config.batch_size - num_unc
        rand_ix = is_annotated[~is_annotated].sample(num_rand).index.tolist()
        is_annotated.loc[rand_ix] = True

        return unc_ix + rand_ix


    def uncertainty_scores(self, preds):
        scores = []
        for prob_dist in preds:
            score = self.margin_of_confidence_score(prob_dist)
            scores.append(score)
        return pd.Series(scores)

    def margin_of_confidence_score(self, prob_dist):
        prob_dist[::-1].sort()
        difference = prob_dist[0] - prob_dist[1]
        return 1 - difference

    def __print_status(self, trial, batch_num, num_batches):
        print(f'Trial {trial.trial_id}/{self.num_trials} -- Start time: {trial.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 45)
        print(f'Batch {batch_num}/{num_batches}')
