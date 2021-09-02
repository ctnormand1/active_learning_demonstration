import pandas as pd
import numpy as np
import os
import queue
from datetime import datetime
from tqdm.notebook import tqdm
from multiprocess import Process, Queue, Pipe
from multiprocess.connection import wait
from experiment.models import Base, Configuration, Trial, Result
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tensorflow.keras.callbacks import EarlyStopping


class Experiment():
    def __init__(self, conn_str, model_func, data):
        self.conn_str = conn_str
        self.engine = create_engine(conn_str)
        self.model_func = model_func
        self.data = data
        self.data=data

    def configure(self, unc_pct_range, div_pct_range, batch_size, num_trials,
        overwrite_db=False):
        if overwrite_db:
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        with Session(self.engine, expire_on_commit=False) as session:
            for unc_pct in unc_pct_range:
                for div_pct in div_pct_range:
                    configuration = Configuration(
                        unc_pct=unc_pct,
                        div_pct=div_pct,
                        batch_size=batch_size,
                        num_trials=num_trials)
                    session.add(configuration)
                    session.commit()
                    for i in range(1, num_trials + 1):
                        trial = Trial(config_id = configuration.config_id)
                        session.add(trial)
            session.commit()

    def run(self, n_jobs=1):
        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        print(f'Starting experiment at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        work_queue = Queue()

        with Session(self.engine) as session:
            trial_ids = session.query(Trial.trial_id).all()
        for trial_id in trial_ids:
            work_queue.put(trial_id)
        print(f'There will be a total of {len(trial_ids)} trials spread across {n_jobs} jobs.')

        readers = []
        # start subprocesses
        for i in range(n_jobs):
            r, w = Pipe()
            readers.append(r)
            p = Process(target=trial_runner, args=(self.data, self.model_func,
                self.conn_str, work_queue, w, i))
            p.start()
            w.close()  # close this connection to the "write" end of the pipe

        pbars = [tqdm(position=i, ncols=600,
            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
            desc='Waiting to start', total=1) for i in range(n_jobs)]

        while readers:
            for reader in wait(readers):
                try:
                    msg = reader.recv()
                except EOFError:
                    readers.remove(reader)
                else:
                    pbar = pbars[msg['pbar_num']]
                    if msg['desc'] != pbar.desc:
                        pbar.reset(msg['n_total'])
                        pbar.set_description(msg['desc'])
                    pbar.update(msg['n_complete'] - pbar.n)
                    pbar.refresh()



def trial_runner(data, model_func, conn_str, work_queue, write_pipe, pbar_num):
    # create engine and start a database session
    engine = create_engine(conn_str)
    with Session(engine, expire_on_commit=False) as session:
        # run this loop as long as there are more trials in the queue
        while True:
            try:
                trial_id = work_queue.get(block=False)
            except queue.Empty:
                break
            trial = session.query(Trial).get(trial_id)
            trial.start_time = datetime.utcnow()
            session.commit()
            run_trial(trial, data, session, write_pipe, model_func, pbar_num)
            trial.end_time = datetime.utcnow()
            session.commit()
    write_pipe.close()

def run_trial(trial, data, session, write_pipe, model_func, pbar_num):
    # unpack data into more intuitive variables
    X_train, y_train = data[0][0], data[0][1]
    X_val, y_val = data[1][0], data[1][1]
    X_test, y_test = data[2][0], data[2][1]

    # easier access to the number of samples in each data set
    n_train, n_val, n_test = y_train.shape[0], y_val.shape[0], y_test.shape[0]

    # keep track of what's been annotated, other setup...
    is_annotated = pd.Series([False] * n_train)
    early_stop_callback = EarlyStopping(patience=5)
    n_annotation_batches = int(np.ceil(n_train / trial.config.batch_size))

    # initial pbar update
    write_pipe.send({
        'pbar_num': pbar_num,
        'desc': f'Trial {trial.trial_id}',
        'n_complete': 0,
        'n_total': n_annotation_batches
    })

    # run annotation batches
    for i in range(1, 1 + n_annotation_batches):
        model = model_func()
        samples_ix = get_samples_ix(X_train, y_train, trial, model,
            is_annotated)
        # is_annotated.loc[samples_ix] = True
        h = model.fit(X_train[is_annotated], y_train[is_annotated],
            validation_data=(X_val, y_val), batch_size=64, epochs=1, verbose=0,
            callbacks=[early_stop_callback])
        n_epochs = len(h.history['loss'])
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
        result = Result(trial_id=trial.trial_id, batch=i, epochs=n_epochs,
            test_acc=test_acc, train_acc=h.history['accuracy'][-1],
            val_acc=h.history['val_accuracy'][-1])
        session.add(result)
        session.commit()
        write_pipe.send({
            'pbar_num': pbar_num,
            'desc': f'Trial {trial.trial_id}',
            'n_complete': i,
            'n_total': n_annotation_batches
        })

def get_samples_ix(X_train, y_train, trial, model, is_annotated):
    # for the first annotation batch, return all random samples
    if is_annotated.sum() == 0:
        samples_ix = is_annotated.sample(trial.config.batch_size).index
        is_annotated.loc[samples_ix] = True
        return samples_ix

    # for subsequent annotation batches, get a portion of samples from
    # uncertainty sampling
    preds = model.predict(X_train)
    n_unc = int(trial.config.batch_size * trial.config.unc_pct)
    unc_scores = get_uncertainty_scores(preds, margin_of_confidence_score)
    unc_samples_ix = unc_scores[~is_annotated].sort_values(
        ascending=False).iloc[:n_unc].index.tolist()
    is_annotated.loc[unc_samples_ix] = True

    # and do random sampling for the rest
    n_rand = trial.config.batch_size - n_unc
    rand_samples_ix = is_annotated[~is_annotated].sample(n_rand).index.tolist()
    is_annotated.loc[rand_samples_ix] = True

    return unc_samples_ix + rand_samples_ix



def get_uncertainty_scores(preds, score_func):
    return pd.Series([score_func(pred) for pred in preds])

def margin_of_confidence_score(prob_dist):
    prob_dist[::-1].sort()  # sort descending
    difference = prob_dist[0] - prob_dist[1]
    return 1 - difference
