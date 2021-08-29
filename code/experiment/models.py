from sqlalchemy import Table, Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Configuration(Base):
    __tablename__ = 'configurations'
    config_id = Column(Integer, primary_key=True)
    unc_pct = Column(Float)
    div_pct = Column(Float)
    batch_size = Column(Integer)
    num_trials = Column(Integer)
    trials = relationship('Trial', backref=backref('config'))

class Trial(Base):
    __tablename__ = 'trials'
    trial_id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey('configurations.config_id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    results = relationship('Result')


class Result(Base):
    __tablename__ = 'results'
    trial_id = Column(Integer, ForeignKey('trials.trial_id'), primary_key=True)
    batch = Column(Integer, primary_key=True)
    train_acc = Column(Float)
    val_acc = Column(Float)
    test_acc = Column(Float)
    epochs = Column(Integer)
