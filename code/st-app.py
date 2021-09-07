import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly as py
import numpy as np
from sqlalchemy import create_engine


def main():
    st.title('Active Learning for AI')
    st.markdown(
    """
    This is a demonstration of active learning applied to a convolutional
    neural network for image classification. Data comes from the CIFAR-10
    dataset, which is commonly used as an introduction to computer vision.
    """
    )

    st.header('What is active learning?')
    st.markdown(
    """
    Active learning is a set of techniques that allows models to be trained
    more efficiently with less data. The benefits of active learning are
    greatest in cases of supervised learning where large quantities of data
    need to be labeled by human annotators. As opposed to random sampling,
    active learning provides a level of selectivity to the data that is labeled
    and added to the training dataset. This ensures that human annotators are
    labeling the data that will have the greatest impact on the model and lead
    to more robust predictions.
    """
    )

    st.header("What's the benefit?")
    st.markdown(
    """
    In many cases, active learning can allow a model to reach target accuracy
    with less training data. The following chart shows the effect of an active
    learning strategy called uncertainty sampling applied to the CIFAR-10
    dataset.
    """
    )
    st.write(make_plotly_figure())

    st.header('Want to learn more?')
    st.markdown(
    """
    You absolutely should! Active learning comprises a really useful set of
    techniques, and I only scraped the surface with the time I had to do this
    project. Please check out the GitHub repository to learn more about my
    methodology. If this project sparked your interest, I'd recommend that you
    read the book [_Human-in-the-Loop Machine Learning_](
    https://www.manning.com/books/human-in-the-loop-machine-learning) by Robert
    Monarch. This book was the inspiration for this project, and it provides
    fascinating perspective on the intersection of humans and machines.
    """
    )


def make_plotly_figure():
    # conn_str ='sqlite:///../data/experiment_data/generated_data.db'
    conn_str ='sqlite:///../data/experiment_data/2021-09-02-experiment.db'
    engine = create_engine(conn_str)

    sql = '''
    SELECT
        a.trial_id,
        a.batch,
        a.test_acc,
        a.config_id,
        unc_pct
    FROM (results
    INNER JOIN trials ON results.trial_id = trials.trial_id) as a
    INNER JOIN configurations ON a.config_id = configurations.config_id
    '''

    df = pd.read_sql(sql, engine)
    grouped = df.groupby(['unc_pct', 'batch'])['test_acc'].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            line=dict(color="#0B88B4", width=3),
            name='Random Sampling',
            x=grouped.loc[0].index * 1000,
            y=grouped.loc[0]))
    fig.update_yaxes(range=[0, 1], tickformat=',.0%', title='Accuracy')
    fig.update_xaxes(title='Samples in Training Dataset')
    fig.update_layout(title=dict(text='Active Learning Demonstration',
        x=0.12),
    legend=dict(
        orientation='h',
        x=0,
        y=1.2
    ))

    ix_lvl_1 = grouped.index.get_level_values(0).unique()
    for x in [i/100 for i in range(101)]:
        if x in ix_lvl_1:
            s = grouped.loc[x]
        else:
            ix_2 = ix_lvl_1[np.where(ix_lvl_1 >= x)[0][0]]
            ix_1 = ix_lvl_1[np.where(ix_lvl_1 >= x)[0][0] - 1]
            s = grouped.loc[ix_1] + (((x - ix_1) / (ix_2 - ix_1)) *
                (grouped.loc[ix_2] - grouped.loc[ix_1]))
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#78CCCF", width=3),
                name="Random and Uncertainty Sampling",
                x=s.index * 1000,
                y=s))

    # Make 10th trace visible
    fig.data[1].visible = True
#
    # Create and add slider
    steps = []
    for i in range(len(fig.data) - 1):
        step = dict(
            method="update",
            args=[{"visible": [True] + [False] * len(fig.data)}],
            label=str(i) + '%'  # layout attribute
        )
        step["args"][0]["visible"][i + 1] = True  # Toggle i'th trace to visible
        steps.append(step)
#
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Uncertainty sampling: "},
        pad={"t": 50},
        steps=steps,
    )]
#
    fig.update_layout(
        sliders=sliders
)
    return fig


if __name__ == '__main__':
    main()
