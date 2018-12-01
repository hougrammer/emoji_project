import numpy as np
import pandas as pd
import tensorflow as tf

#folder = '../data/Are_emojis_predictable_2017/'

def _get_pd_df(folder,tag):
    return pd.read_table(folder+tag,header =  None, names = ['input','target'])

def get_tfdataset(folder, tag):
    d = _get_pd_df(folder,tag)
    return _df_to_tfds(d,list(d.columns))

def _df_to_tfds(dataframe,names):
    return tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(dataframe[names[0]].values,tf.string),tf.cast(dataframe[names[1]].values,tf.string)
        )
    )