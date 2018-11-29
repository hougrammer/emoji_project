import numpy as np
import scipy
from importlib import reload
from IPython.core.display import display, HTML
import tensorflow as tf

def cosine_similarity(v1, v2):
    '''
    args:
        v1: numeric iterable
        v2: numeric iterable
    returns:
        (float) cosine similarity of the two vectors
        
    The cosine distance from scipy is 1 - uv/(uu + vv), so here, 1 cancels out and you get the similarity.
    '''
    return 1 - scipy.spatial.distance.cosine(v1, v2)

def display_emoji(unicode, size=2):
    '''
    renders emoji in html
    '''
    display(HTML('<font size="+{}">{}</font>'.format(size, chr(int(unicode[2:], 16)))))
    
def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)
