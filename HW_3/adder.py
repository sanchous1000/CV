import numpy as np
def get_indices(X_shape, HF, WF, stride, pad):
    m, n_C, n_H, n_W = X_shape
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
    level1 = np.repeat(np.arange(HF), WF)
    level1 = np.tile(level1, n_C)
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    slide1 = np.tile(np.arange(WF), HF)
    slide1 = np.tile(slide1, n_C)
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)
    return i, j, d



def im2col(X, HF, WF, stride, pad):
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
