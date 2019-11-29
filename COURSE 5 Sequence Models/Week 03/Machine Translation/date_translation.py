from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

import random
from tqdm import tqdm

from Date_Translation import nmt_utils
from Date_Translation.nmt_utils import *
import matplotlib.pyplot as plt
from faker import Faker     # generate fake data
from babel.dates import format_date
"""
codes in nmt_utils is really wonderful
"""

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

# dataset is a list length: 1w;  all relevant data is generated via fake package.

Tx = 30  # max length of input
Ty = 10  # output length
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

"""
index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
"""

# 将共享层定义为全局变量
repeator = RepeatVector(Tx)  # repeat for 30 times
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # 在这个 notebook 我们正在使用自定义的 softmax(axis = 1)
dotor = Dot(axes=1)


# GRADED FUNCTION: one_step_attention

def one_step_attention(a, s_prev):
    """
    执行一步 attention: 输出一个上下文向量，输出作为注意力权重的点积计算的上下文向量
    "alphas"  Bi-LSTM的 隐藏状态 "a"

    参数：
    a --  Bi-LSTM的输出隐藏状态 numpy-array 维度 (m, Tx, 2*n_a)
    s_prev -- (post-attention) LSTM的前一个隐藏状态, numpy-array 维度(m, n_s)

    返回：
    context -- 上下文向量, 下一个(post-attetion) LSTM 单元的输入
    """

    # 使用 repeator 重复 s_prev 维度 (m, Tx, n_s) 这样你就可以将它与所有隐藏状态"a" 连接起来。 (≈ 1 line)
    s_prev = repeator(s_prev)
    # 使用 concatenator 在最后一个轴上连接 a 和 s_prev (≈ 1 line)
    concat = concatenator([a, s_prev])
    # 使用 densor1 传入参数 concat, 通过一个小的全连接神经网络来计算“中间能量”变量 e。(≈1 lines)
    e = densor1(concat)
    # 使用 densor2 传入参数 e , 通过一个小的全连接神经网络来计算“能量”变量 energies。(≈1 lines)
    energies = densor2(e)
    # 使用 activator 传入参数 "energies" 计算注意力权重 "alphas" (≈ 1 line)
    alphas = activator(energies)
    # 使用 dotor 传入参数 "alphas" 和 "a" 计算下一个（(post-attention) LSTM 单元的上下文向量 (≈ 1 line)
    context = dotor([alphas, a])

    return context


n_a = 32
n_s = 64
# return_state : Boolean. Whether to return the last state in addition to the output.
post_activation_LSTM_cell = LSTM(n_s, return_state=True)    # Positive integer, dimensionality of the output space.
output_layer = Dense(len(machine_vocab), activation=softmax)    # Positive integer, dimensionality of the output space.


# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    参数:
    Tx -- 输入序列的长度
    Ty -- 输出序列的长度
    n_a -- Bi-LSTM的隐藏状态大小
    n_s -- post-attention LSTM的隐藏状态大小
    human_vocab_size -- python字典 "human_vocab" 的大小
    machine_vocab_size -- python字典 "machine_vocab" 的大小

    返回：
    model -- Keras 模型实例
    """

    # 定义模型的输入，维度 (Tx,)
    # 定义 s0 和 c0, 初始化解码器 LSTM 的隐藏状态，维度 (n_s,)
    X = Input(shape=(Tx, human_vocab_size))     # X.
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # 初始化一个空的输出列表
    outputs = []

    # 第一步：定义 pre-attention Bi-LSTM。 记得使用 return_sequences=True. (≈ 1 line)
    # return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
    a = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(m, Tx, n_a * 2))(X)

    # 第二步：迭代 Ty 步
    for t in range(Ty):
        # 第二步.A: 执行一步注意机制，得到在 t 步的上下文向量 (≈ 1 line)
        context = one_step_attention(a, s)

        # 第二步.B: 使用 post-attention LSTM 单元得到新的 "context"
        # 别忘了使用： initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # 第二步.C: 使用全连接层处理post-attention LSTM 的隐藏状态输出 (≈ 1 line)
        out = output_layer(s)

        # 第二步.D: 追加 "out" 到 "outputs" 列表 (≈ 1 line)
        outputs.append(out)

    # 第三步：创建模型实例，获取三个输入并返回输出列表。 (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model.summary()
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

#  这里为什么Yoh 要更换维度不懂
outputs = list(Yoh.swapaxes(0, 1))     # swapaxes 接受一对轴编号, 转变 第一个dimension 和 第二个dimension。Yoh.swapaxes(0, 1)

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

model.save_weights('E:\\WorkSpace\\pycharm\\DL_Course5_W3\\Date_Translation\\models\\my_model.h5', overwrite=True)
# model.save_weights('./Date_Translation/models/blstm_model.h5', overwrite=True)
# model.save('E:\\WorkSpace\\pycharm\\DL_Course5_W3\\Date_Translation\\models\\my_model.h5')  # save the model
# del model

#  model = load_model('my_model.h5')
