# -*- coding:utf-8 -*
"""
@version: ??
@Author by Ggao
@Mail: 649386435@qq.com 
@File: mm.py
@time: 2017-04-11 下午4:03
"""
import tensorflow as tf
from game import Game
import numpy as np
import random
import cv2
import logging
import sys
from collections import deque
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
'''
Q(s_t, a_t) = R(s_t, a_t) + r*max(Q(s_t+1, a_t+1))
st 表示t时刻的状态 at表示t时刻的行为 r 为学习参数 0-1
神经网络代表转移参数 完成状态矩阵中的状态转移 向Q增大的方向移动
此处 reward 取值为 -1， 0， 1
学习率 取值为 0.99
'''

ACTIONS = 3  # 可用操作数
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 500.  # 在训练之前观测
EXPLORE = 2000000.  # 一共学习多少帧
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 1.0  # starting value of epsilon
REPLAY_MEMORY = 500  # 记录多少帧最为训练数据
BATCH = 32  # 每次训练batch大小
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, shape, stride):
    w = weight_variable(shape=shape)
    b = bias_variable(shape=[shape[-1]])
    c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.relu(c + b)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # network weights
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    nn_input = tf.placeholder("float", [None, 80, 80, 4])
    h_conv1 = conv2d(nn_input, [8, 8, 4, 32], 4)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = conv2d(h_pool1, [4, 4, 32, 64], 2)
    h_conv3 = conv2d(h_conv2, [3, 3, 64, 64], 1)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    nn_output = tf.matmul(h_fc1, W_fc2) + b_fc2
    return nn_input, nn_output


def trainNetwork(nn_input, nn_output_, sess):
    # define the cost function
    train_action = tf.placeholder("float", [None, ACTIONS])  # 移动输入
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(nn_output_, train_action), reduction_indices=1)  # 在移动方向上的权值
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = Game()
    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    img, score, crash = game_state.process(do_nothing)  # image, score [ 0, -1, 1] , crash [True False]
    img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    game_img_list = np.stack((img, img, img, img), axis=2)

    time_step = 0
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("gamenn")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        time_step += int(checkpoint.model_checkpoint_path.split('-')[-1])
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE * time_step

    while True:
        readout_t = nn_output_.eval(feed_dict={nn_input: [game_img_list]})[0]  # aciton
        action = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon:
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            action[action_index] = 1

        if epsilon > FINAL_EPSILON and time_step > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        img, score, crash = game_state.process(action)
        img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (80, 80, 1))
        game_img_list1 = np.append(img, game_img_list[:, :, :3], axis=2)

        D.append((game_img_list, action, score, game_img_list1, crash))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if time_step > OBSERVE and len(D) == REPLAY_MEMORY and time_step % REPLAY_MEMORY == 0:
            minibatch = D
            batch_game_img_list = [d[0] for d in minibatch]  # image_sequence
            batch_action = [d[1] for d in minibatch]  # game actions
            batch_score = [d[2] for d in minibatch]  # game score
            batch_game_img_list1 = [d[3] for d in minibatch]  # image_sequence2

            y_batch = []
            batch_nn_output = nn_output_.eval(feed_dict={nn_input: batch_game_img_list1})
            for i in range(0, len(minibatch)):
                crash = minibatch[i][4]
                # if crash, only equals reward
                if crash:  # crash
                    y_batch.append(batch_score[i])
                else:
                    # 取值每次加 n*0.99+1 和 nn_out 两个值趋近 目标期望为 n = n*0.99+1, n = 100
                    # 每次移动都向着 Q 增大的方向移动
                    y_batch.append(batch_score[i] + GAMMA * np.max(batch_nn_output[i]))
            train_step.run(feed_dict={
                y: y_batch,
                train_action: batch_action,
                nn_input: batch_game_img_list}
            )

        game_img_list = game_img_list1
        time_step += 1

        if time_step % 10000 == 0:
            saver.save(sess, 'gamenn/' + 'model-dqn', global_step=time_step)

        state = ""
        if time_step <= OBSERVE:
            state += "observe"
        elif OBSERVE < time_step <= OBSERVE + EXPLORE:
            state += "explore"
        else:
            state += "train"
        out_put_str = 'TIMESTE {0}; STATE {1}; EPSILON {2}; ACTION {3}; REWARD {4}; Q_MAX {5}'.format(
            time_step, state, round(epsilon, 4), action_index, str(score).zfill(3), np.max(readout_t)
        )
        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write(out_put_str)


def playGame():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
