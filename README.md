# DQN
```
Q(s_t, a_t) = R(s_t, a_t) + r*max(Q(s_t+1, a_t+1))
st 表示t时刻的状态 at表示t时刻的行为 r 为学习参数 0-1
神经网络代表转移参数 完成状态矩阵中的状态转移 向Q增大的方向移动
```
## Requirement:
* python2.7
* pygame
* tensorflow
* opencv2

## Usage
```
python train.py
```
