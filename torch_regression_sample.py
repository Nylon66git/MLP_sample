import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ニューラルネットワークでの再現を試みる一変数関数
def func(x):
    '''
    ニューラルネットワークの最終層の活性化関数を sigmoid とする予定なので，
    この関数 func は値域が 0 以上 1 以下の範囲に収まるように設定するのが望ましい
    '''
    y = 0.3 * (1.4 + math.sin(x))
    return y

# 上記の関数の定義域（入力変数 x の範囲）
# ここで設定した最小値・最大値は，それぞれ X_RANGE[0], X_RANGE[1] で参照できる
X_RANGE = [0, 2 * math.pi]

# 学習条件の設定
BATCH_SIZE = 25 # バッチサイズ
N_EPOCHS = 1000 # 何エポック分，学習処理を回すか（「事例集合をミニバッチに分割 -> 各ミニバッチを1回ずつ用いてパラメータ更新」を1セット分行うことを「エポック」と呼ぶ）


# 事例を一つ生成する関数
def make_sample(noise_level=0):
    x = np.random.rand() # 0 以上 1 未満の一様乱数を一つ生成
    x = (X_RANGE[1] - X_RANGE[0]) * x + X_RANGE[0] # 乱数の値が X_RANGE[0] 以上 X_RANGE[1] 未満となるように変換
    e = np.random.randn() # 平均 0 ，分散 1 の正規乱数を一つ生成
    y = func(x) + noise_level * e # x に対応する関数値 y を求め，ノイズ e を noise_level 倍して付加する
    return x, y

# n_samples 個の事例からなるデータセットを作成する関数
def make_dataset(n_samples, noise_level=0):
    x_set = [] # 事例集合（データセット），最初は空
    y_set = [] # 同上
    for i in range(n_samples):
        x, y = make_sample(noise_level) # 事例を一つ生成
        x_set.append(x) # 事例集合に追加
        y_set.append(y) # 同上
    x_set = np.asarray(x_set, dtype=np.float32) # numpy.ndarray（numpyの配列）型に変換しておく
    y_set = np.asarray(y_set, dtype=np.float32) # 同上
    return x_set, y_set


# ニューラルネットワーク（2回目の講義資料 32 スライド目の構造に定数入力 b を追加したもの）の設計
class TorchNet(nn.Module):

    def __init__(self, m):
        super(TorchNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, m), # 入力層（1次元）-> 中間層（m 個のパーセプトロン）
            nn.Sigmoid(), # 活性化関数
            nn.Linear(m, 1), # 中間層（m 個のパーセプトロン）-> 出力層（1次元）
            nn.Sigmoid(), # 活性化関数
        )

    def forward(self, x):
        return self.layers(x)


# C言語のメイン関数に相当するもの（という認識でOK）
if __name__ == '__main__':

    # 事例数 200 のデータセットを作成
    N = 200 # 事例数
    x_set, y_set = make_dataset(N, noise_level=0.05)

    # 作成したデータセットを散布図としてプロット
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('generated dataset')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(X_RANGE)
    ax1.set_ylim(-0.2, 1.2)
    ax1.scatter(x_set, y_set, s=5)
    plt.pause(1)

    # 5個のパーセプトロンからなる中間層を持つニューラルネットワークを用意
    net = TorchNet(m=5)

    # デバイスの指定とオプティマイザーの用意（基本このままでOK）
    device = 'cpu'
    net = net.to(device)
    optimizer = optim.Adam(net.parameters())

    # 損失関数を用意
    loss_func = nn.MSELoss() # 二乗誤差を計算する関数に相当

    # 学習経過を表示する準備
    ax2 = fig.add_subplot(1, 2, 2)

    # 先ほど作成したデータセットを用いてニューラルネットワークを学習
    for epoch in range(N_EPOCHS):

        net.train()
        n_input = 0
        sum_loss = 0
        perm = np.random.permutation(N) # データセットをランダムに並び替え
        for i in range(0, N, BATCH_SIZE): # i を BATCH_SIZE ずつ増やしながら処理を繰り返す
            net.zero_grad()
            x = torch.tensor(x_set[perm[i : i + BATCH_SIZE]].reshape((BATCH_SIZE, 1)), device=device) # i 番目から (i + BATCH_SIZE - 1) 番目までの事例でミニバッチ x を構成
            y_truth = torch.tensor(y_set[perm[i : i + BATCH_SIZE]].reshape((BATCH_SIZE, 1)), device=device) # 上記の x に対応する正解値を用意
            loss = loss_func(net(x), y_truth) # 上記の x および y_truth を用いてパラメータを 1 回更新：ここから
            loss.backward()
            optimizer.step() # ここまで
            sum_loss += float(loss) * len(x)
            n_input += len(x)
        sum_loss /= n_input

        if epoch < 10 or (epoch < 100 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            net.eval()
            x_test = np.arange(X_RANGE[0], X_RANGE[1], 0.01).astype(np.float32) # X_RANGE の範囲を 0.01 刻みで等分割しテストデータを作成
            y_test = net(torch.tensor(x_test.reshape((len(x_test), 1)), device=device)) # 現在のニューラルネットワークにテストデータを入力
            y_test = y_test.detach().numpy().copy().reshape(-1)
            ax2.cla()
            ax2.set_title('estimated function')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_xlim(X_RANGE)
            ax2.set_ylim(-0.2, 1.2)
            ax2.scatter(x_test, y_test, s=2, c='red') # 現在の結果をグラフ表示
            plt.pause(1)
            print('Epoch {0}: train loss = {1}'.format(epoch + 1, sum_loss)) # 現在の損失関数の値を出力
