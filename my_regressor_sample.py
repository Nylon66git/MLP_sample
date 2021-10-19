import math
import numpy as np
import matplotlib.pyplot as plt


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


# 活性化関数として使うシグモイド関数の定義
def g(h):
    return 1 / (1 + np.exp(-h))

# シグモイド関数の微分（2回目の講義資料 34 スライド目を参照）の定義
def dg(h):
    return (1 - g(h)) * g(h)


# ニューラルネットワーク（2回目の講義資料 32 スライド目のもの）の設計
class myNet():

    # 学習率の設定とパラメータの初期化
    #   - m: 中間層を構成するパーセプトロンの数
    def __init__(self, m):
        self.alpha = 0.1 # 学習率
        self.w = np.random.randn(m) # パラメータ w[0] ～ w[m-1] : m 個分の要素を持つ配列，正規乱数を用いてランダムに初期化
        self.a = np.random.randn(m) # パラメータ a[0] ～ a[m-1] : m 個分の要素を持つ配列，正規乱数を用いてランダムに初期化
        self.m = m # 整数，パーセプトロン数を記憶しておく

    # 入力 x から y の推定値を計算
    #   - x: ミニバッチ（ batch_size 個分の要素を持つ配列 ）
    def forward(self, x):
        self.batch_size = len(x) # 整数，バッチサイズを記憶しておく
        self.u = x.reshape((self.batch_size, 1)) @ self.w.reshape((1, self.m)) # u[i,j] = x[i] * w[j] ，なお u は batch_size 行 m 列の行列（二次元配列）となる
        self.h = g(self.u) # h[i,j] = g(u[i,j]) ，この h も batch_size 行 m 列の行列
        self.z = np.zeros(self.batch_size) # z 用のメモリを確保，なお z は batch_size 個分の要素を持つ配列
        for i in range(self.batch_size):
            self.z[i] = np.sum(self.a * self.h[i]) # z[i] = Σj(a[j] * h[i, j])
        self.y_predicted = g(self.z) # 推定値 y[i] = g(z[i]) ，なお y も batch_size 個分の要素を持つ配列
        return self.y_predicted

    # 入力 x とそれに対応する y の正解値を用いてパラメータ更新
    #   - x: ミニバッチ（ batch_size 個分の要素を持つ配列 ）
    #   - y_truth: 上記 x に対応する正解値（ batch_size 個分の要素を持つ配列 ）
    def update(self, x, y_truth):
        self.y_predicted = self.forward(x) # まず y の推定値を求める
        self.e = self.y_predicted - y_truth # 誤差 e[i] = y[i] - y_hat[i] を求める，なお e は batch_size 個分の要素を持つ配列
        self.da = np.zeros(self.m) # dL/da 用のメモリ確保，なお da は m 個分の要素を持つ配列
        self.dw = np.zeros(self.m) # dL/dw 用のメモリ確保，なお dw は m 個分の要素を持つ配列
        for j in range(self.m):
            self.da[j] = (2 / self.m) * np.sum(self.e * dg(self.z) * self.h[:, j]) # dL/da[j] の計算
            self.dw[j] = (2 * self.a[j] / self.m) * np.sum(self.e * dg(self.z) * dg(self.u[:, j]) * x) # dL/dw[j] の計算
        self.a = self.a - self.alpha * self.da # a[j] <- a[j] - alpha * dL/da[j] としてパラメータ更新
        self.w = self.w - self.alpha * self.dw # w[j] <- w[j] - alpha * dL/dw[j] としてパラメータ更新
        return np.mean(self.e ** 2) # 誤差の二乗平均，つまり (1/batch_size) * Σi(y[i] - y_hat[i])^2 を返す


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
    net = myNet(m=5)

    # 学習経過を表示する準備
    ax2 = fig.add_subplot(1, 2, 2)

    # 先ほど作成したデータセットを用いてニューラルネットワークを学習
    for epoch in range(N_EPOCHS):

        n_input = 0
        sum_loss = 0
        perm = np.random.permutation(N) # データセットをランダムに並び替え
        for i in range(0, N, BATCH_SIZE): # i を BATCH_SIZE ずつ増やしながら処理を繰り返す
            x = x_set[perm[i : i + BATCH_SIZE]] # i 番目から (i + BATCH_SIZE - 1) 番目までの事例でミニバッチ x を構成
            y_truth = y_set[perm[i : i + BATCH_SIZE]] # 上記の x に対応する正解値を用意 
            sum_loss += net.update(x, y_truth) * len(x) # 上記の x および y_truth を用いてパラメータを 1 回更新
            n_input += len(x)
        sum_loss /= n_input

        if epoch < 10 or (epoch < 100 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            x_test = np.arange(X_RANGE[0], X_RANGE[1], 0.01).astype(np.float32) # X_RANGE の範囲を 0.01 刻みで等分割しテストデータを作成
            y_test = net.forward(x_test) # 現在のニューラルネットワークにテストデータを入力
            ax2.cla()
            ax2.set_title('estimated function')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_xlim(X_RANGE)
            ax2.set_ylim(-0.2, 1.2)
            ax2.scatter(x_test, y_test, s=2, c='red') # 現在の結果をグラフ表示
            plt.pause(1)
            print('Eepoch {0}: train loss = {1}'.format(epoch + 1, sum_loss)) # 現在の損失関数の値を出力
