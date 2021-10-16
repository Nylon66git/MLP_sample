import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from classifier_visualizer import ClassifierVisualizer


# データファイルの存在するフォルダ
# ここでは python プログラムと同じフォルダに存在するものとする
DATA_DIR = './'

# 学習データセット（CSVファイル）のファイル名
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'weather_train.csv')

# テストセット（CSVファイル）のファイル名
TEST_DATA_FILE = os.path.join(DATA_DIR, 'weather_test.csv')

# 学習データセットおよびテストセットのファイル形式
# ここでは UTF-8 を想定する
FILE_ENCODING = 'utf_8'

# 学習条件の設定
BATCH_SIZE = 64 # バッチサイズ
N_EPOCHS = 100 # 何エポック分，学習処理を回すか


# データ読み込み関数
def read_data(filename, encoding):

    # ファイル名 filename のファイルを CSV ファイルとしてオープン
    f = open(filename, 'r', encoding=encoding)
    reader = csv.reader(f)

    # ヘッダ（項目名が記載されている先頭の行）は読み捨てる
    next(reader)

    # データを読み込む
    x_set = []
    y_set = []
    for row in reader: # 行ごとに...
        vec = [0.0, 0.0] # 空の2次元ベクトルを作成
        vec[0] = float(row[1]) # 左から2列目の値（気温）を2次元ベクトルの1次元目にセット
        vec[1] = float(row[2]) # 左から3列目の値（湿度）を2次元ベクトルの2次元目にセット
        if row[3] == '晴': # 左から4列目の値（天気）が「晴」のとき...
            lab = 0 # クラスラベル「晴」を表す整数値として 0 を設定
        elif row[3] == '曇': # 「曇」のとき...
            lab = 1 # クラスラベル「曇」を表す整数値として 1 を設定
        else: # 「雨」のとき...
            lab = 2 # クラスラベル「雨」を表す整数値として 2 を設定
        x_set.append(vec)
        y_set.append(lab)

    # ファイルをクローズ
    f.close()

    # 読み込んだデータを numpy.ndarray 型に変換
    x_set = np.asarray(x_set, dtype=np.float32) # 32bit浮動小数点数型に
    y_set = np.asarray(y_set, dtype=np.int64) # 64bit整数型に

    return x_set, y_set


# 天気認識AIを実現するニューラルネットワーク
class WeatherPredictor(nn.Module):

    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10), # 一層目
            nn.ReLU(), # 活性化関数
            nn.Linear(10, 10), # 二層目
            nn.ReLU(), # 活性化関数
            nn.Linear(10, 3), # 三層目
        )

    def forward(self, x):
        return self.layers(x)


# C言語のメイン関数に相当するもの
if __name__ == '__main__':

    # データ読み込み
    x_train, y_train = read_data(TRAIN_DATA_FILE, encoding=FILE_ENCODING)
    x_test, y_test = read_data(TEST_DATA_FILE, encoding=FILE_ENCODING)
    n_train_samples = len(x_train) # 読み込んだデータ数を記憶しておく
    n_test_samples = len(x_test)

    # ニューラルネットワークの作成
    net = WeatherPredictor()

    # 損失関数の定義
    loss_func =  nn.CrossEntropyLoss()

    # デバイスの指定とオプティマイザーの用意（基本このままでOK）
    device = 'cpu'
    net = net.to(device)
    optimizer = optim.Adam(net.parameters())

    # 可視化の準備
    a = np.min(x_test, axis=0)
    b = np.max(x_test, axis=0)
    hrange = ((11 * a[0] - b[0]) / 10, (11 * b[0] - a[0]) / 10) # 可視化結果における横軸の範囲
    vrange = ((11 * a[1] - b[1]) / 10, (11 * b[1] - a[1]) / 10) # 可視化結果における縦軸の範囲
    test_data = [y_test, x_test]
    ccolors = [[255, 0, 0], [127, 127, 0], [0, 0, 255]]
    visualizer_hlabel = 'temperature (degree)' # 可視化結果における横軸のラベル
    visualizer_vlabel = 'humidity (%)' # 可視化結果における縦軸のラベル
    visualizer = ClassifierVisualizer(n_classes=3, clabels=['sunny', 'cloudy', 'rainy'], hrange=hrange, vrange=vrange, hlabel=visualizer_hlabel, vlabel=visualizer_vlabel)

    # 学習データセットを用いてニューラルネットワークを学習
    for epoch in range(N_EPOCHS):

        net.train()
        n_input = 0
        sum_loss = 0
        perm = np.random.permutation(n_train_samples)
        for i in range(0, n_train_samples, BATCH_SIZE):
            net.zero_grad()
            x = torch.tensor(x_train[perm[i : i + BATCH_SIZE]], device=device)
            y = torch.tensor(y_train[perm[i : i + BATCH_SIZE]], device=device)
            loss = loss_func(net(x), y)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            n_input += len(x)
        sum_loss /= n_input

        if epoch < 10 or (epoch < 100 and (epoch + 1) % 5 == 0):
            net.eval()
            n_input = 0
            n_failed = 0
            sum_test_loss = 0
            perm = np.arange(0, n_test_samples)
            for i in range(0, n_test_samples, BATCH_SIZE):
                x = torch.tensor(x_test[i : i + BATCH_SIZE], device=device)
                y = torch.tensor(y_test[i : i + BATCH_SIZE], device=device)
                z = net(x)
                loss = loss_func(z, y)
                sum_test_loss += float(loss) * len(x)
                n_input += len(x)
                y_cpu = y.to('cpu').detach().numpy().copy()
                z_cpu = z.to('cpu').detach().numpy().copy()
                n_failed += np.count_nonzero(np.argmax(z_cpu, axis=1) - y_cpu)
            sum_test_loss /= n_input
            accuracy = (n_test_samples - n_failed) / n_test_samples
            visualizer.show(net, device=device, class_colors=ccolors, samples=test_data, title='Epoch {0}'.format(epoch + 1)) # グラフを表示
            print('Epoch {0}:'.format(epoch + 1)) # 現在の損失関数の値，および認識精度を出力：ここから
            print('  train loss = {0:.6f}'.format(sum_loss))
            print('  valid loss = {0:.6f}'.format(sum_test_loss))
            print('  accuracy = {0:.2f}%'.format(100 * accuracy))
            print('') # ここまで
