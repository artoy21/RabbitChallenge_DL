# 深層学習（前半）レポート
1. [day1 Section1:入力層～中間層](#入力層～中間層)
2. [day1 Section2:活性化関数](#活性化関数)
3. [day1 Section3:出力層](#出力層)
4. [day1 Section4:勾配降下法](#勾配降下法)
5. [day1 Section5:誤差逆伝播法](#誤差逆伝播法)
6. [day2 Section1:勾配消失問題](#勾配消失問題)
7. [day2 Section2:学習率最適化手法](#学習率最適化手法)
8. [day2 Section3:過学習](#過学習)
9. [day2 Section4:畳み込みニューラルネットワークの概念](#畳み込みニューラルネットワークの概念)
10. [day2 Section5:最新のCNN](#最新のCNN)


## 入力層～中間層

### 要点のまとめ
- 入力層には学習データの各特徴量（例：身長、体重、写真の画素値、・・・）が変数として与えられる
- 入力層の各入力値を重みパラメータで加重平均し、バイアスパラメータを加えた結果が中間層に渡される
- 入力層の入力を<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{x}=(x_1,\cdots,x_n)^T"/>、重みパラメータを<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{w}=(w_1,\cdots,w_n)^T"/>、バイアスパラメータを<img src="https://latex.codecogs.com/gif.latex?b" />として、中間層に渡される<img src="https://latex.codecogs.com/gif.latex?u" />は<br/><img src="https://latex.codecogs.com/gif.latex?u=w_1x_1+w_2x_2+\cdots+w_nx_n+b=\boldsymbol{x}^T\boldsymbol{w}+b" />
### 実装演習結果
- 入力層の入力と重み、バイアスを適当に設定し、中間層の入力を計算
<img src="https://user-images.githubusercontent.com/34636490/118356380-fd639c00-b5af-11eb-9a94-02d0e26a3514.png" width=250>

### 考察
- 重みパラメータは入力と同じ数だけ用意する必要がある
- 重み、及びバイアスパラメータは適当な（初期）値を設定する必要がある

## 活性化関数

### 要点のまとめ
- 次の層への出力の大きさを決める**非線形の**関数
- 入力値の値によって、次の層への信号のON/OFFや強弱を定める働きをもつ
- 中間層用の活性化関数
  - ReLU関数（最も使われている活性化関数）
  - シグモイド関数（勾配消失問題を引き起こすことがある）
  - ステップ関数（線形分離可能なものしか学習できないため、実際には使われない）
- 出力層用の活性化関数
  - シグモイド関数
  - ソフトマックス関数
  - 恒等写像

### 実装演習結果

<img src="https://user-images.githubusercontent.com/34636490/118391064-5c3b1b00-b66d-11eb-8572-b9e127bae731.png" width=250>

### 考察


## 出力層

## 勾配降下法

## 誤差逆伝播法

## 勾配消失問題

## 学習率最適化手法

## 過学習

## 畳み込みニューラルネットワークの概念

## 最新のCNN
