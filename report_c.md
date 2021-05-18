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
- 中間層と出力層で利用される活性化関数が異なる
- 中間層用の活性化関数
  - ReLU関数（最も使われている活性化関数）
  - シグモイド関数（勾配消失問題を引き起こすことがある）
  - ステップ関数（線形分離可能なものしか学習できないため、実際には使われない）
- 出力層用の活性化関数
  - シグモイド関数
  - ソフトマックス関数
  - 恒等写像

### 実装演習結果

<img src="https://user-images.githubusercontent.com/34636490/118391100-93113100-b66d-11eb-8ed1-af0cc0a9a697.png" width=250>

### 考察
- 入力値の線形結合を非線形変換するのが活性化関数の役割とみられる
- 0～1の確率表現をしたい場合など、目的に応じた活性化関数を選択する必要がある

## 出力層

### 要点のまとめ
- 出力層の役割
  - ニューラルネットワークで予測したい項目に関する数値の出力
  - 回帰問題であれば予測値
  - 分類問題であれば各クラスに属する確率値
- 誤差関数
  - 平均二乗誤差（回帰問題）
  - クロスエントロピー誤差（分類問題）

### 実装演習結果
- 多クラス分類ネットワークの構成を3-5-4に変更して実装した結果
<img src="https://user-images.githubusercontent.com/34636490/118392524-14b88d00-b675-11eb-8392-edd0aac65e07.png" width=300>

- 回帰ネットワークの構成を3-5-4に変更して実装した結果
<img src="https://user-images.githubusercontent.com/34636490/118392634-8abcf400-b675-11eb-94be-da0ad5e131de.png" width=250>

- 2値分類ネットワークの構成を5-10-1に変更して実装した結果
<img src="https://user-images.githubusercontent.com/34636490/118393439-22244600-b67a-11eb-89d5-e8d87e5abf85.png" width=450>

### 考察
- 出力層の出力値を決める関数、及び誤差関数は、扱う問題が回帰か二値分類か多クラス分類か等に応じて適切に設定する必要がある

## 勾配降下法
### 要点のまとめ
- パラメータを最適化するために、誤差関数を最小化するパラメータを探索する手法<br/><img src="https://latex.codecogs.com/gif.latex?w^{(t+1)}=w^{(t)}-\eta\frac{\partial&space;E}{\partial&space;w}" />
- 学習率<img src="https://latex.codecogs.com/gif.latex?\eta" />の値によって学習効率が大きく異なる
  - 学習率が大きすぎると、最小値にたどり着かずに発散してしまう
  - 学習率が小さすぎると、収束するまでに時間がかかってしまう
- 学習率の決定アルゴリズム
  - Momentum
  - AdaGrad
  - Adadelta
  - Adam
- 確率的勾配降下法（SGD)
  - パラメータの更新ごとに全サンプルの平均誤差を用いるのではなく、ランダムに抽出したサンプルの誤差を使用
  - データが冗長な場合の計算コストの削減
  - 極小値に収束するリスクの軽減
  - オンライン学習（⇔バッチ学習）が可能
- ミニバッチ勾配降下法
  - ランダムに分割したサンプルの部分集合（ミニバッチ）の平均誤差を使用
  - 全サンプルを使用する勾配降下法と、ランダムな1サンプルを使用するSGDの中間
  - SGDのメリットを損なわず、コンピュータの計算資源を有効利用できる

### 実装演習結果
- 実装演習用コードの重み1の更新を計算
<img src="https://user-images.githubusercontent.com/34636490/118654806-e75e1180-b823-11eb-82b3-e43dfe8ad6cf.png" width=300 />

### 考察
- 更新後パラメータは、初期値から学習率×勾配（偏微分係数）を引いた値となっている
- 全てのパラメータにおいて勾配が負であるため、パラメータを大きくすることで誤差関数を小さくする方向に更新できている

## 誤差逆伝播法
### 要点のまとめ
- 勾配の計算
  - 数値微分では各パラメータについて誤差関数を計算するために、ネットワークの順伝播計算を繰り返し行い計算負荷が大きい
  - 誤差逆伝播法の利用で計算負荷を小さくできる
- 誤差逆伝播法
  - 参集された誤差を、出力層側から順に微分し、入力層に近い方へと伝播
  - パラメータの微分値を**解析的に**計算する手法
- 順伝播と逆伝播のイメージ（講義資料より）
<img src="https://user-images.githubusercontent.com/34636490/118656195-440dfc00-b825-11eb-85cc-7c7093d9fc2d.png" width=600 />

### 実装演習結果
- 簡単のため、入力層を1次元、中間層も1次元とした二値分類問題を想定
- 入力層の活性化関数はReLU、中間層の活性化関数はシグモイド関数
- 誤差関数はクロスエントロピー誤差
<img src="https://user-images.githubusercontent.com/34636490/118664663-9ef72180-b82c-11eb-98e2-e2bd86ec19f7.png" width=250 />

### 考察
- 出力層でのデルタ<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;u_2}" />は、シグモイド関数による出力値（0.550）－正解ラベル（1）となっている
- 重み2のデルタ<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_2}" />は、<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;u_2}\frac{\partial&space;u_2}{\partial&space;w_2}=-0.450\times&space;u_1(=0.1)" />となっている
- 中間層でのデルタ<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;u_1}" />は、<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;u_2}\frac{\partial&space;u_2}{\partial&space;z_1}\frac{\partial&space;z_1}{\partial&space;u_1}=-0.450\times&space;w_2(=2.0)" />となっている（<img src="https://latex.codecogs.com/gif.latex?u_1>0"/>のため、ReLU関数の微分は1）
- 重み1のデルタ<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_1}" />は、<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;u_2}\frac{\partial&space;u_2}{\partial&space;z_1}\frac{\partial&space;z_1}{\partial&space;u_1}\frac{\partial&space;u_1}{\partial&space;x_1}=-0.450\times&space;w_2(=2.0)\times&space;x(=0.1)" />となっている

## 勾配消失問題

## 学習率最適化手法

## 過学習

## 畳み込みニューラルネットワークの概念

## 最新のCNN
