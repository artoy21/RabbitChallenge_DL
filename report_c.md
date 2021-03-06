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
- 重み1のデルタ<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_1}" />は、<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;u_2}\frac{\partial&space;u_2}{\partial&space;z_1}\frac{\partial&space;z_1}{\partial&space;u_1}\frac{\partial&space;u_1}{\partial&space;w_1}=-0.450\times&space;w_2(=2.0)\times&space;x(=0.1)" />となっている
- 1次元の簡単な設定ながら、誤差逆伝播法によって偏微分を出力層から入力層に向かって順次計算できていることが分かる

## 勾配消失問題

### 要点のまとめ
- 勾配消失問題
  - 誤差逆伝播法が下位層に進んでいくにつれ、勾配がどんどん緩やかになっていく問題
  - 勾配降下法によるパラメータ更新では、下位層のパラメータがほとんど変わらず、学習が最適値に収束しなくなる
- 活性化関数の影響
  - シグモイド関数
    - 入力値の絶対値が大きくなると勾配（の絶対値）が0に近付くため、勾配消失問題を引き起こす事がある<br/>```def sigmoid(x): return 1/(1+np.exp(-x))```
    - シグモイド関数の微分値は、最大（入力が0の時）でも0.25と1より小さい
  - ReLU関数
    - もっとも使われている活性化関数<br/>```def relu(x): return np.maximum(x, 0)```
    - 勾配消失問題の回避とスパース化に貢献し、よい成果をもたらしている
- 重みの初期値設定
  - Xavier
    - 重みの要素を、前の層のノード数の平方根で除算する
  - He
    - 重みの要素は、前の層のノード数の平方根で除算し、<img src="https://latex.codecogs.com/gif.latex?\sqrt{2}" />を掛ける
- バッチ正規化
  - ミニバッチ単位で、入力値のデータの偏りを抑制する手法
  - 活性化関数を適用する前後に、バッチ正規化層を加える
    - ミニバッチごとに、学習データの平均と分散を<img src="https://latex.codecogs.com/gif.latex?\beta" />（シフトパラメータ）と<img src="https://latex.codecogs.com/gif.latex?\gamma"/>（スケールパラメータ）に変換する
  - ニューラルネットが学習しやすくなり精度や速度が上がる
### 実装演習結果
- MNISTの学習データを784（入力）-40-20-10（出力）のネットワークで学習
- 活性化関数、重みの初期値設定を変えて比較
<img src="https://user-images.githubusercontent.com/34636490/119218930-a7e23e80-bb1d-11eb-8877-cfec15dd886d.png" width=800 />

### 考察
- 活性化関数にシグモイド関数を使い、重みを正規分布で初期化したモデルでは、学習が進まず勾配消失問題が起きていると考えられる
  - 重みにXavier初期化を適用すると、学習が進むものの、シグモイド関数を活性化関数に使用すると最適値に収束するまでの学習スピードは遅いとみられる
- 活性化関数にReLU関数を使うと、重みを正規分布で初期化したモデルでも、最適値に収束する
  - 重みにHe初期化を適用すると、学習スピードが一層速くなる

## 学習率最適化手法
### 要点のまとめ
- 学習率の決め方
  - 初期の学習率は大きく設定し、徐々に学習率を小さくしていく
  - パラメータごとに学習率を可変にする
- 学習率最適化手法
  - モメンタム
    - <img src="https://latex.codecogs.com/gif.latex?w^{(t+1)}=w^{(t)}+V_t,\:\:V_t=\mu&space;V_{t-1}-\eta\nabla&space;E" />
    - <img src="https://latex.codecogs.com/gif.latex?\eta"/>：学習率、<img src="https://latex.codecogs.com/gif.latex?\mu" />：慣性パラメータ
    - 局所的最適解を抜け出て、大域的最適解に早く到達しやすい
  - AdaGrad
    - <img src="https://latex.codecogs.com/gif.latex?w^{(t+1)}=w^{(t)}-\eta\frac{1}{\sqrt{h_t}+\theta&space;}\nabla&space;E,\:\:h_t=h_{t-1}+(\nabla&space;E)^2,\:\:h_0=\theta" />
    - <img src="https://latex.codecogs.com/gif.latex?\theta"/>は分母をゼロにしないための小さな定数
    - 学習率を徐々に小さくすることで勾配の緩やかな誤差関数に対して最適値に近付けやすい一方、鞍点問題を引き起こすことがある
  - RMSProp
    - <img src="https://latex.codecogs.com/gif.latex?w^{(t+1)}=w^{(t)}-\eta\frac{1}{\sqrt{h_t}+\theta&space;}\nabla&space;E,\:\:h_t=\alpha&space;h_{t-1}+(1-\alpha&space;)(\nabla&space;E)^2" />
    - <img src="https://latex.codecogs.com/gif.latex?\alpha"/>は学習率の減衰度合いを調整するハイパーパラメータ
    - 大域的最適解に到達しやすい
  - Adam
    - モメンタムとRMSPropを組み合わせたアルゴリズム
### 実装演習結果
- Adamにおける学習率パラメータの違いによる学習結果の比較
<img src="https://user-images.githubusercontent.com/34636490/119220869-86865000-bb27-11eb-8b52-39633f2fdf49.png" width=800 />

### 考察
- 学習率パラメータが0.01～0.1の場合は、学習率パラメータの設定値によらず学習が早い段階で収束しやすいものとみられる
- 学習率パラメータを小さくし過ぎると（0.001の場合）、Adamによって学習率を更新しても学習が遅くなると言える
- また、学習率パラメータを大きくし過ぎると（0.9の場合）、Adamによって学習率を更新しても収束しない場合があると言える

## 過学習
### 要点のまとめ
- 過学習の原因
  - ネットワークの自由度が高い
    - パラメータの数が多い
    - パラメータの値が適切でない
    - ノードが多い
- 正則化
  - ネットワークの自由度を抑制
    - weight decay
      - 重みが大きいパラメータは学習において重要だが、重みが大きすぎると過学習が起こる
      - 過学習が起こりそうな重みの大きさ以下で重みをコントロール
  - L1正則化
    - 誤差関数に<img src="https://latex.codecogs.com/gif.latex?\lambda\|&space;w\|" />を加える
    - <img src="https://latex.codecogs.com/gif.latex?\lambda"/>はweight decayハイパーパラメータ
    - Lasso推定量（スパース推定）
  - L2正則化
    - 誤差関数に<img src="https://latex.codecogs.com/gif.latex?\lambda\sqrt{\|&space;w\|^2}" />を加える
    - Ridge推定量（縮小推定）
- ドロップアウト
  - ランダムにノードを削除して学習
  - データ量を変化させずに、異なるモデルを学習していると解釈できる
  
### 実装演習結果
- L2正則化のweight decayパラメータを変えて、正則化の強さを確認

<img src="https://user-images.githubusercontent.com/34636490/119262960-14416880-bc18-11eb-8406-09df39d76094.png" width=800 />


### 考察
- 正則化においては、weight decayパラメータをある程度大きくすることで、過学習を防ぐ効果がみられる
  - <img src="https://latex.codecogs.com/gif.latex?\lambda=0.01"/>では学習データの正解率が100％近い一方、検証データの正解率が70％強しか出ておらず、過学習に陥っていた
  - <img src="https://latex.codecogs.com/gif.latex?\lambda=0.1"/>の場合は学習データの正解率が90％程度に下がったが、検証データの正解率は70％程度を維持
  - 一方、<img src="https://latex.codecogs.com/gif.latex?\lambda=1"/>の場合は正則化項の影響が大きくなりすぎ、学習が進まない結果となった

## 畳み込みニューラルネットワークの概念
### 要点のまとめ
- 畳み込み層
  - 画像の場合、縦、横、チャネル（RGB等）の3次元の空間情報も学習できるような層
  - 入力画像に、フィルタを適用し、出力画像を作成
  - パディング
    - 入力画像に比べて出力画像のサイズが小さくなるため、入力画像の周囲に0などの数値を加えて入力画像のサイズを大きくする
  - ストライド
    - フィルタを適用する際に、縦、横それぞれ何画素か飛ばして適用する
- プーリング層
  - 入力画像に対し、局所的な領域の最大値や平均値を取って、出力画像を作成
### 実装演習結果
- シンプルな畳み込みネットワークでMNISTデータを学習
  - 28x28x1次元の入力画像 > 5x5x30の畳み込みフィルタ層（活性化関数はReLU） > 2x2のMAXプーリング層<br/>> 全結合層（100次元、ReLU） > 全結合層（10次元、ソフトマックス）

<img src="https://user-images.githubusercontent.com/34636490/119425590-a5146300-bd42-11eb-8a55-1885c0e16646.png" width=400 />

### 考察
- 学習データの正解率は99%を超え、検証データにおいても97%と、精度の高いモデルが学習できたとみられる
- なお、実装演習用コード：2_6_simple_convolution_network_after.ipynbの```def col2im```において、<br/>
```img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]```となっているところは<br/>
```img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]```ではないかと思われる
## 最新のCNN
### 要点のまとめ
- AlexNet
  - ネットワークの構造（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119358983-18d35300-bce4-11eb-8091-515de166919a.png" width=800 />
  - 2012年に開かれた画像認識コンペティションILSVRC2012で2位に大差をつけて優勝したモデル
    - AlexNetの登場で、ディープラーニングが大きく注目を集める
  - 224x224x3次元の入力画像に、畳み込み層を5回、MAXプーリングを2回、全結合層を3回適用
  - 過学習を防ぐために、出力層の近くの全結合層の出力にドロップアウトを使用
### 実装演習結果
- 深い畳み込みネットワークでMNISTデータを学習
  - 畳み込み層が6層、プーリング層が3層、全結合層＋ドロップアウトが2層
  - 活性化関数はReLU（出力層ではソフトマックス）
<img src="https://user-images.githubusercontent.com/34636490/119433003-4d7cf400-bd50-11eb-80a3-7cf087978fd1.png" width=400 />

### 考察
- 学習データの正解率は99%超でシンプルな畳み込みネットワークと同程度であるが、検証データの正解率は98%を超え、汎化性能がより高まったと考えられる
