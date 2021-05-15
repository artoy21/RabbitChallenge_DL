# 機械学習レポート
1. [線形回帰モデル](#線形回帰モデル)
2. [非線形回帰モデル](#非線形回帰モデル)
3. [ロジスティック回帰モデル](#ロジスティック回帰モデル)
4. [主成分分析](#主成分分析)
5. [k近傍法（kNN）・k-means](#アルゴリズム)
6. [サポートベクターマシン（SVM）](#サポートベクターマシン)

- 各手法の位置付け（講義資料より引用）
<img src="https://user-images.githubusercontent.com/34636490/116880664-408f4800-ac5d-11eb-9c85-572ab9d5921f.png" width="800">


## 線形回帰モデル
- 教師あり学習の一種
- 定式化
  - 説明変数を<img src="https://latex.codecogs.com/gif.latex?\vec{x}=(1,x_1,x_2,\cdots,x_m)^T\in\mathbb{R}^{m+1}" />、目的変数を<img src="https://latex.codecogs.com/gif.latex?y\in\mathbb{R}" />
  - パラメータを<img src="https://latex.codecogs.com/gif.latex?\vec{w}=(w_0,w_1,w_2,\cdots,w_m)^T\in\mathbb{R}^{m+1}" />、誤差項を<img src="https://latex.codecogs.com/gif.latex?\varepsilon\in\mathbb{R}" />
  - として、<img src="https://latex.codecogs.com/gif.latex?y=\vec{w}^T\vec{x}+\varepsilon" />
  - 教師データが<img src="https://latex.codecogs.com/gif.latex?1,\cdots,n" />まである時、<img src="https://latex.codecogs.com/gif.latex?X=(\vec{x}_1,\vec{x}_2,\cdots,\vec{x}_n)^T,\:\vec{y}=(y_1,y_2,\cdots,y_n)^T,\:\vec{\varepsilon}=(\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n)^T" />として、<br/><img src="https://latex.codecogs.com/gif.latex?\vec{y}=X\vec{w}+\vec{\varepsilon}" />
- パラメータの推定
  - 最小二乗法を利用
    - 教師データの平均二乗誤差を最小とするパラメータを求める手法
  - 平均二乗誤差を<img src="https://latex.codecogs.com/gif.latex?\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^n\left(\hat{y}_i-y_i\right)^2" />として、パラメータに対する勾配が0になる点を求める<br/><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;w}\mathrm{MSE}=0\:\Rightarrow\hat{w}=(X^TX)^{-1}X^Ty" />
- 予測
  - <img src="https://latex.codecogs.com/gif.latex?\hat{y}=X_{new}\hat{w}" />
### 実装演習
- 設定
  - ボストンの住宅データセットを線形回帰モデルで分析
- 課題
  - 部屋数が4で犯罪率が0.3の物件はいくらになるか？
- コードのキャプション

<img src="https://user-images.githubusercontent.com/34636490/117115070-908a1e00-adc7-11eb-8882-e0eefc9a05d5.png" width="800">

- 結果
  - 4240.1ドルと予測
  - 回帰係数は符号条件を満たす
    - 部屋数が1室多いと、住宅価格は8.39ドル高い
    - 人口当たりの犯罪発生率が1単位高いと、住宅価格は1.13ドル低い

## 非線形回帰モデル
- 教師あり学習の一種
  - データの構造を線形で捉えられる場合は限られる
  - 非線形な構造を捉えられる仕組みが必要
- 定式化
  - 基底展開法
    - 線形回帰モデルの説明変数に、基底関数と呼ばれる既知の非線形関数を適用<br/><img src="https://latex.codecogs.com/gif.latex?y=w_0+\sum_{j=1}^kw_j\phi_j(x)+\varepsilon" />
    - よく使われる基底関数は、多項式関数、ガウス型基底関数、Bスプライン関数
- パラメータの推定
  - 基底展開法も線形回帰モデルと同様に、最小二乗法で推定可能
  - <img src="https://latex.codecogs.com/gif.latex?\phi(x_i)=(\phi_1(x_i),\phi_2(x_i),\cdots,\phi_k(x_i))^T\in\mathbb{R}^k," /><br/><img src="https://latex.codecogs.com/gif.latex?\Phi=(\phi(x_1),\phi(x_2),\cdots,\phi(x_n))^T\in\mathbb{R}^{n\times&space;k} " />として、<br/><img src="https://latex.codecogs.com/gif.latex?\hat{w}=\left(\Phi^T\Phi\right)^{-1}\Phi^Ty" />
- 予測
  - <img src="https://latex.codecogs.com/gif.latex?\hat{y}=\Phi_{new}\hat{w}" />
- 未学習と過学習
  - 未学習
    - 学習データに対して、十分小さな誤差が得られないモデル
    - モデルの表現力が低い　⇒　表現力の高いモデルを利用する
  - 過学習
    - 学習データに対する誤差は小さいが、検証データに対する誤差が大きいモデル
    - 学習データが少ないか、モデルの表現力が高すぎる
  - 過学習への対応策
    - 学習データ数を増やす
    - 不要な基底関数を削除して、モデルの表現力を抑える<br/>基底関数の数、位置、バンド幅を調整（クロスバリデーションなどで選択）
    - 正則化法を利用して、モデルの表現力を抑える
  - 正則化法
    - モデルの複雑さに応じて、値が大きくなる正則化項（罰則項）を付加した目的関数を最小化する
    - Ridge推定量<br/>L2ノルムを利用：パラメータを0に近付けるように推定
    - Lasso推定量<br/>L1ノルムを利用：いくつかのパラメータを正確に0に推定
  - 適切なモデル（汎化性能が高いモデル）は交差検証法で決定
    - 汎化性能：教師データだけでなく、テストデータ・検証データに対する予測性能
    - バイアス（訓練データに対する誤差の大きさ）とバリアンス（検証データに対する誤差のばらつき）の間にはトレードオフの関係がある
  - ホールドアウト法
    - データを学習用とテスト用の2つに分割し、学習用データでパラメータ推定、テスト用データで予測精度を計算
    - 大量のデータが利用可能な場合を除き、良い性能指標を与えないという欠点がある
  - クロスバリデーション（交差検証法）
    - データを3つ以上のグループに分割し、1つのグループをテスト用、残りを学習用に使うホールドアウト法を全パターン計算する
    - 各パターンにおける予測精度の平均を採用する（CV値と呼ぶ）
### 実装演習
- 設定
  - ボストンの住宅データセットを非線形回帰モデルで分析
- 課題
  - 部屋数が4で犯罪率が0.3の物件はいくらになるか、非線形回帰モデルで予測
- コードのキャプション

<img src="https://user-images.githubusercontent.com/34636490/117565403-f9d1a000-b0eb-11eb-93dc-c327b85c3303.png" width=600>

- 結果
  - 17353.4ドルと予測
    - 多項式関数を用いた基底展開法を1次式から4次式まで作成
    - データセットを5分割したクロスバリデーションの結果、cv値（RMSE）が最小となる2次式を採用

## ロジスティック回帰モデル
- 教師あり学習の一種
  - 分類問題に利用
- 定式化
  - ロジスティック線形回帰モデル
    - 入力とパラメータの線形結合をシグモイド関数に入力
    - 出力は<img src="https://latex.codecogs.com/gif.latex?y=1" />になる確率の値になる<br/><img src="https://latex.codecogs.com/gif.latex?P(y=1\mid&space;x)=\sigma(w_0+w_1x_1+\cdots+w_mx_m)" />
  - シグモイド関数
    - 入力は実数、出力は0~1の値（単調増加）<br/><img src="https://latex.codecogs.com/gif.latex?\sigma(x)=\frac{1}{1+\exp(-x)}" />
    - シグモイド関数の微分は、シグモイド関数自身で表現される<br/><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dx}\sigma(x)=\sigma(x)(1-\sigma(x))" />
- パラメータの推定
  - 最尤法を利用する
  - <img src="https://latex.codecogs.com/gif.latex?y_1,y_2,\cdots,y_n" />が同一のベルヌーイ分布に従う独立した確率変数とすると、尤度関数Lは次のようになる<br/><img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AP%28y_1%2Cy_2%2C%5Ccdots%2Cy_n+%5Cmid+x%29+%26%3D+%5Cprod_%7Bi%3D1%7D%5En+P%28y_i%3D1+%5Cmid+x_i%29%5E%7By_i%7D%281-P%28y_i%3D1+%5Cmid+x_i%29%29%5E%7B1-y_i%7D+%5C%5C%0A%26%3D+%5Cprod+_%7Bi%3D1%7D%5En+%5Csigma%28w%5ETx_i%29%5E%7By_i%7D%281-%5Csigma%28w%5ETx_i%29%29%5E%7B1-y_i%7D+%5C%5C%0A%26%3D+L%28w%29%0A%5Cend%7Balign%2A%7D%0A" />
  - 尤度関数の最大化は対数尤度関数に-1を掛けたものの最小化と同値<br/><img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Chat%7Bw%7D+%26%3D+%5Ctext%7Bargmax%7D_wL%28w%29+%5C%5C%0A%26%3D+%5Ctext%7Bargmin%7D_w+%5Cleft%5C%7B+-%5Clog+L%28w%29+%5Cright%5C%7D+%5C%5C%0A%26%3D+%5Ctext%7Bargmin%7D_w+%5Cleft%5C%7B+-%5Csum_%7Bi%3D1%7D%5En%5Cleft%28y_i%5Clog%5Csigma%28w%5ETx_i%29%2B%281-y_i%29%5Clog%281-%5Csigma%28w%5ETx_i%29%5Cright%29+%5Cright%5C%7D%0A%5Cend%7Balign%2A%7D%0A" />
  - 線形回帰モデルと異なり、ロジスティック回帰モデルの解析解を求めることは困難なため、勾配降下法によりパラメータを探索する
- 勾配降下法
  - 最小化したい目的関数の一次微分を計算して、逐次的にパラメータを更新する<br/><img src="https://latex.codecogs.com/gif.latex?w^{k+1}=w^k-\eta\frac{\partial\text{Loss}(w)}{\partial&space;w}" />
  - <img src="https://latex.codecogs.com/gif.latex?\eta" />は学習率と呼ばれるハイパーパラメータで、学習の収束しやすさに影響する
  - ロジスティック回帰モデルの場合、<br/><img src="https://latex.codecogs.com/gif.latex?w^{k+1}=w^k+\eta\sum_{i=1}^n(y_i-\sigma(w^Tx_i))x_i" />
- 確率的勾配降下法
  - 勾配降下法では、パラメータを1回更新する度にn個全てのデータに対する和を求める必要があり、nが大きいとデータをメモリに載せる容量や計算時間が莫大になる
  - データを1つずつランダムに選んでパラメータを更新する手法が確率的勾配降下法（SGD)<br/><img src="https://latex.codecogs.com/gif.latex?w^{k+1}=w^k+\eta(y_i-\sigma(w^Tx_i))x_i" />
- 評価方法
  - 混同行列(confusion matrix)（講義資料より引用）
  
  <img src="https://user-images.githubusercontent.com/34636490/117457228-fae7bd80-af83-11eb-813a-7934e3a12a03.png" width=800>
  
  - 正解率
    - (TP+TN)/(TP+FP+FN+TN)
    - 分類したいクラスに偏りがある場合、正解率はあまり意味をなさないことがほとんど
  - 再現率(Recall)
    - TP/(TP+FN)
    - 「Positiveなデータ」のうち、Positiveと予測できた割合
    - **抜け漏れの少ない**予測をしたい際に有効な指標
  - 適合率(Precision)
    - TP/(TP+FP)
    - 「Positiveと予測」したもので、実際にPositiveなデータの割合
    - **見逃しが多くても**より正確な予測をしたい際に有効な指標
  - F値
    - 2/(1/Recall + 1/Precision)
    - RecallとPrecisionはトレードオフの関係にあるため、調和平均を取ってバランスを示す指標
### 実装演習
- 設定
  - タイタニックの乗客データを利用しロジスティック回帰モデルを作成
- 課題
  - 年齢が30歳で男の乗客は生き残れるか？
- コードのキャプション

<img src="https://user-images.githubusercontent.com/34636490/117982457-f9960680-b370-11eb-8542-c4e8dc54e82d.png" width=500>

- 結果
  - 生存確率は19.3%と計算され、死亡と予測

## 主成分分析
- 教師なし学習の一種
  - 多変量データの持つ構造をより少数個の指標に圧縮（次元圧縮）
- 定式化
  - 学習データを<img src="https://latex.codecogs.com/gif.latex?x\in\mathbb{R}^m" />としたとき、<br/>係数ベクトル<img src="https://latex.codecogs.com/gif.latex?a\in\mathbb{R}^m"/>による線形変換<img src="https://latex.codecogs.com/gif.latex?s=a^Tx"/>を考える
  - 情報の量を分散の大きさと捉え、<img src="https://latex.codecogs.com/gif.latex?s"/>の分散が最大となる<img src="https://latex.codecogs.com/gif.latex?a"/>を探索する<br/><img src="https://latex.codecogs.com/gif.latex?\text{Var}(s)=a^T\text{Var}(x)a"/>
- パラメータの推定
  - 制約条件付き最適化問題を解く<br/><img src="https://latex.codecogs.com/gif.latex?\text{argmax}_aa^T\text{Var}(x)a\:\:\:s.t.\,a^Ta=1"/>
  - ラグランジュ乗数法を用いる<br/><img src="https://latex.codecogs.com/gif.latex?L(a)=a^T\text{Var}(x)a-\lambda(a^Ta-1)"/>として、<br/><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(a)}{\partial&space;a}=2\text{Var}(x)a-2\lambda&space;a=0\:\Leftrightarrow\:\text{Var}(x)a=\lambda&space;a"/>
  - 元のデータの分散共分散行列の固有値と固有ベクトルが解
    - 分散共分散行列は正定値対称行列のため、固有値は0以上、固有ベクトルは直行ベクトルとなる
- 主成分
  - 固有値を大きい順に並べたとき、対応するk番目の固有ベクトル<img src="https://latex.codecogs.com/gif.latex?a_k"/>を第k主成分ベクトル、<img src="https://latex.codecogs.com/gif.latex?a_k"/>で射影した<img src="https://latex.codecogs.com/gif.latex?s_k=a_k^Tx"/>を第k主成分と呼ぶ
  - 第k主成分の分散は主成分に対応する固有値<img src="https://latex.codecogs.com/gif.latex?\lambda_k"/>
  - 寄与率：第k主成分の分散の全分散に対する割合(第k主成分が持つ情報量の割合)
### 実装演習
- 設定
  - 乳がん検査データを利用しロジスティック回帰モデルを作成
  - 主成分を利用し2次元空間上に次元圧縮
- 課題
  - 32次元のデータを2次元上に次元圧縮した際に、うまく判別できるかを確認
- コードのキャプション

<img src="https://user-images.githubusercontent.com/34636490/117989534-5e545f80-b377-11eb-98d7-45f75c6fa4c8.png" width=450>

- 結果
  - 30変数を使ったモデルの検証スコア（97%）よりは劣るが、第1・第2主成分の2変数でも検証スコア94%と高い精度が得られた
  - また、悪性（y=1）に対するRecallが94%と高く、悪性の見落としが少ない点もモデルの利用場面を考えると評価できる

## アルゴリズム
### k近傍法（kNN）
- 教師あり学習の一種
  - 分類問題のための手法
- 定式化
  - 新しいデータに最も近い学習データをk個取ってきて、それらが最も多く所属するクラスに識別
- 特徴
  - kを変化させると結果も変わる
  - kを大きくすると決定境界は滑らかになる
#### 実装演習
- 設定
  - 人工データを分類
- 課題
  - 人工データと分類結果をプロットする
- 結果のキャプション

<img src="https://user-images.githubusercontent.com/34636490/118144205-23216180-b447-11eb-861d-1a9747b9fc25.png" width=800>

- 考察
  - kを3から5、5から10と大きくするほど、決定境界が滑らかになる様子が確認できた

### k-means
- 教師なし学習の一種
  - クラスタリングの手法
  - 与えられたデータをk個のクラスタに分類する
- 定式化
  1. 各クラスタ中心の初期値を設定する
     - ランダムに選ぶ、既に選んだ中心からの距離をウェイトにして初期値同士が遠くなるようにする、などの方法がある
  2. 各データ点に対して、各クラスタ中心との距離を計算し、最も距離が近いクラスタを割り当てる
  3. 各クラスタの平均ベクトル（中心）を計算する
  4. 収束するまで2と3の処理を繰り返す
- 特徴
  - 中心の初期値を変えるとクラスタリング結果も変わりうる
  - kの値を変えるとクラスタリング結果も変わる
#### 実装演習
- 設定
  - ワインの分類
- コードのキャプション

<img src="https://user-images.githubusercontent.com/34636490/118249058-f1a8a480-b4df-11eb-9d18-c3f80c1a12ec.png" width=300>

- 考察
  - k=3としたk-meansの結果、ラベル0のクラスターがClass_0と、ラベル1のクラスターがClass_1と概ね対応している
  - 一方、ラベル2のクラスターは3つのクラスそれぞれが含まれる結果となり、Class_2のワインはアルコール度数などの特徴量がClass_0やClass_1のワインと近いものが多いと言える

## サポートベクターマシン
- 教師あり学習の一種
  - 2値分類するための手法（回帰にも利用可能）
  - 判別関数（決定境界）と最も近いデータ点（サポートベクトル）との距離（マージン）が最大となる判別関数を求める
  - 判別関数は<img src="https://latex.codecogs.com/gif.latex?y=\boldsymbol{w}^T\phi(\boldsymbol{x})+b" />の形をとる
- 定式化
  - ハードマージンSVM：全ての学習データを正確に分類することを制約条件としたSVM
    - 制約条件付きのマージン最大化問題<br/><img src="https://latex.codecogs.com/gif.latex?\text{max}_{\boldsymbol{w},b}\frac{1}{\|\boldsymbol{w}\|}\:\:\text{s.t.}\,t_i(\boldsymbol{w}^T\phi(\boldsymbol{x}_i)+b)\geq&space;1\,(i=1,\cdots,n)" /><br/><img src="https://latex.codecogs.com/gif.latex?\Rightarrow&space;\text{min}_{\boldsymbol{w},b}\frac{1}{2}\|\boldsymbol{w}\|^2\:\:\text{s.t.}\,t_i(\boldsymbol{w}^T\phi(\boldsymbol{x}_i)+b)\geq&space;1\,(i=1,\cdots,n)" />
    - 双対問題を考える方が簡単<br/><img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%26%5Ctext%7Bmax%7D_a+%5Csum_%7Bi%3D1%7D%5En+a_i+-+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5En+%5Csum_%7Bj%3D1%7D%5En+a_i+a_j+t_i+t_j+%5Cphi%28x_i%29%5ET+%5Cphi%28x_j%29+%5C%5C%0A%26%5Ctext%7Bs.t.%7D+%5C%3A+%5Cleft%5C%7B%0A%5Cbegin%7Balign%2A%7D+%5Csum_%7Bi%3D1%7D%5En+a_i+t_i+%26%3D+0+%5C%5C%0Aa_i+%26%5Cgeq+0%0A%5Cend%7Balign%2A%7D%0A%5Cright.+%2C%5C%3A%28i%3D1%2C%5Ccdots%2Cn%29%0A%5Cend%7Balign%2A%7D%0A" />
  - カーネルトリック
    - 特徴空間への射影<img src="https://latex.codecogs.com/gif.latex?\phi(x)" />は、内積<img src="https://latex.codecogs.com/gif.latex?\phi(x_i)^T\phi(x_j)" />の形でしか現れない
    - カーネル関数を<img src="https://latex.codecogs.com/gif.latex?k(x_i,x_j)=\phi(x_i)^T\phi(x_j)" />と仮定することで、無限次元の特徴空間も扱うことが容易になる
    - RBFカーネル（ガウシアンカーネル）：<img src="https://latex.codecogs.com/gif.latex?k(x_i,x_j)=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right)" />
  - ソフトマージンSVM：学習データの一部が誤分類されたり、誤分類されないもののマージン内部に入ることを許容するSVM
    - スラック変数<img src="https://latex.codecogs.com/gif.latex?\xi_i\geq&space;0"/>を導入し、マージンに関する制約を<img src="https://latex.codecogs.com/gif.latex?t_i(\boldsymbol{w}^T\phi(\boldsymbol{x}_i)+b)\geq&space;1-\xi_i\,(i=1,\cdots,n)" />と修正する
    - <img src="https://latex.codecogs.com/gif.latex?\xi_i>0"/>の学習データは誤分類となっているかマージン内部に入っているため、目的関数に罰則項を追加する<br/><img src="https://latex.codecogs.com/gif.latex?\text{min}_{\boldsymbol{w},b}\frac{1}{2}\|\boldsymbol{w}\|^2+C\sum_{i=1}^n\xi_i\:\:\text{s.t.}\,t_i(\boldsymbol{w}^T\phi(\boldsymbol{x}_i)+b)\geq&space;(1-\xi_i)\,(i=1,\cdots,n)" /><br/>Cはマージン最大化と誤分類等への罰則項最小化のトレードオフを決めるハイパーパラメータ
    - 双対問題を考えると、ハードマージンSVMと類似した最適化問題に帰着する<br/><img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%26%5Ctext%7Bmax%7D_a+%5Csum_%7Bi%3D1%7D%5En+a_i+-+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5En+%5Csum_%7Bj%3D1%7D%5En+a_i+a_j+t_i+t_j+%5Cphi%28x_i%29%5ET+%5Cphi%28x_j%29+%5C%5C%0A%26%5Ctext%7Bs.t.%7D+%5C%3A+%5Cleft%5C%7B%0A%5Cbegin%7Balign%2A%7D+%5Csum_%7Bi%3D1%7D%5En+a_i+t_i+%26%3D+0+%5C%5C%0Aa_i+%26%5Cgeq+0+%5C%5C%0Aa_i+%26%5Cleq+C%0A%5Cend%7Balign%2A%7D%0A%5Cright.+%2C%5C%3A%28i%3D1%2C%5Ccdots%2Cn%29%0A%5Cend%7Balign%2A%7D%0A" />

- 予測
  - 未知のデータに対して、<img src="https://latex.codecogs.com/gif.latex?y(\boldsymbol{x}_{\text{new}})=\boldsymbol{w}^T\phi(\boldsymbol{x}_{\text{new}})+b=\sum_{i=1}^na_it_ik(\boldsymbol{x}_{\text{new}},\boldsymbol{x}_i)+b" />の正負によって分類
  - ここで、サポートベクトルの集合をSとすると<img src="https://latex.codecogs.com/gif.latex?s\in&space;S" />では<img src="https://latex.codecogs.com/gif.latex?t_s(\boldsymbol{w}^T\phi(\boldsymbol{x}_s)+b)=0"/>が成り立つはずだが、全サポートベクトルについてbを計算した平均値を採用することが一般的<br/><img src="https://latex.codecogs.com/gif.latex?b=\frac{1}{|S|}\sum_{s\in&space;S}\left(\frac{1}{t_i}-\sum_{i=1}^na_it_ik(x_i,x_s)\right)"/>

#### 実装演習
- 設定
  - 人工データを分類
- 課題
  - 人工データと分類結果をプロットする
- 結果のキャプション
