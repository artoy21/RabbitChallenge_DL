# 機械学習レポート
1. [線形回帰モデル](#線形回帰モデル)
2. [非線形回帰モデル](#非線形回帰モデル)
3. [ロジスティック回帰モデル](#ロジスティック回帰モデル)
4. [主成分分析](#主成分分析)
5. [k近傍法（kNN）](#k近傍法)
6. [k-means](#k-means)
7. [サポートベクターマシン（SVM）](#サポートベクターマシン)

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
  - 部屋数が4で犯罪率が0.3の物件はいくらになるか？
- コードのキャプション
- 結果
  - 26678.8ドルと予測
  - 

## ロジスティック回帰モデル
## 主成分分析
## k近傍法
## k-means
## サポートベクターマシン
