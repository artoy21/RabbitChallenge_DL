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
  - 説明変数を<img src="https://latex.codecogs.com/gif.latex?x=(x_1,x_2,\cdots,x_m)^T\in\mathbb{R}^m" />、目的変数を<img src="https://latex.codecogs.com/gif.latex?y\in\mathbb{R}" />
  - パラメータを<img src="https://latex.codecogs.com/gif.latex?w_0\in\mathbb{R},\:w=(w_1,w_2,\cdots,w_m)^T\in\mathbb{R}^m" />
  - として、<img src="https://latex.codecogs.com/gif.latex?y=w_0+w^Tx" />
- パラメータの推定
  - 最小二乗法を利用
  - <img src="https://latex.codecogs.com/gif.latex?\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^n\left(\hat{y}_i-y_i\right)^2" />
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
## ロジスティック回帰モデル
## 主成分分析
## k近傍法
## k-means
## サポートベクターマシン
