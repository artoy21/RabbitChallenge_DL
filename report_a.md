# 応用数学レポート
1. [線形代数](#線形代数)
2. [確率・統計](#確率統計)
3. [情報理論](#情報理論)

## 線形代数
- スカラー、ベクトル、行列
  - スカラーの一次元配列がベクトル
  - スカラーの二次元配列が行列
  - 連立方程式は行列とベクトルで表すことができる
- 単位行列と逆行列
  - 単位行列は対角成分が全て1、それ以外の要素が全て0の行列
  - 逆行列は元の正方行列との積が単位行列になる行列<br/>
単位行列を<img src="https://latex.codecogs.com/gif.latex?I" />、
元の行列を<img src="https://latex.codecogs.com/gif.latex?A" />、
元の行列の逆行列を<img src="https://latex.codecogs.com/gif.latex?A^{-1}" />とすると、
<img src="https://latex.codecogs.com/gif.latex?AA^{-1}=A^{-1}A=I" />を満たす。
  - 行列式が0の行列は逆行列を持たない
  - Pythonで逆行列を取得する関数
```
import numpy as np
# Aは元の行列
np.linalg.inv(A)
```
- 固有値と固有ベクトル
  - 正方行列<img src="https://latex.codecogs.com/gif.latex?A" />に対して、<img src="https://latex.codecogs.com/gif.latex?Av={\lambda}v" />となる時、
ベクトル<img src="https://latex.codecogs.com/gif.latex?v" />は<img src="https://latex.codecogs.com/gif.latex?A" />の固有ベクトル、
スカラー<img src="https://latex.codecogs.com/gif.latex?\lambda" />は<img src="https://latex.codecogs.com/gif.latex?A" />の固有値
- 固有値分解
  - 正方行列<img src="https://latex.codecogs.com/gif.latex?A" />が固有値<img src="https://latex.codecogs.com/gif.latex?\lambda_1,\lambda_2,\lambda_3,\cdots" />
と固有ベクトル<img src="https://latex.codecogs.com/gif.latex?v_1,v_2,v_3,\cdots" />を持つ時、<br/>
対角行列<img src="https://latex.codecogs.com/gif.latex?\Lambda=\left(\begin{matrix}\lambda_1&&&\\&\lambda_2&&\\&&\lambda_3&\\&&&\ddots\end{matrix}\right)" />と、<br/>
直行行列<img src="https://latex.codecogs.com/gif.latex?V=(v_1\:v_2\:v_3\:\cdots)" />を用いて、<br/>
<img src="https://latex.codecogs.com/gif.latex?A=V\Lambda&space;V^{-1}" />と変形できる
  - この変形によって、行列<img src="https://latex.codecogs.com/gif.latex?A" />の累乗は<br/>
<img src="https://latex.codecogs.com/gif.latex?A^n=V\left(\begin{matrix}\lambda_1^n&&&\\&\lambda_2^n&&\\&&\lambda_3^n&\\&&&\ddots\end{matrix}\right)V^{-1}" /><br/>
として計算できる
  - Pythonで固有値と固有ベクトルを取得する関数
```
import numpy as np
# Aは元の行列
eigenvalue, eigenvector = np.linalg.eig(A)
```
- 特異値分解
  - 正方行列以外の行列に対しても、固有値分解と似た分解ができる
  - <img src="https://latex.codecogs.com/gif.latex?MM^T" />を固有値分解した固有ベクトルを並べた行列を<img src="https://latex.codecogs.com/gif.latex?U" />、固有値の2乗を対角成分とする対角行列を<img src="https://latex.codecogs.com/gif.latex?SS^T" />、<br>
<img src="https://latex.codecogs.com/gif.latex?M^TM" />を固有値分解した固有ベクトルを並べた行列を<img src="https://latex.codecogs.com/gif.latex?V" />、固有値の2乗を対角成分とする対角行列を<img src="https://latex.codecogs.com/gif.latex?S^TS" />として、<br><img src="https://latex.codecogs.com/gif.latex?M=USV^{-1}" />
  - Pythonで特異値と特異ベクトルを取得する関数
```
import numpy as np
# Mは元の行列
U, sv, VT = np.linalg.svd(M)

# 元の行列を復元できるか確認
S = np.zeros_like(M, dtype=float)
for i in range(len(sv)):
  S[i,i] = sv[i]
np.allclose(M, U.dot(S).dot(VT))
```

## 確率統計
- 条件付き確率
  - ある事象<img src="https://latex.codecogs.com/gif.latex?X=x" />を条件とした下で、事象<img src="https://latex.codecogs.com/gif.latex?Y=y" />となる確率<br/>
<img src="https://latex.codecogs.com/gif.latex?P(Y=y\,|\,X=x)=\frac{P(Y=y,\,X=x)}{P(X=x)}" /><br/>
（ここで、<img src="https://latex.codecogs.com/gif.latex?P(Y=y,\,X=x)" />は、事象<img src="https://latex.codecogs.com/gif.latex?X=x" />と事象<img src="https://latex.codecogs.com/gif.latex?Y=y" />が同時に発生する確率）
- ベイズ則
  - 事象<img src="https://latex.codecogs.com/gif.latex?X=x" />と事象<img src="https://latex.codecogs.com/gif.latex?Y=y" />に対して、<br/>
<img src="https://latex.codecogs.com/gif.latex?P(X=x\,|\,Y=y)P(Y=y)=P(Y=y\,|\,X=x)P(X=x)" />が成り立つ
- 期待値
  - 確率変数<img src="https://latex.codecogs.com/gif.latex?X" />が離散値を取る場合、<br/><img src="https://latex.codecogs.com/gif.latex?E[f(X)]=\sum_{k=1}^nf(X=x_k)P(X=x_k)" />
  - 確率変数<img src="https://latex.codecogs.com/gif.latex?X" />が連続値を取る場合、<br/><img src="https://latex.codecogs.com/gif.latex?E[f(X)]=\int&space;f(X=x)P(X=x)dx" />
- 分散と共分散、標準偏差
  - 分散はデータの散らばり具合<br/><img src="https://latex.codecogs.com/gif.latex?\text{Var}\left(f(X)\right)=E\left[\left(f(X)-E[f(X)]\right)^2\right]=E\left[f(X)^2\right]-E[f(X)]^2" />
  - 共分散は2つのデータ系列の傾向の違い<br/><img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Ctext%7BCov%7D+%5Cleft%28+f%28X%29%2C+%5C%2C+g%28Y%29+%5Cright%29+%26%3D+E+%5Cleft%5B+%5Cleft%28+f%28X%29+-+E%5Bf%28X%29%5D+%5Cright%29+%5Cleft%28+g%28Y%29+-+E%5Bg%28Y%29%5D+%5Cright%29+%5Cright%5D+%5C%5C%0A%26%3D+E+%5Cleft%5B+f%28X%29+g%28Y%29+%5Cright%5D+-+E%5Bf%28X%29%5DE%5Bg%28Y%29%5D%0A%5Cend%7Balign%2A%7D%0A" />
  - 標準偏差は分散の平方根<img src="https://latex.codecogs.com/gif.latex?\sigma(f(X))=\sqrt{\text{Var}(f(X))}" />で、<img src="https://latex.codecogs.com/gif.latex?f(X)" />と単位が同じ
- ベルヌーイ分布
  - 確率<img src="https://latex.codecogs.com/gif.latex?\mu" />で表（<img src="https://latex.codecogs.com/gif.latex?X=1" />）が、確率<img src="https://latex.codecogs.com/gif.latex?1-\mu" />で裏（<img src="https://latex.codecogs.com/gif.latex?X=0" />）が出るコインを投げた時の、<img src="https://latex.codecogs.com/gif.latex?X" />の確率分布<br/><img src="https://latex.codecogs.com/gif.latex?P(X=x|\mu)=\mu^x(1-\mu)^{1-x}" />
- 二項分布
  - 同様のコインを<img src="https://latex.codecogs.com/gif.latex?n" />回投げた時に、表が出る回数を確率変数<img src="https://latex.codecogs.com/gif.latex?X" />とする確率分布<br/><img src="https://latex.codecogs.com/gif.latex?P(X=x|\mu,\,n)=\frac{n!}{x!(n-x)!}\mu^x(1-\mu)^{n-x}" />
- ガウス分布
  - 左右対称の連続値を取る確率分布（正規分布）<br/><img src="https://latex.codecogs.com/gif.latex?p(x;\mu,\sigma^2)=\sqrt{\frac{1}{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)" />

## 情報理論
- 自己情報量
  - 確率<img src="https://latex.codecogs.com/gif.latex?P(x)" />の事象<img src="https://latex.codecogs.com/gif.latex?x" />が発生した時の珍しさ<br/><img src="https://latex.codecogs.com/gif.latex?I(x)=-\log_2P(x)" />
  - 確率1/2で起こる事象の自己情報量は1bit
- エントロピー
  - 自己情報量の期待値<br/><img src="https://latex.codecogs.com/gif.latex?H(x)=E[I(x)]=E[-\log_2P(x)]=-\sum\left(P(x)\log_2P(x)\right)" />
  - 確率分布<img src="https://latex.codecogs.com/gif.latex?P(x)" />の不確実性の尺度
  - 表と裏の出る確率が共に1/2のコイン投げのエントロピーは、<br/><img src="https://latex.codecogs.com/gif.latex?H(x)=-\left(\frac{1}{2}\log_2\frac{1}{2}+\frac{1}{2}\log_2\frac{1}{2}\right)=1" />
  - 表の出る確率が1、裏の出る確率が0のコイン投げ（不確実性が無い場合）のエントロピーは、<br/><img src="https://latex.codecogs.com/gif.latex?H(x)=-\left(1\log_21+\lim_{x\to+0}x\log_2x\right)=-\left(0+\lim_{x\to+0}\frac{-x}{\log2}\right)=0" />
- カルバック・ライブラー　ダイバージェンス（KLダイバージェンス）
  - 異なる確率分布間の距離みたいな概念<br/><img src="https://latex.codecogs.com/gif.latex?D_{KL}(P||Q)=E^P\left[\log_2\frac{P(x)}{Q(x)}\right]=\sum&space;P(x)\log_2\frac{P(x)}{Q(x)}" />
- 交差エントロピー
  - KLダイバージェンスの一部分を取り出したもの<br/><img src="https://latex.codecogs.com/gif.latex?H(P,Q)=E^P\left[-\log_2Q(x)\right]=-\sum&space;P(x)\log_2Q(x)=H(P)+D_{KL}(P||Q)" />
  - <img src="https://latex.codecogs.com/gif.latex?P(x)" />が所与の場合、交差エントロピーの最小化はKLダイバージェンスの最小化（<img src="https://latex.codecogs.com/gif.latex?Q(x)" />を<img src="https://latex.codecogs.com/gif.latex?P(x)" />に**近付ける**こと）と同義
