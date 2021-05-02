# 応用数学レポート
1. [線形代数](#線形代数)
2. <a href="prob">確率・統計</a>
3. <a href="entropy">情報量</a>

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

## 第二章　確率・統計
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
- ベルヌーイ分布
- カテゴリカル分布
- 二項分布
- ガウス分布

## 第三章　情報理論
