# 深層学習（後半）レポート
1. [day3 Section1:再帰型ニューラルネットワークの概念](#再帰型ニューラルネットワークの概念)
2. [day3 Section2:LSTM](#LSTM)
3. [day3 Section3:GRU](#GRU)
4. [day3 Section4:双方向RNN](#双方向RNN)
5. [day3 Section5:Seq2Seq](#Seq2Seq)
6. [day3 Section6:Word2Vec](#Word2Vec)
7. [day3 Section7:Attension Mechanism](#Attension機構)
8. [day4 Section1:強化学習](#強化学習)
9. [day4 Section2:AlphaGo](#AlphaGo)
10. [day4 Section3:軽量化・高速化技術](#軽量化・高速化技術)
11. [day4 Section4:応用モデル](#応用モデル)
12. [day4 Section5:Transformer](#Transformer)
13. [day4 Section6:物体検知・セグメンテーション](#物体検知・セグメンテーション)


## 再帰型ニューラルネットワークの概念
### 要点のまとめ
- RNNとは時系列データに対応可能なニューラルネットワーク
  - 音声データ
  - テキストデータ
- 時系列情報を扱うために、過去の時間の状態を保持し、そこから次の時間の状態を再帰的に求める構造<br/>
  - RNNの構造イメージ（講義資料より）
<img src="https://user-images.githubusercontent.com/34636490/119506206-a8dad080-bda8-11eb-8fff-8aff117acc71.png" width=600/>

- BPTT
  - パラメータを更新する際に、時間方向にも遡ることが必要<br/>
  - W_inの更新部分（講義資料より）
<img src="https://user-images.githubusercontent.com/34636490/119507371-c0668900-bda9-11eb-8983-da190c00e22c.png" width=600 />

### 実装演習結果
### 考察

## LSTM

## GRU

## 双方向RNN
