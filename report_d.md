# 深層学習（後半）レポート
1. [day3 Section1:再帰型ニューラルネットワークの概念](#再帰型ニューラルネットワークの概念)
2. [day3 Section2:LSTM](#LSTM)
3. [day3 Section3:GRU](#GRU)
4. [day3 Section4:双方向RNN](#双方向RNN)
5. [day3 Section5:Seq2Seq](#Seq2Seq)
6. [day3 Section6:Word2Vec](#Word2Vec)
7. [day3 Section7:Attention Mechanism](#Attention機構)
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
  - RNNの構造イメージ（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119506206-a8dad080-bda8-11eb-8fff-8aff117acc71.png" width=600/>
- BPTT
  - パラメータを更新する際に、時間方向にも遡ることが必要<br/>
    - <img src="https://latex.codecogs.com/gif.latex?W_{(in)}"/>及び<img src="https://latex.codecogs.com/gif.latex?W"/>の更新部分（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119507371-c0668900-bda9-11eb-8983-da190c00e22c.png" width=400 />

### 実装演習結果
### 考察

## LSTM
### 要点のまとめ
- RNNの課題
  - 時系列を遡れば遡るほど勾配が消失していくため、長い時系列の学習が困難
  - ネットワークの構造自体を変えて解決したものがLSTM
- LSTMモデル
  - 全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119625990-babe8100-be45-11eb-8b74-9ac643f2f573.png" width=800/>
  - CEC
    - 勾配消失および勾配爆発の解決方法として、勾配が常に1となるユニット
    - ニューラルネットワークの学習特性が無い
  - 入力ゲート・出力ゲート
    - それぞれのゲートへの入力値の重みを可変とすることで、CECの課題を解決
  - 忘却ゲート
    - 過去の情報が要らなくなった場合、CECに情報を忘却させるための機能
  - 覗き穴結合
    - CECに保存されている過去の情報を、任意のタイミングで他のノードに伝播させるための構造

### 実装演習結果
### 考察

## GRU
### 要点のまとめ
- LSTMの課題
  - パラメータ数が多く、計算負荷が高くなる問題
- GRU
  - パラメータを大幅に削減しながら、精度は同程度となる構造
  - 計算負荷が低い
- GRUの全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119627503-3a991b00-be47-11eb-8572-6184d4d7f921.png" width=800/>

### 実装演習結果
### 考察

## 双方向RNN
### 要点のまとめ
- 過去の情報だけでなく、未来の情報を加味することで、精度を向上させるためのモデル
  - 例）文章の推敲、機械翻訳
- 双方向RNNのイメージ（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119627792-7a600280-be47-11eb-8c13-e4954314ee28.png" width=400/>
  - 時系列順（古いデータから新しいデータへ）だけでなく、新しいデータから古いデータを入力して再帰的に状態を更新するユニットを持つ
### 実装演習結果
### 考察

## Seq2Seq
### 要点のまとめ
- Seq2Seqモデル
  - Encoder-Decoderモデルの一種
  - 機械対話や機械翻訳などに用いられる
- Seq2Seqの全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119629268-d0817580-be48-11eb-8dc5-edf5e8e81786.png" width=400/>
  - Encoder RNN
    - 入力テキストデータを単語等のトークンに区切り、Embeddingにより分散表現ベクトルを用意
    - 分散表現ベクトルをエンコーダRNNに入力し、hidden stateを計算
    - 最後のhidden stateはthought vectorと呼ばれ、入力した文の文脈を表す
  - Decoder RNN
    - エンコーダRNNのthought vectorを入力とし、出力トークンの生成確率を計算
    - 選ばれたトークンをEmbeddingして次の入力とする手順を繰り返す
- HRED
  - Seq2Seq＋Context RNN
    - Context RNN: エンコーダによる各文章の系列をまとめて、これまでの会話コンテキスト全体を表すベクトルに変換する構造
  - 過去複数個の会話から次の発話を生成する
  - Seq2seqでは会話の文脈無視で応答がなされるが、HREDでは前の会話の流れに即して応答するため、より人間らしい文章が生成される
- VHRED
  - HREDの課題を、VAEの潜在変数の概念を追加することで解決した構造
- VAE
  - オートエンコーダ
    - 教師なし学習の一種
    - 入力データから潜在変数zに変換するニューラルネットワークがEncoder
    - 潜在変数zをインプットとして入力データを復元するニューラルネットワークがDecoder
    - zの次元が入力データより小さい場合、次元削減とみなすことができる
  - VAE
    - 潜在変数zが標準正規分布に従うことを仮定したオートエンコーダ
### 実装演習結果
### 考察

## Word2Vec
### 要点のまとめ
- RNNには固定長形式で入力を渡す必要がある
- Word2Vec
  - 単語を固定長ベクトルで表す分散表現の学習を、現実的な計算速度とメモリ量で実現可能
  - ボキャブラリ数×単語ベクトルの次元数の重みパラメータを学習

### 実装演習結果
### 考察

## Attention機構
### 要点のまとめ
- Seq2Seqの課題
  - 長い文章への対応が難しい
  - 文章が長くなるほど、内部表現の次元も大きくなっていく仕組みが必要
- Attention Mechanism
  - 「入力と出力のどの単語が関連しているのか」の関連度を学習する仕組み
### 実装演習結果
### 考察