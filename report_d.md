# 深層学習（後半）レポート
1. [day3 Section1:再帰型ニューラルネットワークの概念](#再帰型ニューラルネットワークの概念)
2. [day3 Section2:LSTM](#LSTM)
3. [day3 Section3:GRU](#GRU)
4. [day3 Section4:双方向RNN](#双方向RNN)
5. [day3 Section5:Seq2Seq](#Seq2Seq)
6. [day3 Section6:Word2Vec](#Word2Vec)
7. [day3 Section7:Attention Mechanism](#AttentionMechanism)
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
- バイナリ加算のRNN予測モデルにおいて、活性化関数をシグモイド関数からReLU関数に変えて比較
<img src="https://user-images.githubusercontent.com/34636490/119667929-ae501d80-be71-11eb-82fd-4b3f9dd6aaff.png" width=600 />

### 考察
- シグモイド関数を活性化関数とした場合、学習が進んでバイナリ加算をほぼ再現できるようになった
- 活性化関数にReLUを用いると、勾配消失問題が起きたとみられ、学習が進まなかったとみられる

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
- 前の5単語から次の単語を予測する実装演習において、LSTMセルを使用<br/><img src="https://user-images.githubusercontent.com/34636490/120073048-b5756680-c0d1-11eb-9973-ddf2ead87722.png" width=300 />
- 100エポックで学習<br/><img src="https://user-images.githubusercontent.com/34636490/120073130-05ecc400-c0d2-11eb-9411-bd6b53ce466e.png" width=300/>
- テストデータに対して、```Some of them looks like 'look'```との予測
### 考察
- 学習が進むにつれ、訓練データに対する正解率が上がる一方、検証データに対する正解率が下がり、過学習に陥ったとみられる
- 全てのコーパスファイルを読み込むとGoogle Colabのメモリ不足で学習できないため、コーパスファイルを100に絞ったことの影響と考えられる

## GRU
### 要点のまとめ
- LSTMの課題
  - パラメータ数が多く、計算負荷が高くなる問題
- GRU
  - パラメータを大幅に削減しながら、精度は同程度となる構造
  - 計算負荷が低い
- GRUの全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119627503-3a991b00-be47-11eb-8572-6184d4d7f921.png" width=800/>

### 実装演習結果
- 前の5単語から次の単語を予測する実装演習において、GRUセルを使用<br/><img src="https://user-images.githubusercontent.com/34636490/120073397-0df93380-c0d3-11eb-858c-d6d232951530.png" width=300 />
- LSTM同様に100エポックで学習<br/><img src="https://user-images.githubusercontent.com/34636490/120073503-85c75e00-c0d3-11eb-897e-c576a6714b5b.png" width=600/>
- テストデータに対して、```Some of them looks like 'good'```との予測
### 考察
- GRUを用いた場合でも、LSTMと同様に過学習の傾向がみられた
- ただし、テストデータに対しては、一見文章として意味の通る単語（good）を予測しており、LSTMより文法構造を学習しているのかもしれない

## 双方向RNN
### 要点のまとめ
- 過去の情報だけでなく、未来の情報を加味することで、精度を向上させるためのモデル
  - 例）文章の推敲、機械翻訳
- 双方向RNNのイメージ（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/119627792-7a600280-be47-11eb-8c13-e4954314ee28.png" width=400/>
  - 時系列順（古いデータから新しいデータへ）だけでなく、新しいデータから古いデータを入力して再帰的に状態を更新するユニットを持つ
### 実装演習結果
- 前の5単語から次の単語を予測する実装演習において、双方向RNN（15ユニットのGRUセル×2）を使用<br/><img src="https://user-images.githubusercontent.com/34636490/120073929-6e897000-c0d5-11eb-896b-2432b8a243e3.png" width=400 />
- LSTMとGRU同様に100エポックで学習<br/><img src="https://user-images.githubusercontent.com/34636490/120074060-06875980-c0d6-11eb-8b60-d97293547dfe.png" width=900/>
- テストデータに対して、```Some of them looks like 'money'```との予測
### 考察
- 双方向RNNを用いた場合でも、やはり過学習の傾向がみられた
  - ユニット数が2倍のGRUセル1つの場合とほぼ同じ正解率であり、今回の課題である5単語から次の単語を予測する問題において逆方向の単語列情報を使用するメリットは小さいと考えられる
- Google Colabのメモリ不足の問題で、学習に使用するコーパスファイル数を100に絞ったことの影響も大きいとみられる

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
- EncoderとDecoder共にGRUとしたSeq2Seqモデルによる機械翻訳
  - Embeddingの次元、GRUのユニット数はすべて256
  - 40,000サンプルの文章対を用いて10エポックの学習を実行
<img src="https://user-images.githubusercontent.com/34636490/120207540-8abd1680-c267-11eb-958f-3cf7761d5ab9.png" width=400 />

- テストデータのBLUE=17.34
- テストデータにおける翻訳結果
  - ```show your own business . -> 自分 の 仕事 を を し 。 。```
  - ```he lived a hard life . -> 彼 は 人生 い 生活 に 送 っ た 。```
  - ```i can 't swim at all . > 私 は は 泳げ な い 。```
### 考察
- テストデータにおける翻訳結果を見ると、それなりに意味の通じる訳もあるが、不完全な文章が多い
- 検証データのBLEUは10エポックで既に頭打ち傾向が見られるため、これ以上学習を進めても汎化性能はあまり高まらないかもしれない
- 翻訳の質を高めるには、GRUのユニット数を増やしてモデルの表現力を高めるとともに、過学習に陥らないよう学習データを増やす必要があると考えられる

## Word2Vec
### 要点のまとめ
- RNNには固定長形式で入力を渡す必要がある
- Word2Vec
  - 単語を固定長ベクトルで表す分散表現の学習を、現実的な計算速度とメモリ量で実現可能
  - ボキャブラリ数×単語ベクトルの次元数の重みパラメータを学習
  - 代表的な手法としては、CBoW(continuous bag-of-words)とskip-gramの2つがある

### 実装演習結果
- Transformer演習用の機械翻訳学習データから、日本語の文章50,000サンプルを使用
  - 単語の語彙数8,778を、Word2Vecを用いて32次元のベクトル表現に埋め込み
- CBoW及びskip-gramで学習したモデルをもとに、「家族」に近い単語、「父－男＋女」に近い単語をスコアの高い順に5つ出力
1. CBoWを用いた結果<br/>
<img src="https://user-images.githubusercontent.com/34636490/120907865-26071f00-c6a0-11eb-9112-3cdcceba9ab7.png" width=150 />

2. skip-gramを用いた結果<br/>
<img src="https://user-images.githubusercontent.com/34636490/120907853-f3f5bd00-c69f-11eb-86e1-ee256e57df1b.png" width=150/>

### 考察
- 「家族」に近い単語としては、CBoWでは「親友」以外は親族を表す単語に高いスコアが与えられた
  - 一方、skip-gramでは「親友」や「友人」
- 「父－男＋女」に近い単語として、CBoWとskip-gramのどちらも正解と言える「母」を最も高いスコアで出力
  - 父親の男女を逆にした概念が母親であることをベクトル表現として学習できているとみられる

## AttentionMechanism
### 要点のまとめ
- Seq2Seqの課題
  - 長い文章への対応が難しい
  - 文章が長くなるほど、内部表現の次元も大きくなっていく仕組みが必要
- Attention Mechanism
  - 「入力と出力のどの単語が関連しているのか」の関連度を学習する仕組み
### 実装演習結果
### 考察

## 強化学習
### 要点のまとめ

## AlphaGo
### 要点のまとめ

## 軽量化・高速化技術
### 要点のまとめ

## 応用モデル
### 要点のまとめ

## Transformer
### 要点のまとめ

## 物体検知・セグメンテーション
### 要点のまとめ
