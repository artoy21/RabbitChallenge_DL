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
10. [day4 Section3:軽量化・高速化技術](#軽量化-高速化技術)
11. [day4 Section4:応用モデル](#応用モデル)
12. [day4 Section5:Transformer](#Transformer)
13. [day4 Section6:物体検知・セグメンテーション](#物体検知-セグメンテーション)


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
- 「家族」に近い単語としては、CBoWでは「親友」以外は実際の家族を表す単語に高いスコアが与えられた
  - 一方、skip-gramでは「友達」「親友」「友人」という、近しい間柄の人間関係という点で家族に意味が似た単語に高いスコアが与えられた
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
- Transformer演習用の機械翻訳学習データを使用
- エンコーダ及びデコーダにLSTM（ユニット数256）を用いたSeq2SeqモデルにAttentionを追加<br/>
<img src="https://user-images.githubusercontent.com/34636490/120928804-3d3a2100-c721-11eb-8cd1-f64d2f031970.png" width=700/>

- 翻訳例におけるAttentionの分布を確認
  - 入力文: ```<s> i study at school . </s>```
  - 出力文: ```<s> 私 は 学校 で 勉強 し ま す 。 </s>```<br/>
<img src="https://user-images.githubusercontent.com/34636490/120928587-2e06a380-c720-11eb-913a-3c25841f2cf7.png" width=300/>


### 考察
- 翻訳例におけるAttentionの分布から、「学校」「で」とデコーダから出力する際に「school」「at school」のスコアが高くなっていたり、「勉強」「し」を出力する際に「study」「study at」のスコアが高くなっているなど、短い文章ながら次の出力において重要となる入力単語が何かをうまく学習できていると考えられる

## 強化学習
### 要点のまとめ
- 機械学習の一分野
  - 行動の結果として与えられる報酬をもとに、行動を決定する原理を改善していく仕組み
  - 与えられた環境の中で、エージェントが方策（policy <img src="https://latex.codecogs.com/gif.latex?\pi"/>）に基づいて行動を取る
  - エージェントの行動によって環境中の状態（state）が更新される
  - 新しい状態に応じて報酬（reward）が得られる
  - 報酬の累積である価値（value）を最大化することを目的に、最適な方策を探索する
- 強化学習の応用例：マーケティング
  - 環境：会社の販売促進部
  - エージェント：キャンペーンメールを送信する顧客を決めるソフトウェア
  - 行動：顧客ごとに送信／非送信の2種類がある
  - 報酬：キャンペーンの費用という負の報酬と、キャンペーンによる売上増加という正の報酬
- 探索と利用のトレードオフ
  - 探索が足りない状態
    - 過去のデータで最良とされる行動のみを常に取り続ける場合、他にもっと良い行動を見つけることはできない
  - 利用が足りない状態
    - 未知の行動のみを常に取り続ける場合、過去の経験がいかせない
- 価値関数
  - 状態価値関数
    - ある状態の価値に注目
  - 行動価値関数
    - 状態と価値を組み合わせた価値に注目
- 方策関数
  - ある状態でどのような行動を採るのかの確率を与える関数
- 方策勾配法
  - 方策パラメータ<img src="https://latex.codecogs.com/gif.latex?\theta"/>をモデル化して最適化する手法<br/><img src="https://latex.codecogs.com/gif.latex?\theta^{(t+1)}=\theta^{(t)}+\varepsilon\nabla&space;J(\theta)"/>
  - 方策の良さ<img src="https://latex.codecogs.com/gif.latex?J"/>は平均報酬や割引報酬合計として定義
  - 行動価値関数<img src="https://latex.codecogs.com/gif.latex?Q(s,a)"/>を定義すると、方策勾配定理が成り立つ<br/><img src="https://latex.codecogs.com/gif.latex?\nabla_{\theta}J(\theta)=E_{\pi_{\theta}}\left[(\nabla_{\theta}\log\pi_{\theta}(a\mid&space;s)Q^{\pi}(s,a))\right]"/>

## AlphaGo
### 要点のまとめ
- AlphatGo (Lee)
  - 学習法
    1. 教師あり学習によるRollOutPolicyとPolicyNetの学習
       - 囲碁対局サイトの棋譜データから3000万局面分の教師データを用意し、教師と同じ着手を予測できるよう学習 
    2. 強化学習によるPolicyNetの学習
       - PolicyNet同士の対局シミュレーションを行い、方策勾配法で学習
    3. 強化学習によるValueNetの学習
       - PolicyNetを使用して対局シミュレーションを行い、勝敗を教師として学習
  - PolicyNet
    - 全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/121106065-3b1ab400-c840-11eb-90cf-3ce8d67c8865.png" width=500/>
    - 入力データは現在の盤面の特徴を表す19ｘ19ｘ48チャネル
    - 出力は```softmax```関数によって0～1の範囲になり、各（19ｘ19ある）マスに対する着手確率を表す
  - ValueNet
    - 全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/121106109-571e5580-c840-11eb-9838-07aa0ee0ac0d.png" width=500/>
    - 入力データはPolicyNetと同じ特徴に加え、現在の手番を表す1チャネル
    - 出力は```tanh```関数によって-1～+1の範囲になり、現局面の勝率を表す
  - RollOutPolicy
    - 各19ｘ19マスの着手予想確率を高速で計算するための線形関数

  - モンテカルロ木探索
    - 盤面の評価値に頼らず、末端評価値（勝敗）を使う探索法
    - 現局面から末端局面までPlayOutと呼ばれるランダムシミュレーションを多数回行い、その勝敗を集計して着手の優劣を決定
- AlphaGoZero
  - AlphaGo (Lee)との違い
    - 教師あり学習を一切行わず、強化学習のみ
    - 特徴入力は石の配置のみで、ヒューリスティックな特徴量を使わない
    - PolicyNetとValueNetを１つのネットワークに統合
    - ResidualNetを導入
  - 学習法
    1. 自己対局による教師データの作成
       - 現状のネットワークでモンテカルロ木探索を用いて自己対局を行う
       - 30手までランダムで打ち、そこから探索を行い勝敗を決定する
       - 自己対局中の各局面での着手選択確率分布と勝敗を記録する
       - 教師データの形は（局面、着手選択確率分布、勝敗）
    2. 学習
       - 自己対局で作成した教師データを使い学習
       - Policy部分の教師に着手選択確率分布、Value部分の教師に勝敗を用いる
       - 損失関数はPolicy部分はクロスエントロピー、Value部分は平均二乗誤差
    3. ネットワークの更新
       - 現状のネットワークと学習後のネットワークとで対局テストを行う
       - 学習後のネットワークの勝率が高かった場合、学習後のネットワークを現状のネットワークとする
    
  - PolicyValueNet
    - 全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/121192426-e743b580-c8a7-11eb-9556-3590dfd37153.png" width=600/>
    - 入力データは現在の盤面の特徴を表す19ｘ19ｘ17チャネル
    - 出力は各（19ｘ19ある）マスに対する着手確率を表すPolicy部分と、現局面の勝率を表すValue部分
  - ResidualNet
    - ネットワークにショートカット構造を追加して、勾配爆発や勾配消失を抑える効果を狙ったもの
    - 数の違うNetworkのアンサンブル効果が得られているという説もある
    - 基本形（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/121193304-b0ba6a80-c8a8-11eb-8642-20e8c4cc0402.png" width=500/>
    - 派生形
      - Bottleneck<br/>1x1カーネルの畳み込みを利用し、1層目で次元削減を行って3層目で次元を復元する3層構造にしたもの      
      - PreActivation<br/>ResidualBlockの並びについて、BatchNorm＋ReLUとConvolutionの順序を逆にしたもの
      - WideResNet<br/>Convolutionのフィルタ数をk倍にしたもの。段階的に幅を増やしていくのが一般的
      - PyramidNet<br/>WideResNetのように段階的にではなく、各層でフィルタ数を増やしていくもの
## 軽量化-高速化技術
### 要点のまとめ
- 分散深層学習
  - 深層学習は多くのデータを使用したり、パラメータ調整のために多くの時間を使用したりするため、高速な計算が求められる
  - データ並列化
    - 親モデルを各ワーカーに子モデルとしてコピー
    - データを分割し、各ワーカーごとに計算させる
    - 各モデルのパラメータの合わせ方で、同期型と非同期型に分かれる
    - 同期型
      - 全ワーカーの勾配が出たところで勾配の平均を計算し、親モデルのパラメータを更新する
      - 非同期型より精度が良いことが多いので、主流となっている
    - 非同期型
      - 各ワーカーはお互いの計算を待たず、学習が終わった子モデルはパラメータサーバにPushされる
      - 新たに学習を始める時は、パラメータサーバからPopしたモデルに対して学習していく
      - 最新のモデルのパラメータを利用できないので、学習が不安定になりやすい（Stale Gradient Problem）

  - モデル並列化
    - 親モデルを各ワーカーに分割し、それぞれのモデルを学習させる
    - 全てのデータで学習が終わった後で、一つのモデルに復元
    - モデルのパラメータ数が多いほど、スピードアップの効率も向上する傾向
    - モデルが大きい時はモデル並列化を、データが大きい時はデータ並列化をすると良い
  - GPUによる高速化
    - CPU
      - 高性能なコアが少数
      - 複雑で連続的な処理が得意
    - GPU
      - 比較的低性能なコアが多数
      - 簡単な並列処理が得意
      - ニューラルネットの学習は単純な行列演算が多いので、高速化が可能
      - GPGPU: 元々の使用目的であるグラフィック以外の用途で使用されるGPUの総称
- モデルの軽量化
  - モデルの精度を維持しつつパラメータや演算回数を低減する手法の総称
    - モバイル端末やIoT機器において有用な手法
    - 性能（主に計算速度と搭載されているメモリ）が劣る計算環境とと相性が良い手法
  - 量子化
    - 通常のパラメータの64bit浮動小数点を32bitなど下位の精度に落とすことで、メモリと演算処理を削減
    - 利点
      - 計算の高速化
      - 省メモリ化
    - 欠点
      - 精度の低下<br/>（実際の問題では倍精度を単精度にしてもほぼ精度は変わらない） 
  - 蒸留
    - 規模の大きなモデルの知識を使い、軽量なモデルを作成する
      - 精度の高いモデルから知識を継承することにより、軽量ながら複雑なモデルに匹敵する精度のモデルとなることが期待できる
    - 教師モデルのウェイトを固定し、生徒モデルのウェイトを更新していく
      - 教師モデル<br/>予測精度の高い、複雑なモデルやアンサンブルされたモデル
      - 生徒モデル<br/>教師モデルをもとに作られる軽量なモデル
  - プルーニング
    - モデルの精度に寄与が少ないニューロンを削減することでモデルの軽量化・高速化を図る
    - ウェイトが閾値以下の場合ニューロンを削減し、再学習を行う
## 応用モデル
### 要点のまとめ
- MobileNets
  - 通常の畳み込みレイヤーは計算量が多いという問題に対し、軽量化・高速化・高精度化を実現するモデル
    - 入力のサイズHｘW、チャネル数C、カーネルのサイズKｘK、フィルタ数（出力のチャネル数）Mとして、ストライド1でパディングを適用した場合の計算量は<img src="https://latex.codecogs.com/gif.latex?H\times&space;W\times&space;K\times&space;K\times&space;C\times&space;M"/>
  - Depthwise ConvolutionとPointwise Convolutionの組み合わせで軽量化を実現
  - Depthwise Convolution
    - 入力マップのチャネルごとに畳み込みを実施
      - 出力マップの計算量は<img src="https://latex.codecogs.com/gif.latex?H\times&space;W\times&space;K\times&space;K\times&space;C"/>
    - 各層ごとの畳み込みなので層間の関係性は全く考慮されない
      - 通常はPW畳み込みとセットで使うことで解決
  - Pointwise Convolution
    - 1ｘ1カーネルを使用するため、1x1Convとも呼ばれる
      - 出力マップの計算量は<img src="https://latex.codecogs.com/gif.latex?H\times&space;W\times&space;C\times&space;M"/>
  - 通常の畳み込みの計算量が<img src="https://latex.codecogs.com/gif.latex?H\times&space;W\times&space;K\times&space;K\times&space;C\times&space;M"/>なのに対して、<br/><img src="https://latex.codecogs.com/gif.latex?H\times&space;W\times&space;K\times&space;K\times&space;C+H\times&space;W\times&space;C\times&space;M=H\times&space;W\times&space;C\times(K\times&space;K+M)"/>に削減できる

- DenseNet
  - 層が深くなるにつれて学習が難しくなるというニューラルネットワークの問題に対して、DenseBlockと呼ばれるモジュールを用いる
    - 出力層に前の層の入力を足しあわせる
    - 入力特徴マップのチャンネル数が<img src="https://latex.codecogs.com/gif.latex?n\times&space;k"/>だった場合、出力は<img src="https://latex.codecogs.com/gif.latex?(n+1)\times&space;k"/>となる
    - kはネットワークのgrowth rateと呼ばれる
  - DenseBlock同士の間にはTransition Layerと呼ばれるダウンサンプリングを行う層が挟まる
  - ResNetとの違い
    - DenseBlockでは前方の各層からの出力全てを後方の層へ入力
    - RessidualBlockでは前1層の入力のみ後方の層へ入力
- WaveNet
  - 音声波形を生成するモデル
    - Pixel CNNを音声に応用したもの
  - 時系列データに対して畳み込み（Dilated causal convolution）を適用
    - 深層学習を用いた結合確率の学習が効率的に行えるアーキテクチャ
    - 層が深くなるにつれて畳み込むリンクを離し、パラメータ数に対する受容野が広いという利点がある

## Transformer
### 要点のまとめ
- ニューラル機械翻訳の問題点として、翻訳元の文の内容をひとつのベクトルで表現するため、文長が長くなると表現力が足りなくなる
- 翻訳先の各単語を選択する際に、翻訳元の文中の各単語の隠れ状態を利用：Attention
- Attentionは辞書オブジェクトと解釈できる
  - queryに一致するkeyを索引し、対応するvalueを取り出す操作と見做すことができる
- Transformer
  - Encoder-DecoderモデルであるがRNNは使わない
  - 必要なのはAttentionだけ
  - 当時のSOTAをはるかに少ない計算量で実現
- Transformerの全体像（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/121360770-ad3de680-c96f-11eb-9daf-92b4036cb079.png" width=500/>
  - 2種類のAttentionを持つ
  - Source Target Attention
    - queryはTarget（翻訳先の単語の隠れ状態）、keyとvalueはSource（翻訳元の単語の隠れ状態）
  - Self-Attention
    - query、key、valueすべて翻訳元あるいは翻訳先の単語
  - Encoder
    - Self-AttentionとFeed-Forwardの組み合わせが6層
    - Self-Attentionにより、文脈を考慮して各単語をエンコード
      - 次元に応じてスケーリング（Scaled dot product attention）
      - 重みパラメタの異なる８個のヘッドを使用（Multi-Head attention）
    - Position-Wise Feed-Forward Networksにより、位置情報を保持したまま順伝播
  - Decoder
    - Encoderと同じく6層
    - 各層で2種類のAttention
    - Attentionの仕組みはEncoderとほぼ同じ
  - Positional Encoding
    - RNNを用いないので、文章の語順情報を追加する必要
    - 単語の位置情報をエンコード<br/><img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0APE_%7B%28pos%2C+2i%29%7D+%26%3D+%5Csin+%5Cleft%28+%5Cfrac%7Bpos%7D%7B10000%5E%7B2i%2F512%7D%7D+%5Cright%29+%5C%5C%0APE_%7B%28pos%2C+2i%2B1%29%7D+%26%3D+%5Ccos+%5Cleft%28+%5Cfrac%7Bpos%7D%7B10000%5E%7B2i%2F512%7D%7D+%5Cright%29+%0A%5Cend%7Balign%2A%7D%0A" width=200/>
  - Attentionの可視化
    - 言語構造を捉えていることが多い（講義資料より）<br/><img src="https://user-images.githubusercontent.com/34636490/121367111-fd6b7780-c974-11eb-8764-d54d348cb912.png" width=400/>

## 物体検知-セグメンテーション
### 要点のまとめ
- a
