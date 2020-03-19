# Seq2Seq Model
- 최종모델
- Parameter
- 모델 별 비교
## Parameter

### 1. File Path
- data_path : 불러올 Data 위치
- dictionary_path : Dictionary 저장위치
- src_train_filename : 한국어 train파일 이름
- tar_train_filename : 영어 train파일 이름
- src_val_filename : 한국어 validation 파일 이름
- tar_val_filename : 영어 validation 파일 이름
- model_path : 모델 저장 위치

### 2. Model Hyper Parameter
- sequence_size : 문장 최대길이 (default : 50)
- embedding_dim : 임베딩 차원 (default : 500)

### 3. Encoder
- encoder_rnn_dim : Encoder Hidden Dimension(default : 300)
- encoder_n_layers : 인코더의 LSTM의 층 수 (default : 3)
- encoder_embedding_dropout : 임베딩 시 dropout (default :0.1)
- encoder_rnn_dropout : LSTM의 dropout (default :0.1)
- encoder_dropout : 인코더의 Dropout (default :0.1)
- encoder_residual_used : 인코더 잔차연결(Residual Connection) 사용여부 (default :True)
- encoder_bidirectional_used : 인코더 양방향(Bidirectional) LSTM 사용여부(default :True)
- encoder_output_transformer : Bidirectional Output의 출력 크기 (default :300) 
- encoder_output_transformer_bias : Bidirectional Output bias 사용여부 (default: True)
- encoder_hidden_transformer : Bidirectional Hidden layer의 출력 크기 (default : 300)
- encoder_hidden_transformer_bias : Bidirectional Hidden layer bias 사용여부 (default: True)

### 4. Decoder 
- decoder_rnn_dim : Decoder Hidden Dimension(default : 300)
- decoder_n_layers : 디코더의 LSTM의 층 수 (default : 3)
- decoder_embedding_dropout : 임베딩 시 dropout (default :0.1)
- decoder_rnn_dropout : LSTM의 dropout (default :0.1)
- decoder_dropout : 디코더의 Dropout (default :0.1)
- decoder_residual_used : 디코더의 잔차연결(Residual Connection) 사용여부 (default :True)

### 5. Learning Hyper Parameter
- learning_method : Teacher Forcing과 Scheduled Sampling 중 선택 (default : )
- learning_rate : learning rate (default : 0.001)
- epochs : 전체 데이터 셋에 대해 몇번 학습을 할 것인가? (default : 100)
- batch_size : 한번 batch에 데이터 샘플의 Size (default : 400)
- train_step_print : train 결과를 보는 step 수(default : 10)
- val_step_print : valdation 결과를 보는 step의 수(default : 100)
- step_save : model을 저장하는 step의 수(default : 1000)

## 모델 별 성능 비교
- Validation의 경우 Teacher Forcing을 안하니깐 학습 결과를 정확히 보기 어려워서 어쩔 수 없이 Teacher Forcing을 적용한 결과입니다.
- small Data set(train 10만개, validation 1만개) 사용
![모델비교1](https://user-images.githubusercontent.com/47970983/77049414-6468dd80-6a0b-11ea-910b-d87268ea2cf4.png)
- Train vs val <br>
![모델비교1_train](https://user-images.githubusercontent.com/47970983/77049415-65017400-6a0b-11ea-8ca8-3795fe5df36b.png)
![모델비교1_val](https://user-images.githubusercontent.com/47970983/77049417-65017400-6a0b-11ea-9e5b-28b3b7d58f17.png)
<br>
![모델비교2](https://user-images.githubusercontent.com/47970983/77049418-659a0a80-6a0b-11ea-9668-66e27bb703d3.png)
- Train vs val <br>
![모델비교2_train](https://user-images.githubusercontent.com/47970983/77049419-659a0a80-6a0b-11ea-9e02-74c62519370d.png)
![모델비교2_val](https://user-images.githubusercontent.com/47970983/77049421-6632a100-6a0b-11ea-8d7c-093a1babaa3d.png)
<br>
![모델비교3](https://user-images.githubusercontent.com/47970983/77049422-6632a100-6a0b-11ea-9520-b4ec12902933.png)
- Train vs val <br>

![모델비교3_train](https://user-images.githubusercontent.com/47970983/77049409-6337b080-6a0b-11ea-8d38-e48cb1e8a277.png)
![모델비교3_val](https://user-images.githubusercontent.com/47970983/77049413-63d04700-6a0b-11ea-84ae-0bb4fb1310c2.png)
