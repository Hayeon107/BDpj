# BD term project
- 음성 데이터를 활용한 감정 분류 프로젝트 </br>
  - data : ai hub - 감정 분류를 위한 대화 음성 데이터셋 </br>
    (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263) </br>
  - 사용 특성 : mfcc
  - 사용 모델 : LSTM, (추가 예정) </br>

## 오디오 데이터 처리에 대한 간단 요약
### Feature engineering
- 스펙트럼
  - 파동의 시간 영역을 주파수 영역으로 변환
  - 음향 신호를 주파수, 진폭으로 분석하여 보여줌
  - 고속 푸리에 변환을 적용
  - X축은 주파수, y 축은 진폭을 나타냄
- 멜스펙트럼
  - 주파수 특성이 시간에 따라 달라지는 오디오를 분석하기 위한 특징 추출 기법
  - 인간의 청각 영역을 반영한 mel scale을 적용
    - 보통 고주파로 갈수록 사람이 구분하는 주파수 간격이 넓어진다는 것을 반영
  - 프레임의 길이와 슬라디이 범위를 하이퍼 파라미터로 설정
  - x축은 시간 y축은 주파수 z 축은 진폭을 나타냄
- MFCC
  - 멜 스펙트럼에서 켑스트럴 분석을 통해 추출된 값
    - 켑스트럴은 스펙트럼에서 배음 구조를 유추할 수 있도록 도와주는 분석
  - 로그 멜 스펙트럼에 역푸리에 변환을 적용
    - 주파수 정보의 상관관계가 높은 문제를 해소
- librosa 패키지
  - 오디오 신호를 분석하는 Python 모듈
  - 오디오 데이터 입출력 및 다양한 특징 추출 방법론을 쉽게 사용할 수 있음
  - 오디오 데이터로 추출된 특징의 시각화 기능 제공
- 추출된 특징 사용
  - 고차원 특징이기에 머신러닝 모델 적용을 위해 기술 통계량을 사용
  - 최근 딥러닝 모델에는 특징 고유의 값이나 히트맵 이미지로 변환하여 사용
    - 멜스펙트럼을 CNN 하기도 함
### Data augmentation
- 데이터 증강 기법
  - 일반적으로 레이블이 존재하는 데이터에 변화를 주어 원본 데이터와 같은 레이블을 갖는 새로운 데이터를 만드는 기법
    - 좌우 반전 / 이미지 자르기 / 이미지 회전 / 밝기 조절 등
- 소리 데이터에 적합한 데이터 증강 기법
  - Adding noise : white noise를 추가하여 원본 오디오에 잡음 생성
  - Shifting : 원본 오디오 데이터를 좌우로 이동
  - Stretching : 원본 오디오 데이터의 빠르기를 조정
    - 잘못하면 소리 특성이 사라질 수 있기때문에 적절하게 사용해야함
### Deep learning model
- 소리 데이터를 위한 딥러닝 모델
  - 텍스트를 음성으로 변환하는 TTS(Text to Speech) 를 수행하기에 적합한 딥러닝 모델
- Wavenet
  - 오디오의 파형 형태를 직접 사용해서 새로운 파형을 생성하는 확률론적 모델
  - 30개의 residual block 을 쌓은 형태의 구조를 보임
  - 주요 특징
    - 음성 파형 학습을 위한 새로운 구조를 제시
    - 조건부 모델링을 이용해 특징적인 음성을 생성할 수 있음
    - 오디오 파형만을 이용해 자연스럽고 새로운 음성 파형을 생성할 수 있음
  - SoftMax 함수
    - Wavenet 입.출력은 아날로그 음성 데이터를 변환한 디지털 데이터 값
    - 오디오는 16비트의 정수 값으로 저장해서 사용하기에 확률론적 모델링이 힘듦
      - 65,536개의 확률 고려해야함
    - 뉴 laq companding 변환을 통해 정수버위로 총 256개의 확률 고려
  - Dilated Causal Convolutions
    - Dilation 기법을 이용해 일정 스탭을 건너 뛰며 필터를 적용하는 dilated causal convolution
    - 적은 층의 레이어로 receptive field를 넓힐 수 있는 효과를 가져옴
    - Dilated convolution과 Causal convolution 개념을 결합한 컨볼루션 연산
    - 필터에 zero padding 을 추가해 모델의 receptive field를 늘려줌
      - receptive field란 필터가 한 번에 볼 수 있는 데이터를 탐색할 수 있는 영역
      - 입력된 데이터의 특징을 잡아내기위해 receptive field 는 높을수록 좋음
    - Causal Convolutions : 시간 순서를 고려하여 필터를 적용하는 컨볼루션 연산
      - RNN 계열의 모델처럼 시계열 데이터를 모델링
      - Receptive field를 넓히기 위해서 많은 양의 레이어를 쌓아야한다는 단점이 존재
  - Conditional wavenet
    - 확률 모델에 조건  정보를 추가함으로써 특정한 성질을 가진 오디오를 생성할 수 있음
    - 조건부 모델링방법
      - 전역적 조건 : 시점 변화에 영향을 받지 않는 정보추가
      - 지역적 조건 : 시점 변화에 영향을 받지 않는 정보추가
  - Generated Aduio by Wavenet
    - 기본 TTS 방법론의 부자여느러운 음성과 달리 자연스러운 음성 생서어
    - 화자의 음성 정보로 활용하여 다양한 음색의 음성 생성
    
- Feature & CNN model for audio classification
  - Feature : Mel Spectrogram / MFCC 권장
  - CNN Model : Resnet 권장
