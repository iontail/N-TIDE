<p align="center">
  <h1 align="center"> N-TIDE:Debiasing Unimodal Vision Models via Neutral Text Inversion with CLIP
</h1>
  <p align="center">
    <a>Chanhee Lee</a>
    ·
    <a>Jinho Jang</a>
    ·
    <a>Sarang Han/a>
  </p>


## :sparkles: N-TIDE: Neutral Text-Inversion for Distillation-based Equilibration
Sungkyunkwan University Applied Arificial Intelligence


## 1. 설치 

프로젝트를 설치하려면 다음 단계를 따르세요:

1. 다음 명령어를 사용하여 프로젝트를 복제:
   ```
   git clone https://github.com/iontail/mitigating_bias.git
   ```
2. 복제된 프로젝트 폴더로 이동:
   ```
   cd mitigating_bias
   ```
3. 새로운 가상환경 생성 및 필요한 패키지를 설치:
   ```
   conda create -n MB python=3.10 # 해당 코드는 최초 1회만 실행
   conda activate MB  
   pip install -r requirements.txt
   ```

4. configuration 수정:
   - src/models/utils/config.py 에서 하이퍼파라미터 설정합니다.
   - 이때 아래 코드는 반드시 본인 실행환경 및 테스크에 맞게 수정해주세요.
   ```
   _C.MODEL_NAME = 'Final_MB_SLM'
   
   _C.DATASET.TARGET_LABEL_IDX = 1 # 0: age, 1: gender, 2: race
   _C.NUM_CLASSES = 2 # age: 0~116, gender: 0~1, race: 0~4
   ```

5. wandb 설정:
   - wandb에서 로그를 확인하기 위해 wandb 설정을 진행해주세요. (모델 이름도 수정해주세요.)
      

7. 데이터셋 디렉토리 생성:
   - Root 디렉토리에 'data' 파일을 만들고, 그 안에 원하는 데이터셋 파일을 만드세요.
   ```
   mkdir data
   cd data
   mkdir [데이터셋이름] #mkdir utkface
   ```
8. 데이터셋 다운로드
   - 원하는 데이터셋을 다운받고 압축해제해주세요.
   ```
   cd utkface
   gdown 16uEk67PncGCl0GxBRfa0iXfobODcs4tu
   unzip 04_UTKFace.zip
   ```
7. config.py에서 데이터셋의 루트 디렉토리를 설정하세요
   - 아래 코드를 자신의 환경에 맞게 바꿔야합니다
   ```
   _C.DATASET.ROOT_DIR = '/Users/[username]/mitigating_bias/data/utkface' # MAC
   ```
9. 원하는 코드 실행
   ```
   python train.py
   ```
10. 가상환경 나가기
    ```
    conda deactivate
    ```
