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
   git clone https://github.com/iontail/N-TIDE.git
   ```
2. 복제된 프로젝트 폴더로 이동:
   ```
   cd N-TIDE
   ```
3. 새로운 가상환경 생성 및 필요한 패키지를 설치:
   ```
   conda create -n N_TIDE python=3.10 # 해당 코드는 최초 1회만 실행
   conda activate N_TIDE  
   pip install -r requirements.txt
   ```
4. 코드 실행:
   ```
   python train.py
   ```
5. Configuration (e.g., Dataset, Learning Rate) 설정:
   ```
   python train.py --dataset_name "FairFace" --m_backbone_lr 1e-5 --m_head_lr 1e-4
   ```
6. 가상환경 나가기:
    ```
    conda deactivate
    ```
