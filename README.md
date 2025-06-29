<p align="center">
  <h1 align="center">N-TIDE: Debiasing Unimodal Vision Models via Neutral Text Inversion with CLIP</h1>
  <p align="center">
    <a>Chanhee Lee</a>
    ·
    <a>Jinho Jang</a>
    ·
    <a>Sarang Han</a>
  </p>
  <p align="center">
    <i>Sungkyunkwan University · Department of Applied Artificial Intelligence</i><br>
    <i>2025-Spring Introduction to Deep Learning Course Term Project</i>
  </p>
</p>

---
## 📄 [Paper](https://drive.google.com/file/d/1XQGbxueJkxlq0xKpMo7ILOakWykr1_zk/view?usp=sharing)

### 🖼️ Model Overview
![overview](./assets/N_TIDE.png)

---

## 📝 Abstract

Mitigating bias in vision models is challenging, particularly when semantic attributes subtly influence predictions. While vision-language models like CLIP provide strong debiasing signals, they require text input at inference, limiting their use in image-only settings. We introduce **N-TIDE** (Neutral Text-Inversion for Distillation-based Equilibration), a two-stage framework that distills CLIP’s fairness guidance into a unimodal vision model. In the first stage, we propose a novel *neutral-text inversion* process, which regularizes the model by aligning a trainable neutral-text embedding with CLIP’s null-text embedding. This alignment captures semantic debiasing cues without requiring text at test time. In the second stage, we transfer these cues into an image-only encoder via cosine-based feature matching. We further interpret this process through the lens of deterministic diffusion, framing semantic alignment as a guided trajectory.

Experiments on FairFace show that N-TIDE improves fairness metrics such as Equalized Odds and Representation Bias Difference with minimal accuracy loss. Though the fairness gains are moderate and the diffusion analogy remains conceptual, N-TIDE offers a practical path to integrating multimodal supervision into efficient vision-only models.


---

## :sparkles: N-TIDE: Neutral Text-Inversion for Distillation-based Equilibration

Sungkyunkwan University Applied Artificial Intelligence

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
4. 데이터 설정 및 코드 실행:
   ```
   python train.py --dataset_name "FairFace"                       # FairFace, Race 7-class
   python train.py --dataset_name "FairFace" --is_fairface_race_4  # FairFace, Race 4-class (White, Black, Asian, Indian)
   ```
6. 가상환경 나가기:
    ```
    conda deactivate
    ```

---

## 📚 Citation

> This work was completed as part of the 2025-Spring *Introduction to Deep Learning* course term project at Sungkyunkwan University. Although the paper is not officially published, if you wish to cite it in your work, please use the following BibTeX entry:

```bibtex
@misc{lee2024ntide,
  title     = {N-TIDE: Debiasing Unimodal Vision Models via Neutral Text Inversion with CLIP},
  author    = {Chanhee Lee and Jinho Jang and Sarang Han},
  note      = {Class project report, Sungkyunkwan University},
  year      = {2025},
  howpublished = {\url{https://github.com/iontail/N-TIDE}},
  institution = {Sungkyunkwan University}
}
