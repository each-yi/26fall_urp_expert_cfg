# Expert-CFG 논문의 두 Med baseline 재현 (Colab T4)

대상: [arXiv:2507.09209v1](https://arxiv.org/abs/2507.09209), **Phi3V-Med / Phi3.5V-Med**, RAG·Expert-CFG 없이 실행한다. [Colab notebook](notebooks/colab_t4.ipynb)을 Colab에 업로드하여 순서대로 실행한다. 논문의 모든 표·그림·부록 실험과 구현 범위는 [실험 감사표](docs/EXPERIMENT_AUDIT.md)에 정리했다.

**실제 논문 수치 재현은 아직 완료하지 않았다.** 이 서버는 GPU가 없고 Hugging Face 접근도 차단되어 있어 모델 추론·학습은 미실행이다. CPU 단위/통합 테스트만 통과했다. T4 양자화 결과가 논문과 일치한다고 보장하지 않는다.

## 두 가지 실행 설정

|설정|초기 모델과 작업|논문 비교|
|---|---|---|
|`zero_med`|공개 medical LoRA를 불러와 평가|표3 Closed만. Open/Overall은 측정하지만 논문 대응값 없음|
|`downstream`|medical LoRA → 각 benchmark train으로 별도 10epoch QLoRA → test|표1 Open/Closed/Overall|

원본 Phi만 불러온 `zero_base`는 Med baseline이 아니며 표3 원본 행에 비교한다. 공개 adapter의 논문 stage 일치 여부는 저자 검증이 필요하다. 표1을 얻으려면 downstream이 필요하며, 공개 medical adapter를 곧바로 표1에 대조하지 않는다.

## 1. Colab 설치

새 T4 런타임을 사용한다. notebook 첫 셀이 GitHub main의 코드를 `/content/26fall_urp_expert_cfg`에 자동으로 clone한다. Drive는 데이터와 checkpoint 보관에 사용하며 코드 폴더를 직접 업로드할 필요가 없다.

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-colab.txt
```

설치 후 **런타임을 재시작**한다. Python3.10–3.12를 대상으로 고정한 stack이다. 2026년 Colab 기본 torch를 그대로 쓰면 remote model code와 API가 달라질 수 있어 별도 고정한다. FlashAttention2·BF16은 T4 경로에 사용하지 않는다. HF model remote code를 사용하며 실행 시 현재 revision을 정확한 SHA로 resolve하여 기록한다. 처음 온라인 실행 후 `--revision SHA --adapter-revision SHA`로 고정한다. 개인 토큰은 notebook이나 저장소에 기록하지 않는다.

## 2. 공식 데이터의 split을 보존해 준비

소스: [VQA-RAD](https://osf.io/89kps/), [SLAKE](https://www.med-vqa.com/slake/), [PathVQA](https://github.com/UCSD-AI4H/PathVQA). 저자 [공식 저장소](https://github.com/ecoxial2007/Expert-CFG)에도 데이터 링크가 있다. 이용 조건을 확인하여 이미지와 공식 train/test annotation을 확보한다. 임의로 train/test를 나누거나 test를 train에 섞지 않는다. 서로 다른 HF mirror의 split을 논문과 동일하다고 가정하지 않는다.

JSONL 스키마 (한 줄=한 질문):

```json
{"id":"test:0","dataset":"vqa_rad","split":"test","image":"synpic1.jpg","question":"Is there an abnormality?","answer":"yes","answer_type":"closed"}
```

`dataset`: vqa_rad/slake/pathvqa. `split`: train/validation/test. `image`: image-root 기준 상대경로. `answer_type`: open/closed. train/test 원 ID를 보존한다. validation은 별도 공식 split 또는 **train 내부에서 미리 분리한 고정 partition**만 사용한다.

```bash
python -m baseline.prepare --input /data/RAD/test.json --dataset vqa_rad --split test --image-root /data/RAD/images --output data/vqa_rad/test.jsonl
# train도 별도 실행. SLAKE는 --dataset slake; 영어(q_lang=en)만 유지.
# PathVQA는 JSON array로 export한 annotation + 실제 파일 구조에 맞춘 template:
python -m baseline.prepare --input /data/PathVQA/test.json --dataset pathvqa --split test --image-root /data/PathVQA/images --image-template '{split}/{img_id}.jpg' --output data/pathvqa/test.jsonl
```

PathVQA pickle은 신뢰할 수 있는 공식 파일을 로컬에서 JSON으로 변환한다. 이 코드는 임의의 pickle을 자동 역직렬화하지 않는다. converter는 이미지를 전부 확인하며 잘못된 경로에서 중단한다. 데이터의 출처 URL, 버전, 선택한 split, 언어 필터, 원본 checksum을 별도 보관한다.

## 3. 두 모델 추론 및 논문 수치 비교

```bash
python -m baseline.infer --model phi35 --setting zero_med --data data/vqa_rad/test.jsonl --image-root /data/RAD/images --output outputs/phi35_zero_vqa_rad.jsonl
python -m baseline.evaluate --data data/vqa_rad/test.jsonl --predictions outputs/phi35_zero_vqa_rad.jsonl --output outputs/phi35_zero_vqa_rad_report --plots
```

`--model phi3`도 실행하고 SLAKE/PathVQA에서도 반복한다. notebook은 두 모델×세 dataset loop를 제공한다. 초기 smoke는 `--limit 3` + **별도 output 경로**로 실행한다. 평가에는 전체 test manifest를 넘겨 coverage를 표시한다. `--limit`를 바꿔 같은 파일에 resume하지 않는다. 정상 중단 후 동일 command는 완료 질문을 건너뛴다. JSONL 끝줄이 쓰다 중단되었다면 파일 사본을 보관하고 불완전 끝줄만 제거한 뒤 재시작한다.

산출물:

* `*.jsonl`: 모든 질문의 원문·정답·응답, Eq4 entropy, 선택 토큰 NLL, truncation 여부. entropy와 NLL은 다른 값이다.
* `*.meta.json`: base/adapter SHA(로컬은 모든 파일 hash), dependency/GPU/양자화/crop/seed/data hash 기록.
* `comparison.md`: **논문값 / 실행값 / 차이(pp)**, zero-shot이면 표3 Closed 기준.
* `report.json`: coverage, counts, ROC, sensitivity/specificity, ECE 계열, answer별 entropy, low-entropy error ID, 모든 응답. optional ROC/hist PNG.

기본 생성은 greedy, batch1, 128 new tokens, crops16, NF4/FP16. EOS 등 special token은 entropy 평균에서 제외한다. truncation이 있으면 별도 run에서 `--max-new-tokens`를 늘린다. OOM시 런타임 재시작 후 `--crops 4`를 사용하고 output 이름도 바꾼다. `--precision fp16`은 메모리가 허용되는 경우 양자화 영향 비교용이다. 자동으로 더 작은 모델로 바꾸지 않는다.

## 4. 표1을 위한 downstream QLoRA

```bash
python -m baseline.train --model phi35 --train data/vqa_rad/train.jsonl --test data/vqa_rad/test.jsonl --image-root /data/RAD/images --output /content/drive/MyDrive/urp_runs/phi35_vqa_rad
# 세션 중단 시 같은 명령에 --resume 추가 (동일 환경 및 설정)
python -m baseline.infer --model phi35 --setting downstream --adapter /content/drive/MyDrive/urp_runs/phi35_vqa_rad --data data/vqa_rad/test.jsonl --image-root /data/RAD/images --output outputs/phi35_downstream_vqa_rad.jsonl
python -m baseline.evaluate --data data/vqa_rad/test.jsonl --predictions outputs/phi35_downstream_vqa_rad.jsonl --output outputs/phi35_downstream_vqa_rad_report --plots
```

모델×dataset별 독립 학습(6개)이다. train은 LR1e-4, AdamW/cosine, 10epoch, rank64, effective batch64(1×64 accumulation), FP16/NF4, checkpointing. 기본 train crops4는 명시적인 T4 절충이며 `--crops 16`으로 원 demo의 crop 설정을 시도할 수 있다. LoRA rank/target은 공개 adapter 설정을 사용하고 rank64를 확인한다. optimizer/checkpoint는 Drive에 보관한다. 10epoch 미만 실험은 논문과 동일하지 않으며 provenance에 epoch 수를 남긴다. test 파일은 split 점검에만 사용하고 학습 loss/모델 선택에 사용하지 않는다.

표1 baseline 논문값 (%):

|모델|dataset|Open|Closed|Overall|
|---|---|---:|---:|---:|
|Phi3V-Med|VQA-RAD|70.39|85.29|79.38|
|Phi3V-Med|SLAKE|82.02|87.02|83.98|
|Phi3V-Med|PathVQA|36.88|91.80|64.43|
|Phi3.5V-Med|VQA-RAD|69.27|90.44|82.04|
|Phi3.5V-Med|SLAKE|82.95|88.70|85.20|
|Phi3.5V-Med|PathVQA|37.42|92.89|65.24|

## 5. 부록 uncertainty 평가와 보조 artifact

* `baseline.infer --label-samples 10`: closed별 추가10회 sampling, temperature1, label 일치 confidence. 원 greedy 출력은 유지. 많은 시간이 추가된다.
* `--self-check`: image+Q+생성답변에 “Is the answer true?” 추가 질의, True/False 상대확률. zero-shot 비교용. 논문의 fine-tuned 모델에는 이 방법의 유효값이 없다.
* `--split validation --data ...`: 별도 validation inference. evaluate에 `--validation-predictions ...`를 주면 validation에서만 temperature를 fit한다. train시 validation partition을 학습에서 제외해야 한다. validation 없으면 ECE-t/BS-t를 계산하지 않는다.
* 논문이 confidence/정규화/bin/calibration split 규칙을 모두 공개하지 않아 표6 비교는 **잠정**이다. exp(-H), 15 equal-width bins를 사용하고 metadata에 명시한다. 논문 entropy 자체는 vocabulary 정규화가 아니라 생성 길이 평균이다.
* low entropy<.35 오답은 `report.json`에 보관. 수작업 tag JSONL(`dataset,id,category`)을 `--error-tags`로 넘겨 표8 비중과 비교한다. category는 감사표/`baseline/paper.py` 참조. annotation 없는 비중을 완전한 오류 분석으로 해석하지 않는다.
* 이미 확보한 retrieval artifact/keyword gold가 있을 때만 `python -m baseline.auxiliary --retrieval FILE --keywords FILE --output outputs/aux.json`으로 표2·5, 그림5·10–13의 hit/score/precision/recall/F1을 계산한다. input schema는 `python -m baseline.auxiliary --help`. 이는 RAG 파이프라인 구현을 뜻하지 않는다.
* 그림4/표7 Expert 개입 sweep은 사용자가 제외한 범위이며 실행하지 않는다. 그래프만 있는 논문값은 임의 추정하지 않는다.

평가 기본 규칙은 lowercase alphanumeric token **multiset recall**, closed는 첫 yes/no 일치이다. 논문 설명 외에 정확한 저자 평가 코드가 제공되지 않아 동일 scorer임을 보장할 수 없다. 문장 전체를 정답 문자열과 비교하거나 synonyms를 임의로 정답 처리하지 않는다. Open 불확실성의 이진 정오답은 recall<1 기준이며 논문 미공개 선택임을 표시한다.

## 검증

```bash
python -m unittest discover -s tests -v
```

실제 의미가 있는 CPU 검증: recall/yes-no/entropy-AUC/calibration edge cases, 데이터 중복·누출·coverage 검출, synthetic 예측→논문수치 비교표+JSON+PNG 통합 경로. **synthetic 결과는 재현 결과가 아니다.** GPU forward/backward와 실제 benchmark 값은 Colab 실행 후 검증해야 한다.
