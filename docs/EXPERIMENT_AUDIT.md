# 논문 v1 전체 실험 감사표

대상: *Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models*, arXiv:2507.09209v1 (2025-07-12). 첨부 PDF의 본문과 부록 A–K를 확인했다. 문서 속 프롬프트는 실험 대상 자료로만 취급했다.
공식 코드 확인 시점 HEAD: `edcb03c1ad406cc87171319d59a628ccefa9fbdc`.

사용자가 선택한 범위는 **Phi3V-Med와 Phi3.5V-Med의 순수 baseline**이다. RAG·Expert-RAG·Expert-CFG를 구현하거나 실행했다고 주장하지 않는다. 아래 표는 범위 밖 실험도 누락하지 않고 그 방법과 필요 자료를 기록한다. “지원”은 코드 경로가 있다는 뜻이며 GPU 실험 완료라는 뜻이 아니다.

| 위치 | 실험·방법·metric | 이번 구현 / 비교 조건 |
|---|---|---|
| §3.1, §4.3 | Phi3V/3.5V 4.2B → PubMedVision alignment → instruction → 벤치마크별 downstream LoRA | 공개 medical adapter를 재사용하고 downstream 학습 지원. 대규모 사전 두 단계 학습을 T4에서 다시 실행하지 않음. 공개 adapter의 정확한 학습 단계/데이터/논문 대응 SHA는 저자 확인 필요 |
| 표 1 | VQA-RAD, SLAKE, PathVQA × Open recall, Closed accuracy, Overall(샘플 수 가중 평균) × 두 Med baseline | 18개 baseline 논문 수치 저장. downstream 완료 adapter로만 표 1 비교. Open을 exact-match accuracy라고 표시하지 않음 |
| 표 1 나머지 | 분류형 M2I2/Prefix/PubMedCLIP/M3AE/BioMedCLIP/MUMC; 생성형 LLaVA-Med 7B/13B, BioMed-VITAL; RAG 전체·상위5%, Expert-RAG5%, Expert-CFG5% | 외부 비교 모델 및 개입 방법은 선택한 baseline 범위 밖. 논문 수치를 자신의 실행 결과로 재사용하지 않음 |
| 그림 3 | Phi3.5V+PubMedVision baseline의 entropy로 오답 검출, ROC/AUROC, sensitivity, specificity | 전체 ROC 좌표와 sensitivity/specificity 출력. 논문 AUROC RAD .83 / SLAKE .85 / Path .91. open 오답 이진화 기준 미공개 → 이 구현은 recall<1을 명시. 논문의 box plot 반복/threshold 집계 방식 미공개 |
| 그림 4a | entropy 높은 test 상위 5,10,15,20,25,30%에 expert 개입; Open/Closed/Overall 변화 | baseline은 그림의 시작점만 해당. 곡선의 Expert-CFG는 범위 밖. 숫자 원자료 미공개라 그래프 점을 임의 정밀수치로 만들지 않음 |
| 그림 4b | entropy threshold .3,.4,.6,.8,1.1,1.5 이상 개입; 같은 3 metrics | baseline 시작점만 해당. 개입 생성·오라클 GT·전문가 자료 필요 |
| 그림 5 | open QA만, caption에 정답 직접 등장하는 Top-K hit rate(K=1…10 축, 4에서 포화) | 외부 retrieval artifact가 있으면 K=1,2,3,4,6,8,10 hit 평가 지원. 그래프 원수치 미공개. closed yes/no 제외 |
| 표 2 | BioMedCLIP image / text / sum / union feature Top-1 hit rate | 12개 논문값 저장, `baseline.auxiliary`가 정렬된 retrieval 결과를 평가. 실제 feature 추출·FAISS 검색은 baseline 범위 밖 |
| 표 3 | downstream 없는 zero-shot **closed만**; 모델·instruction data 비교 | Phi3/3.5 원본과 공개 Med adapter 평가 지원. 전체 7행 논문값 보관. LLaVA7B, +LLaVA-Med, Phi3+LLaVA-Med 체크포인트는 제공되지 않아 실행 지원하지 않음 |
| 그림 6 | 세 데이터셋 zero-shot case, 원본/RAG/CFG 응답, 생성 토큰 확률 시각화 | baseline 응답·평균 entropy·선택 토큰 NLL 저장. RAG/CFG 및 그림의 token별 색칠은 미구현. 같은 이미지/질문 ID의 공개 매핑 필요 |
| 표 4 / 부록 A | 데이터 규모·용도 | 아래 데이터 프로토콜 참조. 전체 규모를 test 표본 수로 잘못 사용하지 않음 |
| 그림 7 / 부록 B | train QA를 이미지 단위 합쳐 GPT-4 caption 작성; no 답변 제외 | retrieval 전처리, 범위 밖. train만 사용, test QA를 지식베이스에 넣으면 누출 |
| 그림 8–9 / 부록 C | GPT-4 caption 키워드 추출 → 질문 관련 highlight 선정 | 범위 밖. 별도 expert gold 필요. GT를 일반 baseline prompt에 넣지 않음 |
| 표 5 | VQA-RAD GPT-4o keyword Recall 89.57 / Precision 98.36 / F1 93.76 | 외부 gold/predicted phrase artifact의 micro P/R/F1 지원. 저자 gold·버전·matching 규칙 미공개로 완전 동일 비교 불가. 본문 'recall>98%' 서술은 표와 상충하며 **표 값을 채택** |
| 표 6 / 부록 D | yes/no QA, zero-shot/fine-tune; label prob(10 sampling,T=1), Is True?, entropy; HuatuoV32B는 verbal top1 포함; ECE↓/ECE-t↓/BS-t↓/AUC↑ | 두 Phi baseline의 label samples, 자기검증, full-vocab entropy 지원. ECE 15 equal-width bins, exp(-H) confidence, disjoint validation temperature scaling을 명시한 **잠정 프로토콜**. label confidence는 10응답 중 greedy label 일치 비율(무효 응답도 분모 포함). Is True는 True/False 단일 token 합의 상대확률. 저자와 동일함 미확인. validation 없으면 ECE-t/BS-t는 null. fine-tuned Is True는 논문도 제외. HuatuoV32B 제외 |
| 그림 10–12 / 부록 E | Path/RAD/SLAKE answer별 entropy 분포, answer별 Top1/2/3 retrieval hit | report.json answer별 entropy 원자료/평균/개수, auxiliary answer별 hit 지원. 논문에 선택된 answer 목록/순서는 원자료가 없어 자동 동일 복제 불가 |
| 그림 13 / 부록 F | caption hit/miss CLIP score 분포, Top-K 선택; .6 미만 대체로 비정보적 | 외부 retrieval artifact hit/miss score 배열 출력. encoder/KB 필요. .6을 universal threshold로 사용하지 않음 |
| 표 7 / 부록 G | 100% RAG, Expert-RAG, Expert-CFG; α=0,.01,.1; β=1,3,5; γ=1,1.3,1.5; Overall | 순수 baseline은 82.04/85.20/65.24로 표1과 동일. CFG grid는 범위 밖. 본문 γ=1.5와 표7 중심값1.3은 상충; baseline에는 영향 없음 |
| 그림 14 / 부록 H | 추가 zero-shot baseline vs CFG 영문 사례 | baseline 예측 전량 보관. 이미지·샘플 매핑을 확보하면 같은 예제 선택 가능. expert 응답을 생성했다고 주장하지 않음 |
| 그림 15 / 부록 I | 중국어 zero-shot baseline vs CFG, 영/중 불일치 | 명시적으로 번역한 질문으로 별도 infer 실행 가능. 데이터 준비기는 기본 SLAKE English만 사용. 공인 중국어 번역·같은 이미지 매핑 미공개, 전체 benchmark와 섞지 않음 |
| 표 8 / 부록 J | entropy<.35 오답을 8종류로 수작업 분류, RAD 23/SLAKE 69 사례의 비중 | 오답 ID 추출 및 수작업 tag별 비중/논문비중 출력. 질문모호성·동의어·위치·정도·질환·숫자·형태·학습미등장; 자동 의료 오류 판정하지 않음 |
| 그림 1–2 / §6 / 부록 K | 개념도·pipeline·결론·한계 | 별도 수치 실험 아님. 짧은 답/greedy의 제한 명시 |

## 데이터·학습 프로토콜

* 전체 데이터 규모: VQA-RAD 315 images/3,515 QA, SLAKE 642/7,000, PathVQA 4,998/32,799. SLAKE에는 언어 구분이 있어 영어만 평가한다. 논문의 정확한 split manifest·질문 ID·필터 규칙은 공개 저장소에서 찾을 수 없다. 임의 split 생성하지 않는다.
* PubMedVision: alignment 647k와 instruction 647k, §4.3 두 단계 LR2e-4 / effective batch256 / 3 epochs / AdamW / cosine. 표4는 1,009,700 images,647,031 descriptions,647,031 QA로 표기하며 단계별 647k를 하나로 혼동하지 않는다.
* downstream: **각 데이터셋별 별도 adapter**, LR1e-4 / effective batch64 / 10 epochs / LoRA rank64 / AdamW / cosine. test는 학습/온도선택에 사용하지 않는다.
* 원 논문은 24GB RTX3090 / BF16. T4는 BF16 미지원: FP16 + NF4 QLoRA로 메모리 절감, microbatch1 × accumulation64. accumulation은 effective batch를 맞추지만 BF16 full-precision LoRA와 수학적으로 동일하지 않다. warmup/weight decay/LoRA target modules/seed/프롬프트별 상세 학습 설정은 논문에 모두 없으므로 같다고 주장하지 않는다.
* 공식 demo 기본값은 crops16. 추론 기본16 유지; 학습 기본4는 T4 메모리 절충이다. 여유가 있으면 학습도 `--crops 16`. OOM시 crops4로 **별도 이름의 실험**을 실행하고 차이를 기록한다. 자동 축소하지 않는다.
* 지식베이스: 세 benchmark의 train QA를 이미지 단위 합쳐 negative 답 제외, ROCO70,306 + PEIR33,572 포함. 본 실험의 순수 baseline에는 KB·FAISS·GPT API가 필요 없다.

## 재현성의 경계

현재 환경에서 검증한 것은 CPU metric/변환/보고서 코드이다. GPU, 모델 다운로드, adapter 호환성, downstream 학습, 실제 수치는 아직 검증되지 않았다. 공식 README가 안내하는 Hugging Face adapter의 존재와 정확한 학습 stage는 현재 네트워크 차단으로 재확인할 수 없다. Colab은 이를 온라인으로 검증하며 실패를 숨기지 않는다.

표1과 표3의 논문값은 서로 다른 설정이다. 양자화·crop 수·학습 stage·split·metric 구현이 다르면 수치차를 단순 재현 실패로 해석할 수 없다. 647k×두 단계 재학습은 무료 T4 세션에 적합하지 않아 공개 adapter를 사용하는 경로를 선택했다. 세 dataset×두 모델 downstream도 여러 세션이 걸릴 수 있으므로 Drive에 checkpoint를 저장한다. 학습 완료 marker가 없는 checkpoint는 표1 비교 경로에서 거절한다.
