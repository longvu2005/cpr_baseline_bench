# Audit 29/29 baseline đối chiếu RCR

Nguồn: baseline `65a7c373b98ece2fb632242d13f2ea0cdddd4de6`; RCR
`2fe92766d5158f8dd87297d3bc4cb6280b8ed63b`, ngày 2026-11-09.
Các cột “trước sửa” mô tả đúng hai revision này. Giữ nguyên 29 method IDs,
tên hiển thị và bố cục nhóm. “Đồng bộ” dưới đây nói về code/adapter;
không có phương pháp nào đã chạy full native final benchmark trong môi trường này.

## Những khác biệt trước khi sửa

- Scene CLIP do benchmark sở hữu vẫn dùng OpenAI B/32 và center crop, khác
  HF B/32 letterbox của Proposed. Khác biệt lan qua S6–S9 và S11, không chỉ S1–S4.
  Chưa có đường đọc trực tiếp native globals trong RCR scene index.
- Cả 11 person routes đã có adapter chung và đọc packed RCR detections ở base
  commit. Geometry, fallback, Hungarian anchors và membership không cần viết lại.
  Còn thiếu ràng buộc index với byte ảnh nguồn, môi trường selector độc lập
  với native dependencies, fingerprint implementation/runtime và kiểm tra cap khi
  đọc cache. Đây là lỗ hổng provenance, không phải bằng chứng 11 thuật toán đều sai.
- Kết quả phụ thuộc S5/S8→S9 và S6→S11 có kiểm tra manifest/person metadata,
  nhưng chưa khóa đủ source config, checkpoint bytes, actual implementation và
  scene runtime mới. Cache/result cũ có thể bị hiểu là cùng một protocol.
- Instruct-ReID dùng `-1,000,000` chưa calibration cho thiếu người. Native vectors
  không normalized nên `-1` cũng không có cơ sở. Word4Per chưa bắt buộc receipt
  buộc Stage-2 với dữ liệu train/selection ngoài RCR. WISER chưa lưu actual loaded
  OpenCLIP/BLIP2 states và raw API response/model identity.
- Final importer/evaluator đã dùng đúng val/test, Full/ID positives, stable ties,
  reference exclusion và Macro-Full-mAP. Giữ nguyên công thức và kiểm chứng trực tiếp.

## Bảng từng phương pháp

C = code/adapter đã đồng bộ; N = native algorithm giữ nguyên có chủ đích;
R = còn thiếu bằng chứng chạy full native trên final data. R áp dụng cho **29/29**.
Tất cả rows nhận fingerprint/runtime/result-version guard và evaluator chung;
bảng chỉ nêu phần riêng của từng phương pháp.

| # | Method ID không đổi | Trước sửa: khác RCR/rủi ro chính | Sau sửa và phần giữ nguyên | Trạng thái/bằng chứng còn thiếu |
| --- | --- | --- | --- | --- |
| S1 | `clip_image` | OpenAI center crop khác Proposed; cache riêng | Chung frozen HF B/32 letterbox; đọc globals RCR nếu proven; cosine giữ nguyên | C, R; actual pretrained image/cache parity đã qua, chưa full test inference |
| S2 | `clip_text` | OpenAI tokenizer/loader, scene gallery khác Proposed | Chung HF native tokenizer/EOS/right padding 77 và raw global projection | C, R; actual pretrained text parity đã qua |
| S3 | `clip_early_fusion` | Gallery/text encoder khác Proposed | HF chung; normalized early composition giữ nguyên; 21 weights chọn val Full-mAP | C, R; fusion/evaluator contracts qua; cần val selection + full test thật |
| S4 | `clip_late_fusion` | Encoder khác Proposed | HF chung; z-score bỏ reference; 21 weights chọn val Full-mAP | C, R; cần val selection + full test thật |
| S5 | `groundingdino_clipreid_set` | Person index thiếu source-image/runtime binding | Dùng chung packed boxes/selector; CLIP-ReID MSMT17 B/16, transform và cosine giữ nguyên | C+N, R; thiếu native ReID weights/data run |
| S6 | `groundingdino_clipreid_set_text` | Nhánh scene CLIP sai pipeline; phụ thuộc cache S5 | HF text/gallery chung; native ReID, score fusion và val alpha rule giữ nguyên; khóa dependency config/assets | C+N, R; cần native S5/val/test |
| S7 | `qwen25vl_rewrite_clip` | Retriever center crop; rewrite/features thiếu runtime binding | Qwen2.5-VL 7B/prompt/generation giữ nguyên; rewrite encode bằng HF chung | C+N, R; chưa chạy native rewrite |
| S8 | `qwen_image_edit_clip` | Generated-image encoding khác Proposed; cache chưa đủ identity | Qwen-Image-Edit-2509 giữ nguyên; ảnh sinh ra và gallery cùng HF letterbox; fingerprint pixel output | C+N, R; chưa native editor inference |
| S9 | `reid_set_qwen_edit_proxy` | S5/S8 scores có thể stale; val proxy dùng CLIP cũ | Kiểm tra source score/config/weights/scene metadata; val proxy và test dùng cùng S8 HF pipeline; fusion giữ nguyên | C+N, R; cần hai source runs thật |
| S10 | `per_person_clip_compose_setmatch` | Native L/14 khác scene B/32 nhưng có chủ đích; provenance chưa đủ | Giữ native OpenAI L/14 crop/image-text composition, val alpha; chung person adapter | C+N, R; không đổi L/14 thành HF B/32 |
| S11 | `retrieve_qwen25vl_rerank` | First stage S6 mang CLIP cũ; stale inheritance | Kế thừa S6 HF; kiểm tra val artifacts/source assets; Qwen verifier, shortlist reranking và val grids giữ nguyên | C+N, R; cần S6 + native verifier |
| S12 | `siglip2_compose` | Native SigLIP2 khác CLIP, đúng bản chất; chưa có full inference evidence | Giữ SigLIP2 SO400M 384/64 tokens/native composition, alpha 0.5 predeclared; chỉ provenance chung | N, R; không ép sang CLIP |
| S13 | `qwen_caption_to_caption` | Cache caption/text chưa khóa source implementation/runtime đầy đủ | Giữ Qwen caption/target prompt và MPNet native pooling/384 tokens | N, R; cần caption + text encoder thật |
| S14 | `structured_caption_retrieval` | Person/structured-caption caches cần image/runtime binding | Person adapter chung; giữ Qwen structured fields, MPNet và native text score; Partial SetMatch chung | C+N, R; cần actual person-caption inference |
| S15 | `gme_qwen2_vl_2b` | Native fused multimodal encoder, không thuộc shared CLIP | Giữ GME Qwen2-VL 2B, native fused forward/token budgets/final-token pooling | N, R; không ép sang CLIP |
| P1 | `word4per_setmatch` | Stage-2 REPRODUCED chưa bắt receipt/inventory; checkpoint identity thiếu content SHA | Giữ B/16/mapper/native pseudo-token; bắt exact Stage-2+config+source/Stage-1 receipt, external CUHK/ITCPR inventory và no-overlap RCR val/test bằng bytes | C+N, R; thiếu reproduced artifacts/receipt hoặc Stage-1+external data+GPU |
| P2 | `fafa_setmatch` | Geometry đã chung nhưng thiếu image/runtime binding; checkpoint identity chưa đầy đủ | FAFA squarepad, native resize, 128 words/tokens, FDA top-k và weights giữ nguyên; person adapter/cache chung | C+N, R; actual FAFA weights/native pairwise output chưa chạy |
| P3 | `instruct_reid_setmatch` | Unmatched `-1e6` chưa calibration; native distance không có cosine bound | Giữ PASS dual-attention/BERT/preprocessing/native negative squared Euclidean; chọn penalty trên val theo grid quantiles, freeze receipt trước test | C+N, R; val glue/contracts qua; chưa native checkpoint calibration/inference |
| P4 | `pic2word` | Native pseudo-token L/14 khác common B/32 có chủ đích; provenance cần khóa | Giữ author CLIP/pseudo-token mapper/prompt/tokenizer/native preprocessing; chỉ fingerprint/runtime guard | N, R; thiếu released weights native run |
| P5 | `lincir` | Native encoder/mapper riêng; cache runtime chưa đầy đủ | Giữ encoder, `$` token, native composition/checkpoint; chỉ provenance | N, R; chưa full native run |
| P6 | `magiclens_large` | Native JAX/Flax khác torch/CLIP pipeline do benchmark sở hữu | Giữ converted official JAX/Flax weights, full-scene joint forward/tokenizer/normalized output | N, R; cần native JAX GPU runtime và checkpoint |
| P7 | `bge_vl_mllm_s1` | Native BGE LLaVA embedding không phải common CLIP | Giữ pinned remote code, q/c role prompts, 4096-d embedding/native image preprocessing | N, R; cần native sharded model inference |
| P8 | `basic_setmatch` | Native polynomial pairwise score không phải cosine | Chung adapter; giữ official BASIC preset/resources/parity check; giữ analytic unmatched bound từ external resources | C+N, R; cần native resources/model parity thực |
| P9 | `adafocal_setmatch` | Full-scene+anchor query khác crop-only có chủ đích; provenance thiếu runtime | Giữ full reference, predicted anchor, TargetPad, native max-token similarity và weights; shared selection/matching | C+N, R; cần AdaFocal native inference |
| P10 | `imagine_seek` | Nhiều generation/proxy cache và isolated runtime cần khóa | Giữ BLIP2/Qwen32-GPTQ/MIGC/ELITE/LDRE-L/native OpenAI L/14 retrieval, generation recipe và proxy aggregation; khóa từng job/runtime | N, R; cần đúng giant mounted assets, native GPU/runtime; full-scene mask là adapter đã khai báo |
| P11 | `wiser` | Native OpenCLIP LAION B32 không phải Proposed HF; actual weights/API alias chưa bound | Giữ native OpenCLIP/BLIP2/BAGEL/Qwen/prompts/loops/fusion; lưu exact state SHA, raw responses/request SHA và resolved API model; chặn mixed cohorts | N, R; thiếu native assets/resources/API key; 80GiB là budget, không kết luận GPU minimum |
| P12 | `searle_xl` | Native Phi/raw CLIP features/TargetPad không thuộc common scene CLIP | Giữ author L/14/Phi, raw features vào mapper, `$` pseudo-token và TargetPad 1.25 | N, R; optional author forward chưa chạy do thiếu source/mapper |
| P13 | `context_i2w` | Custom vocabulary khác stock OpenAI/HF tokenizer có chủ đích | Giữ `<\|replace\|>` token 49405, native context/patch mapper và weights | N, R; thiếu author source/dependencies/weights |
| P14 | `fti4cir` | Serialized IMG2TEXT và subject/attribute token replacement đặc thù | Giữ author L/14, native fp32, TargetPad 1.25 và placeholder count/native token replacement | N, R; thiếu author serialized mapper/source |

## Kiểm tra bổ sung đường chạy

Rà soát gói trước còn phát hiện thiếu `execution` trong metadata cuối của S8,
S9 và Pic2Word. Đã bổ sung và kiểm tra đủ 29 đường ghi metadata, bao gồm các
phương pháp gọi shared runner. Runner kiểm tra runtime identity trước khi cấp
completed receipt; publication checker đối chiếu receipt với runtime đã ghi.
Cache scores S6 khóa thêm gallery source fingerprint. Cache val-proxy S9 kiểm
tra native generator marker, fingerprint edit, manifest/byte ảnh sinh ra và
RCR scene-index fingerprint trước khi dùng lại. Thay checkpoint/ảnh/index
làm cache mất hiệu lực mà không đổi generation recipe hay công thức fusion.

## Person adapter và GROUP

Một implementation duy nhất trong `cpr_person`: original float32 pixel `xyxy`,
original detection order, PIL float crop/zero padding, stored RGB không EXIF
transpose, one flagged full-image fallback khi detector rỗng và padding fallback
khi ít candidate hơn số Subject. Chỉ selector OpenAI B/32 của RCR FAFA được dùng
cho chọn người; không dùng GT, cardinality GT hoặc learned Proposed localizer.
Hungarian anchors rồi extras disjoint theo threshold 0.20, margin 0.03,
cap **5 người/Subject**, case agnostic. GROUP và DUAL/RELATIONAL dùng cùng rule.
Có kiểm thử native packed-cache projection, crop, padded fallback, one-to-many,
cap, negative score và `K>N` đối chiếu code RCR.

Cap 5 giữ nguyên từ RCR; val có một GROUP gồm 6 identities. Test cũng có một
trường hợp, chỉ được ghi nhận để mô tả hạn chế. Không dùng test để tăng cap.
Muốn khảo sát cap khác phải chọn bằng val rồi áp dụng đồng loạt và chạy test mới.

Partial SetMatch bắt buộc `min(K,N)` real pairs, không optional null matching;
score là `(sum(real pairs)+(K-min(K,N))*b)/K`. Giữ native score space của từng model.

## Dữ liệu, cache và công bằng cho paper

Val: 1,799 queries/5,684 gallery; test: 2,307/7,868. Gallery order và reviewed
query order không đổi. Full positives là reviewed positives chiếu vào đúng split;
ID positives phải chứa **mọi** required identities. Bỏ reference, stable ties theo
canonical order, AP lấy toàn bộ positives làm mẫu số; R@K là hit/success, không
fraction of positives. Macro-Full-mAP là trung bình đều bốn case hiện diện.

Các chỉ số evaluator đã so với RCR trên synthetic full rankings của **toàn bộ
val và test**. Đây không phải model accuracy. Test GT chỉ đi vào evaluator;
labels trên val chỉ đi vào selector tham số đã khai báo. Model adapters không
được tra positive IDs, identity IDs, seed target hay case label để suy luận.

HF scene assets, byte ảnh, index, checkpoint, config, source code, preprocessing,
adapter, runtime/package inventory và các artifacts dependency được fingerprint.
`rcr-packed-boxes-v5` và result protocol mới vô hiệu hóa kết quả cũ. Không sửa
metadata của score cũ để ép nó vào bảng mới. Read-only mounts chỉ đọc;
checkpoint metadata và mọi output nằm trong working checkout.

Hai repo chưa chứng minh paper-ready theo nghĩa full native 29 inference.
Shared protocol hỗ trợ so sánh transfer systems với adapter tường minh; nó không
làm equalize backbone capacity, external training data, compute, native token
budgets hoặc supervision với Proposed đã train. Word4Per phải ghi REPRODUCED;
Instruct penalty/fusion tuning ghi Val only; giữ native truncation/fallback limits.
Cần thực nghiệm thật với đủ checkpoints, ảnh, GPU, API và báo runtime/failure counts.
`check_publication.py` kiểm tra tính nhất quán artifact full runs; không thay thế
independent native fidelity validation hay external pretraining-overlap audit.

Xem [test evidence](additional_validation.md) và [lệnh Kaggle](kaggle.md).
