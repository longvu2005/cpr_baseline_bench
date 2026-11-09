# Audit 29/29 baseline đối chiếu RCR

Rà soát hiện tại: baseline `5f7a9d395f5c98aa16ac8e3f07fe57cdc5f8dfe9`, RCR
`2fe92766d5158f8dd87297d3bc4cb6280b8ed63b` (không sửa RCR), ngày 2026-11-09.
Phần "Lần rà soát trước" bên dưới mô tả lần đồng bộ từ `65a7c37` và vẫn đúng,
trừ các điểm được sửa ở đây. Giữ nguyên 29 method IDs, tên hiển thị và nhóm.
Không phương pháp nào được chạy full native trên final data trong môi trường này.

## Kiểm chứng trực tiếp với RCR (lần này)

| Hạng mục | Cách kiểm | Kết quả |
| --- | --- | --- |
| Import val/test | `prepare_data` so với `load_rcr_data`, `split_samples`, `split_image_ids`, `required_identity_ids`, `sample_selection_texts` trên toàn bộ final | Khớp tuyệt đối: thứ tự gallery/query, Full positives chiếu theo split, ID required, case, instruction, selection text; val 1,799/5,684, test 2,307/7,868 |
| Định tuyến text theo Subject | `modify_text` so với `subject_changes` của RCR FAFA trên 4,106 queries | 0 khác biệt; 160 val và 339 test giữ nguyên mệnh đề quan hệ |
| Evaluator | `evaluate_scores` so với `evaluate_rankings` của RCR, điểm nguyên có nhiều tie | Sai lệch tối đa 3.5e-18; Macro-Full-mAP bằng trung bình đều các case có mặt |
| Subject selection/GROUP | `localize_queries` so với `choose_boxes`+`select_subject_members` của RCR, có ảnh 8 người, ảnh rỗng, thiếu candidate, DUAL, cap 5 | Khớp box/fallback từng member; mutation đảo tie-break bị bắt |
| Partial SetMatch | `aggregate_people` so với RCR với K=1..10, N=0..8, tie và penalty native | Khớp bit-for-bit |
| Native algorithms | 118 thân hàm được bảo vệ trong 27 file | Không đổi, trên Python 3.12 và 3.13 |

## Lỗi tìm thấy và đã sửa (lần này)

1. **Không tái sử dụng được cache RCR gốc.** `read_scene_globals` và nhánh
   detection bắt buộc trường `image_content_sha256`, nhưng `build_cache`/
   `build_detection_cache` của RCR không ghi trường này. Với `cache/scene/index.pt`
   và `cache/detections/index.pt` thật, mọi person method và mọi scene method dùng
   `--rcr-scene-cache` đều dừng. Kịch bản Kaggle buộc chạy lại Grounding DINO và
   CLIP trên toàn bộ 37,107 ảnh để tạo bản "verified", với revision detector khác
   RCR (RCR không pin revision), nên có thể thất bại hoặc tạo box khác Proposed.
   Sửa: đọc index gốc theo đúng hợp đồng của RCR FAFA/Proposed (cache_id, thứ tự
   gallery, cấu hình, kích thước từng ảnh; byte ảnh hiện tại vào fingerprint). Scene
   globals còn được kiểm bằng cách encode lại 16 ảnh chọn theo SHA256 của ID
   (không dùng nhãn), yêu cầu cosine >= 0.999. Digest vẫn được kiểm khi có.
   Bản audit đầy đủ chuyển thành tùy chọn `CPR_VERIFY_RCR_CACHES=1`.
   `RESULT_PROTOCOL` tăng lên v6 để kết quả cũ phải chạy lại.
2. **Hướng dẫn Kaggle lỗi thời.** Tài liệu yêu cầu checkout `65a7c37` rồi apply
   patch đã nằm sẵn trong HEAD, cùng một patch RCR không có trong repo nào. Đã
   viết lại: clone HEAD, dùng RCR không sửa.
3. **Kiểm tra chữ ký native phụ thuộc phiên bản Python.** Python 3.13 đổi mặc định
   `ast.dump`, làm test báo sai rằng `load_word4per` thay đổi. Sửa test, giữ digest.
4. **CPU suite không chạy sạch khi thiếu torch** dù `requirements-dev.txt` nói
   không cần torch: 6 test import torch trực tiếp. Nay chúng skip có lý do.
5. **Test ghi đè kết quả thật.** Hai test chạy `run_baseline.py --all` với pipeline
   giả, ghi trạng thái "completed" giả vào `runs/batch_status.json` (đầu vào của
   `check_publication.py`) và xóa `runs/<reid_set_qwen_edit_proxy|retrieve_qwen25vl_rerank>/run.json`. Chạy test sau một batch
   Kaggle sẽ phá kết quả. Nay hai test dùng thư mục tạm.
6. **Rác trong repo:** `.vscode/` (task tự chạy `find … -delete` khi mở thư mục),
   `checkpoints/lincir/prepared.json` và `config.json` sinh ra khi chuẩn bị (va chạm
   với `stage_checkpoint_mount.py` khi mount có bản khác), khóa
   `cache.gallery_features` không còn được đọc trong config clip_image–clip_late_fusion.

Không thay đổi: encoder, weights, preprocessing native, tokenizer, composition,
pairwise score, unmatched penalty, lưới tuning, cap 5 và thresholds của selector.
Cap 5 giữ theo RCR FAFA; val chỉ có 1/1,799 query có Subject 6 identities.

## Lần rà soát trước (từ `65a7c37`)

### Những khác biệt trước khi sửa

- Scene CLIP do benchmark sở hữu vẫn dùng OpenAI B/32 và center crop, khác
  HF B/32 letterbox của Proposed. Khác biệt lan qua groundingdino_clipreid_set_text–reid_set_qwen_edit_proxy và retrieve_qwen25vl_rerank, không chỉ clip_image–clip_late_fusion.
  Chưa có đường đọc trực tiếp native globals trong RCR scene index.
- Cả 11 person routes đã có adapter chung và đọc packed RCR detections ở base
  commit. Geometry, fallback, Hungarian anchors và membership không cần viết lại.
  Còn thiếu ràng buộc index với byte ảnh nguồn, môi trường selector độc lập
  với native dependencies, fingerprint implementation/runtime và kiểm tra cap khi
  đọc cache. Đây là lỗ hổng provenance, không phải bằng chứng 11 thuật toán đều sai.
- Kết quả phụ thuộc groundingdino_clipreid_set/qwen_image_edit_clip→reid_set_qwen_edit_proxy và groundingdino_clipreid_set_text→retrieve_qwen25vl_rerank có kiểm tra manifest/person metadata,
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
| clip_image | `clip_image` | OpenAI center crop khác Proposed; cache riêng | Chung frozen HF B/32 letterbox; đọc trực tiếp globals RCR (spot-check); cosine giữ nguyên | C, R; actual pretrained image/cache parity đã qua, chưa full test inference |
| clip_text | `clip_text` | OpenAI tokenizer/loader, scene gallery khác Proposed | Chung HF native tokenizer/EOS/right padding 77 và raw global projection | C, R; actual pretrained text parity đã qua |
| clip_early_fusion | `clip_early_fusion` | Gallery/text encoder khác Proposed | HF chung; normalized early composition giữ nguyên; 21 weights chọn val Full-mAP | C, R; fusion/evaluator contracts qua; cần val selection + full test thật |
| clip_late_fusion | `clip_late_fusion` | Encoder khác Proposed | HF chung; z-score bỏ reference; 21 weights chọn val Full-mAP | C, R; cần val selection + full test thật |
| groundingdino_clipreid_set | `groundingdino_clipreid_set` | Person index thiếu source-image/runtime binding | Dùng chung packed boxes/selector; CLIP-ReID MSMT17 B/16, transform và cosine giữ nguyên | C+N, R; thiếu native ReID weights/data run |
| groundingdino_clipreid_set_text | `groundingdino_clipreid_set_text` | Nhánh scene CLIP sai pipeline; phụ thuộc cache groundingdino_clipreid_set | HF text/gallery chung; native ReID, score fusion và val alpha rule giữ nguyên; khóa dependency config/assets | C+N, R; cần native groundingdino_clipreid_set/val/test |
| qwen25vl_rewrite_clip | `qwen25vl_rewrite_clip` | Retriever center crop; rewrite/features thiếu runtime binding | Qwen2.5-VL 7B/prompt/generation giữ nguyên; rewrite encode bằng HF chung | C+N, R; chưa chạy native rewrite |
| qwen_image_edit_clip | `qwen_image_edit_clip` | Generated-image encoding khác Proposed; cache chưa đủ identity | Qwen-Image-Edit-2509 giữ nguyên; ảnh sinh ra và gallery cùng HF letterbox; fingerprint pixel output | C+N, R; chưa native editor inference |
| reid_set_qwen_edit_proxy | `reid_set_qwen_edit_proxy` | groundingdino_clipreid_set/qwen_image_edit_clip scores có thể stale; val proxy dùng CLIP cũ | Kiểm tra source score/config/weights/scene metadata; val proxy và test dùng cùng qwen_image_edit_clip HF pipeline; fusion giữ nguyên | C+N, R; cần hai source runs thật |
| per_person_clip_compose_setmatch | `per_person_clip_compose_setmatch` | Native L/14 khác scene B/32 nhưng có chủ đích; provenance chưa đủ | Giữ native OpenAI L/14 crop/image-text composition, val alpha; chung person adapter | C+N, R; không đổi L/14 thành HF B/32 |
| retrieve_qwen25vl_rerank | `retrieve_qwen25vl_rerank` | First stage groundingdino_clipreid_set_text mang CLIP cũ; stale inheritance | Kế thừa groundingdino_clipreid_set_text HF; kiểm tra val artifacts/source assets; Qwen verifier, shortlist reranking và val grids giữ nguyên | C+N, R; cần groundingdino_clipreid_set_text + native verifier |
| siglip2_compose | `siglip2_compose` | Native SigLIP2 khác CLIP, đúng bản chất; chưa có full inference evidence | Giữ SigLIP2 SO400M 384/64 tokens/native composition, alpha 0.5 predeclared; chỉ provenance chung | N, R; không ép sang CLIP |
| qwen_caption_to_caption | `qwen_caption_to_caption` | Cache caption/text chưa khóa source implementation/runtime đầy đủ | Giữ Qwen caption/target prompt và MPNet native pooling/384 tokens | N, R; cần caption + text encoder thật |
| structured_caption_retrieval | `structured_caption_retrieval` | Person/structured-caption caches cần image/runtime binding | Person adapter chung; giữ Qwen structured fields, MPNet và native text score; Partial SetMatch chung | C+N, R; cần actual person-caption inference |
| gme_qwen2_vl_2b | `gme_qwen2_vl_2b` | Native fused multimodal encoder, không thuộc shared CLIP | Giữ GME Qwen2-VL 2B, native fused forward/token budgets/final-token pooling | N, R; không ép sang CLIP |
| word4per_setmatch | `word4per_setmatch` | Stage-2 REPRODUCED chưa bắt receipt/inventory; checkpoint identity thiếu content SHA | Giữ B/16/mapper/native pseudo-token; bắt exact Stage-2+config+source/Stage-1 receipt, external CUHK/ITCPR inventory và no-overlap RCR val/test bằng bytes | C+N, R; thiếu reproduced artifacts/receipt hoặc Stage-1+external data+GPU |
| fafa_setmatch | `fafa_setmatch` | Geometry đã chung nhưng thiếu image/runtime binding; checkpoint identity chưa đầy đủ | FAFA squarepad, native resize, 128 words/tokens, FDA top-k và weights giữ nguyên; person adapter/cache chung | C+N, R; actual FAFA weights/native pairwise output chưa chạy |
| instruct_reid_setmatch | `instruct_reid_setmatch` | Unmatched `-1e6` chưa calibration; native distance không có cosine bound | Giữ PASS dual-attention/BERT/preprocessing/native negative squared Euclidean; chọn penalty trên val theo grid quantiles, freeze receipt trước test | C+N, R; val glue/contracts qua; chưa native checkpoint calibration/inference |
| pic2word | `pic2word` | Native pseudo-token L/14 khác common B/32 có chủ đích; provenance cần khóa | Giữ author CLIP/pseudo-token mapper/prompt/tokenizer/native preprocessing; chỉ fingerprint/runtime guard | N, R; thiếu released weights native run |
| lincir | `lincir` | Native encoder/mapper riêng; cache runtime chưa đầy đủ | Giữ encoder, `$` token, native composition/checkpoint; chỉ provenance | N, R; chưa full native run |
| magiclens_large | `magiclens_large` | Native JAX/Flax khác torch/CLIP pipeline do benchmark sở hữu | Giữ converted official JAX/Flax weights, full-scene joint forward/tokenizer/normalized output | N, R; cần native JAX GPU runtime và checkpoint |
| bge_vl_mllm_s1 | `bge_vl_mllm_s1` | Native BGE LLaVA embedding không phải common CLIP | Giữ pinned remote code, q/c role prompts, 4096-d embedding/native image preprocessing | N, R; cần native sharded model inference |
| basic_setmatch | `basic_setmatch` | Native polynomial pairwise score không phải cosine | Chung adapter; giữ official BASIC preset/resources/parity check; giữ analytic unmatched bound từ external resources | C+N, R; cần native resources/model parity thực |
| adafocal_setmatch | `adafocal_setmatch` | Full-scene+anchor query khác crop-only có chủ đích; provenance thiếu runtime | Giữ full reference, predicted anchor, TargetPad, native max-token similarity và weights; shared selection/matching | C+N, R; cần AdaFocal native inference |
| imagine_and_seek | `imagine_seek` | Nhiều generation/proxy cache và isolated runtime cần khóa | Giữ BLIP2/Qwen32-GPTQ/MIGC/ELITE/LDRE-L/native OpenAI L/14 retrieval, generation recipe và proxy aggregation; khóa từng job/runtime | N, R; cần đúng giant mounted assets, native GPU/runtime; full-scene mask là adapter đã khai báo |
| wiser | `wiser` | Native OpenCLIP LAION B32 không phải Proposed HF; actual weights/API alias chưa bound | Giữ native OpenCLIP/BLIP2/BAGEL/Qwen/prompts/loops/fusion; lưu exact state SHA, raw responses/request SHA và resolved API model; chặn mixed cohorts | N, R; thiếu native assets/resources/API key; 80GiB là budget, không kết luận GPU minimum |
| searle_xl | `searle_xl` | Native Phi/raw CLIP features/TargetPad không thuộc common scene CLIP | Giữ author L/14/Phi, raw features vào mapper, `$` pseudo-token và TargetPad 1.25 | N, R; optional author forward chưa chạy do thiếu source/mapper |
| context_i2w | `context_i2w` | Custom vocabulary khác stock OpenAI/HF tokenizer có chủ đích | Giữ `<\|replace\|>` token 49405, native context/patch mapper và weights | N, R; thiếu author source/dependencies/weights |
| fti4cir | `fti4cir` | Serialized IMG2TEXT và subject/attribute token replacement đặc thù | Giữ author L/14, native fp32, TargetPad 1.25 và placeholder count/native token replacement | N, R; thiếu author serialized mapper/source |

## Kiểm tra bổ sung đường chạy

Rà soát gói trước còn phát hiện thiếu `execution` trong metadata cuối của qwen_image_edit_clip,
reid_set_qwen_edit_proxy và Pic2Word. Đã bổ sung và kiểm tra đủ 29 đường ghi metadata, bao gồm các
phương pháp gọi shared runner. Runner kiểm tra runtime identity trước khi cấp
completed receipt; publication checker đối chiếu receipt với runtime đã ghi.
Cache scores groundingdino_clipreid_set_text khóa thêm gallery source fingerprint. Cache val-proxy reid_set_qwen_edit_proxy kiểm
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
