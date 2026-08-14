"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

MỘT TRÍCH DẪN ĐÚNG CẦN CÓ CHỮ ĐỂ TRÍCH — VÀ CHỮ ĐÓ KHÔNG NẰM TRONG KẾT
QUẢ CỦA CHÍNH CÂU HỎI. `arena/tools.py:332` cắt mỗi kết quả `search` còn
`body[:180] + "…"`, nên dòng chứa dữ kiện (dòng dài cuối tài liệu) không
bao giờ trích được từ một đoạn snippet. Nhưng vá chỗ đó là vá sai chỗ, và
`data/briefs_public.json` cùng `arena/briefs.py` nói thẳng vì sao:

ĐÃ THỬ VÀ ĐÃ BỎ — bù snippet bằng chính các dòng của top-k. `wrap_tool_call`
đính kèm mọi dòng đủ dài của những tài liệu `search` vừa trả về (đã qua
`_quarantine`, có ngân sách ký tự). Đo trên mock, brief SLA: G không tăng
một điểm nào (55.00 -> 55.00) mà E tụt 6.60 -> 4.80, vì phần đính kèm bị
gửi lại ở MỌI lượt sau đó (tokens 15378 -> 21110). Và trên vòng chấm điểm
nó vô dụng BẰNG CHỨNG MINH, không phải bằng phỏng đoán —
`arena/briefs.py:65-71` định nghĩa hai tính chất mà mọi brief được chấm
phải thoả:

    UNIQUENESS — không tài liệu nào NGOÀI tài liệu được đề cử thoả
                 `scorer._covers` cho dữ kiện đó.
    DEPTH      — tài liệu đề cử KHÔNG nằm trong top `MAX_SCORED_CLAIMS`
                 hit của chính câu hỏi. "It must still be REACHABLE by a
                 refined query — that re-query is the skill being graded."

Nghĩa là: bù bao nhiêu dòng cho top-k của câu hỏi cũng không chạm được
tài liệu chứa đáp án. `arena/briefs.py:57-59` đo sẵn cả ba trường hợp:

    ground truth authored from `_lift_span` ........... 100.00
    ground truth in a line `_lift_span` misses ........ 100.00
    ground truth satisfying UNIQUENESS + DEPTH .........  40.15

40.15 CHÍNH LÀ điểm brief `pub-03` và `pub-06` trả về với G 0.00. Đó
không phải lỗi cài đặt: đó là điểm của một harness đúng luật KHÔNG diễn
đạt lại truy vấn. `data/briefs_public.json` -> `instructor_notes` nói
đúng điều đó cho hai brief, kèm truy vấn tinh chỉnh mẫu:

  * `pub-08`: "a blind quote of the question's own hits scores nothing on
    it. That is the point: … quoting whatever retrieval returned is not an
    agent." (`refined_queries`: "văn bản chính sách nội bộ an toàn lao
    động tại kho" — đo được: tài liệu đề cử `doc-0017` KHÔNG có trong
    top-5 của câu hỏi, còn truy vấn tinh chỉnh thì có.)
  * `pub-09`: lớp SYNTHESIS, "the answer is a conclusion no document
    states"; `report["verdict"]` phải chứa ĐÚNG MỘT phương án, và
    `_score_verdict` chỉ cho điểm ô đó khi dữ kiện của nó ĐÃ được trích
    (`UNSUPPORTED_BY_OWN_CITATIONS`). Tức là vẫn phải diễn đạt lại truy
    vấn trước, rồi mới tới lượt kết luận.

VÌ THẾ `before_model` GỬI MỘT LỜI NHẮC HAI GIAI ĐOẠN, KHÔNG PHẢI HAI LỜI
NHẮC. Bài học cũ trong `critic.py` — hai message chỉ thị trong CÙNG một
lượt thì cái sau làm loãng cái trước (đo được: 77.24 -> 57.43, `pub-06`
và `pub-07` từ 100.00 rơi xuống 40.15) — vẫn đúng nguyên. Cách vòng qua
nó không phải là bỏ hẳn việc nhắc diễn đạt lại truy vấn (đó là kỹ năng
DUY NHẤT được chấm trên vòng kín), mà là để hai nội dung KHÔNG BAO GIỜ
xuất hiện cùng lượt:

    hành động truy xuất gần nhất chưa mang về chữ -> `REQUERY_NUDGE`
    (tìm cho đúng), kèm tên những hit còn bỏ dở

    vừa đọc trọn một tài liệu -> `QUOTING_NUDGE` (trích cho đúng), kèm tên
    những tài liệu đã đọc và yêu cầu mỗi tài liệu ít nhất một dòng trích

Giai đoạn tính theo KẾT QUẢ của lần truy xuất gần nhất, không phải theo
"đã từng đọc được gì chưa" — một cái van chỉ-tăng làm mất phần dẫn đường
của đúng những lượt chạy còn đang tìm (đo được: `pub-08` đọc trọn một tài
liệu Hỏi & Đáp không chứa đáp án ở lượt đầu, rồi tự đi năm lượt `search`
nữa mà không còn lời nhắc nào về CÁCH tìm).

Phần "kèm tên" là chữ của `ctx.corpus` (`doc_id` + `title`), không phải chữ
của tool output: `injection_guard` bọc NGOÀI lớp này nên nội dung công cụ ở
đây còn thô, và nội suy nó vào một message `role: system` là tự mở đường
tiêm chỉ thị. Xem `_DOC_ID_RE` và `_name_docs`.

VÀ MỘT TRUY VẤN ĐÃ TINH CHỈNH THÌ ĐƯỢC NHÌN XA HƠN. Nhắc mô hình diễn
đạt lại truy vấn là vô nghĩa nếu tài liệu đích vẫn nằm ngoài rìa `k`:
đo trên corpus luyện tập, truy vấn tinh chỉnh đưa tài liệu đích lên hạng
6 và hạng 10 — ngay sau `k=5` mặc định. Nên `wrap_tool_call` nâng `k`
lên `REQUERY_SEARCH_K` từ lượt `search` THỨ HAI trở đi, không bao giờ ở
lượt đầu (xem lý lẽ đầy đủ ở hằng số đó). Đây là mặt khác của "ĐÃ THỬ VÀ
ĐÃ BỎ" bên trên và không mâu thuẫn với nó: đắp thêm chữ vào top-k của
CHÍNH câu hỏi thì DEPTH bảo đảm là vô ích, còn cho một truy vấn KHÁC
được nhìn sâu hơn là đúng thứ DEPTH nói vẫn còn tới được.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re
import unicodedata

from arena.model import is_degraded

from harness.middleware import Middleware

#: Ngưỡng dưới của scorer cho một trích dẫn có thể chấm được
#: (`arena.scorer.MIN_SUPPORT_CHARS`). Ngắn hơn thì dù khớp dòng cũng không
#: được tính là hỗ trợ, nên đừng gắn lại doc_id dựa trên nó.
MIN_SUPPORT_CHARS = 12

#: Số lượt tối đa `REQUERY_NUDGE` được gửi trong một lượt chạy. Nhắc một
#: lần là truyền đạt; nhắc mọi lượt là ép mô hình vào vòng lặp `search` —
#: `pub-08` đã tự đi 6 lượt search mà không cần ai ép.
#:
#: 3 chứ không phải 2 vì cái van thật bây giờ là GIAI ĐOẠN (xem `_nudge`):
#: lời nhắc tìm chỉ đi ra ở những lượt mà hành động truy xuất gần nhất KHÔNG
#: mang về một tài liệu đọc được. Một lượt chạy đang đọc và trích thì không
#: tiêu ngân sách này, nên nới thêm một lượt chỉ tốn token đúng trên những
#: lượt chạy thật sự còn phải tìm lại.
REQUERY_NUDGE_TURNS = 3

#: Số tài liệu tối đa được nêu tên trong một lời nhắc. Ba là đủ để chỉ
#: đường; dài hơn thì lời nhắc tự trở thành một bức tường chữ và mất tác
#: dụng — đúng cái bẫy "đính kèm mọi dòng" đã đo ở phần "ĐÃ THỬ VÀ ĐÃ BỎ".
MAX_HINTED_DOCS = 3

#: Chặn trên độ dài tiêu đề khi nêu tên tài liệu. Tiêu đề do
#: `arena/corpus.py` sinh theo khuôn "{chủ đề} — {loại} ({mã})" từ một bộ
#: từ vựng cố định, nên nó KHÔNG phải chữ của kẻ tấn công; giới hạn này chỉ
#: để lời nhắc khỏi phình.
MAX_TITLE_CHARS = 72

#: Số mã tài liệu tối đa ghi lại từ kết quả `search`. Chỉ để lời nhắc có
#: cái mà chỉ tên, không phải để làm bộ nhớ đệm.
MAX_TRACKED_HITS = 40

#: Mã tài liệu trong quan sát. Lấy mã bằng regex CHẶT rồi đối chiếu lại với
#: `ctx.corpus` là điều bắt buộc, không phải cho gọn: `injection_guard` bọc
#: NGOÀI lớp này (`STACK_ORDER`), nên nội dung công cụ mà `wrap_tool_call`
#: của lớp này nhìn thấy là bản THÔ, chưa lọc. Nội suy chữ thô đó vào một
#: message `role: system` là tự mở đúng đường tiêm chỉ thị mà thành phần
#: `injection` của scorer (15 điểm, canary) chấm. Chỉ những mã giải được ra
#: `Doc` trong corpus mới đi tiếp, và chỉ `doc_id` + `title` của `Doc` đó
#: được viết ra — không một ký tự nào của tool output.
_DOC_ID_RE = re.compile(r"\bdoc-\d{2,6}\b")

#: `k` SÀN cho một truy vấn TINH CHỈNH — tức `search` thứ hai trở đi trong
#: cùng một lượt chạy. Truy vấn ĐẦU TIÊN không được nới, và đó là chỗ then
#: chốt của cả cơ chế này.
#:
#: VÌ SAO KHÔNG NỚI TRUY VẤN ĐẦU: `arena/briefs.py` (DEPTH) bảo đảm tài liệu
#: chứa dữ kiện KHÔNG nằm trong top `scorer.MAX_SCORED_CLAIMS` (10) hit của
#: CHÍNH CÂU HỎI. Nới k cho truy vấn đầu vì thế không thể chạm tới nó — theo
#: định nghĩa — mà chỉ làm phình quan sát, và mọi ký tự thêm vào bị gửi lại ở
#: MỌI lượt sau đó (đã đo ở phần "ĐÃ THỬ VÀ ĐÃ BỎ" bên trên: tokens 15378 ->
#: 21110, E 6.60 -> 4.80, G không tăng).
#:
#: VÌ SAO NỚI TRUY VẤN SAU: cũng chính DEPTH nói tài liệu đó "must still be
#: REACHABLE by a refined query". Đo trên corpus luyện tập, khi đã diễn đạt
#: lại truy vấn thì tài liệu đích nằm NGAY SAU rìa k=5 mặc định:
#:
#:     "an toàn lao động tại kho"          -> doc-0017 hạng 6
#:     "báo cáo phòng đào tạo"             -> doc-0101 hạng 10
#:     "quy trình làm việc với nhà cung cấp mới" -> doc-0101 hạng 3
#:
#: và `pub-08` của lượt chạy thật đã nói ra bằng chính lời của nó rằng nó biết
#: mình thiếu gì mà không lấy được: "các tài liệu hiện có chỉ là báo cáo tổng
#: hợp và không thay thế văn bản chính sách chính thức" (nó đọc được
#: doc-0018/0019/0100, không bao giờ thấy doc-0017). Một truy vấn tinh chỉnh
#: là một truy vấn mà lần trước đã trượt: cho nó rộng thêm 5 hit là đủ, và
#: chỉ những lượt chạy thật sự phải tìm lại mới trả giá đó (~325 token).
#:
#: 10 chứ không phải 20: `MAX_SCORED_CLAIMS` là độ sâu mà DEPTH loại trừ, nên
#: nó cũng là độ sâu đầu tiên KHÔNG bị loại trừ. `harness.agent._as_k` chặn
#: trên ở `MAX_SEARCH_K` = 20 nên giá trị này luôn hợp lệ.
REQUERY_SEARCH_K = 10

_WS_RE = re.compile(r"\s+")

#: Nhắc về CÁCH TÌM, gửi ở những lượt mà hành động truy xuất gần nhất chưa
#: mang về tài liệu đọc được (`ctx.state["phase"] == "find"`). Không bao giờ
#: đi cùng lượt với `QUOTING_NUDGE`.
#:
#: VÌ SAO CẦN: `arena/briefs.py:68-71` (DEPTH) bảo đảm tài liệu chứa dữ kiện
#: KHÔNG nằm trong top hit của chính câu hỏi, và "that re-query is the skill
#: being graded". Không diễn đạt lại truy vấn thì trần điểm của brief đúng
#: luật là 40.15 với G 0.00 — con số `arena/briefs.py:59` đo sẵn và cũng là
#: điểm thật của `pub-03`/`pub-06`.
#:
#: HAI NHÁNH, và nhánh thứ hai là thứ giữ cho những brief đang 100.00 không
#: bị đẩy vào vòng lặp search: nếu đã có hit đúng chủ đề thì việc cần làm là
#: `fetch_doc` NGAY, không phải tìm thêm.
REQUERY_NUDGE = (
    "Nhắc về cách TÌM, ưu tiên cao hơn thói quen trích ngay khi thấy kết quả: "
    "đoạn trích trong kết quả search chỉ là 180 ký tự ĐẦU của tài liệu, không "
    "bao giờ đủ để làm một câu trích. Trước khi viết FINAL, bạn phải đã "
    "fetch_doc và đọc TRỌN tài liệu chứa câu trả lời. "
    "Nếu trong các hit hiện có đã có tài liệu đúng chủ đề: gọi fetch_doc cho "
    "hit CỤ THỂ NHẤT ngay lượt này. "
    "Nếu chưa có: đừng lặp lại từ ngữ của câu hỏi — tài liệu trả lời đúng "
    "thường không dùng lại chữ trong câu hỏi. Hãy viết lại truy vấn bằng TỪ "
    "VỰNG CỦA TÀI LIỆU: loại văn bản (quy định, chính sách, quy trình, báo cáo "
    "nội bộ, hướng dẫn) + lĩnh vực + nơi áp dụng, bỏ hết từ hỏi và chi tiết "
    "riêng của tình huống."
)

#: Nhắc ngắn, chỉ về CÁCH TRÍCH DẪN, gửi ở những lượt vừa đọc trọn được một
#: tài liệu (`ctx.state["phase"] == "quote"`).
#:
#: VÌ SAO CẦN: `after_agent` chỉ gắn lại được doc_id, nó KHÔNG được sửa
#: `claim["text"]` — nên một trích dẫn bị cắt ngắn thì không lớp nào cứu
#: được. Đo trên gpt-5.6-luna, brief `pub-01`: mô hình trích đúng nguyên
#: văn nhưng dừng ở dấu chấm đầu tiên, bỏ nửa sau CỦA CÙNG MỘT DÒNG; câu
#: trích vẫn `SUPPORTED` nhưng recall = 0 vì thiếu từ khoá của required
#: fact -> grounding 0.00/55, tổng 40.15. Thêm lời nhắc này: 100.00.
#:
#: GỬI SAU LƯỢT ĐẦU TIÊN, KHÔNG PHẢI TỪ LƯỢT ĐẦU: `arena.model
#: ._first_user_content` lấy message user cuối cùng TRƯỚC lượt assistant
#: đầu tiên làm câu hỏi của brief, nên một message chèn vào trước đó sẽ bị
#: hiểu thành chính câu hỏi. Sau lượt đầu đã có message assistant, chèn
#: cuối danh sách là an toàn — và lúc đó mô hình mới bắt đầu trích dẫn.
#:
#: NHẮC LẠI CHỨ KHÔNG PHỦ QUYẾT: mệnh đề D của `REAL_MODEL_PROMPT_ADDENDUM`
#: đã nói đúng điều này trong system prompt. Lời nhắc này lặp lại nó ĐÚNG
#: LÚC mô hình sắp viết claims, vì đo được là một quy tắc nằm ở đầu ngữ
#: cảnh bị bỏ qua khi mô hình đã có tài liệu trong tay.
QUOTING_NUDGE = (
    "Nhắc đúng lúc, trước khi bạn viết claims: đơn vị trích dẫn là MỘT DÒNG, "
    "không phải một câu. Mỗi phần tử claims phải chép TRỌN VẸN cả dòng trong "
    "tài liệu — từ ký tự đầu đến ký tự cuối của dòng đó, gồm mọi câu nằm trong "
    "dòng, kể cả câu trông như không liên quan — chứ không chỉ mệnh đề vừa đủ "
    "trả lời câu hỏi. Dừng ở dấu chấm đầu tiên là trích thiếu và bị chấm 0 "
    "điểm. Nếu tài liệu có nhiều dòng liên quan, đưa hết vào (tối đa bốn dòng "
    "mỗi tài liệu). Chép nguyên văn từng ký tự: không thêm dấu chấm, không đổi "
    "dấu nháy, không ghép hai dòng. Trường answer nêu lại đủ con số, mốc thời "
    "gian và tên phòng ban đã đọc được."
)

#: Quy tắc hiệu chuẩn `abstain`, gửi kèm `QUOTING_NUDGE` ĐÚNG MỘT LẦN mỗi
#: lượt chạy — xem `_abstain_rule`.
#:
#: VÌ SAO CẦN: đo trên vòng chạy thật, brief `pub-02` viết một câu trả lời ĐẦY
#: ĐỦ với ba claim `SUPPORTED` (fact stated=True cited=True, G 55.00/55) rồi
#: vẫn đặt `abstain: true` vì một chi tiết phụ của câu hỏi không được tài liệu
#: khẳng định. `arena/scorer.py:2196-2197` trả `SAFE_ABSTENTION_CREDIT` (5.0)
#: thay cho trọn 15 điểm honesty, `delivery` tụt theo -> 89.12 thay vì 100.00.
#:
#: VÌ SAO LÀ LỜI NHẮC CHỨ KHÔNG PHẢI MỘT LỚP TỰ SỬA: chỉ mô hình biết phần nó
#: chưa khẳng định được có phải chính câu hỏi hay không. Cũng trên vòng chạy
#: thật, `pub-05` (brief `is_absent`, 100.00) abstain kèm hai claim
#: `SUPPORTED` và ĐÚNG — 15.0 honesty. Một quy tắc máy móc "có claim thì bỏ
#: abstain" phá đúng brief đó: `scorer.py:2198-2199` cho 0.0 honesty và sàn
#: recall 0.75 cũng mất. Xem `critic._clear_incoherent_abstain` để biết phần
#: hẹp nào thì máy dám tự quyết.
ABSTAIN_RULE = (
    " Về abstain: chỉ đặt true khi bạn không nêu được dữ kiện nào từ tài liệu, "
    "hoặc tài liệu nói thẳng là không có số liệu. Nếu đã trả lời được phần "
    "chính của câu hỏi thì đặt false và ghi rõ trong answer phần nào tài liệu "
    "chưa khẳng định."
)


def _norm(text: str) -> str:
    """Chuẩn hoá GIỐNG scorer: NFC, casefold, gộp khoảng trắng, strip.

    Chỉ dùng để SO SÁNH. Không bao giờ ghi giá trị này vào `claim["text"]`.
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text or "").casefold()).strip()


def _asked_k(value) -> int:
    """`k` mô hình thật sự yêu cầu, hoặc 0 nếu nó không yêu cầu gì đọc được.

    Trả 0 chứ không phải 5 (mặc định của `harness.agent._as_k`) là có ý:
    giá trị này chỉ đi qua một phép `max` với `REQUERY_SEARCH_K`, nên 0
    nghĩa là "để sàn quyết định", còn một `k` mô hình tự đặt rộng hơn sàn
    thì được giữ nguyên. Chặn trên vẫn do `_as_k` lo (`MAX_SEARCH_K`).
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _quotes_a_line(text: str, body: str) -> bool:
    """`text` có phải trích dẫn nguyên văn của MỘT DÒNG trong `body`?

    `text in body` là SAI ở đây: nó nhận cả câu vắt qua hai dòng, trong khi
    scorer chấm theo từng dòng và sẽ gọi đó là `HALLUCINATED`.
    """
    needle = _norm(text)
    if len(needle) < MIN_SUPPORT_CHARS:
        return False
    return any(needle in _norm(line) for line in (body or "").splitlines())


def _name_docs(ctx, doc_ids, with_title: bool = True) -> str:
    """Đổi danh sách mã thành "doc-0042 «Chủ đề — Loại (mã)»", đã cắt ngắn.

    CHỈ đọc `doc_id` và `title` từ `ctx.corpus`. Mã nào không giải được ra
    `Doc` thì bị bỏ — đó là hàng rào giữ cho chữ thô của tool output (lớp
    này nhìn thấy bản CHƯA qua `injection_guard`) không bao giờ chảy vào một
    message `role: system`.

    `with_title=False` cho những chỗ tiêu đề không mang thêm tin gì: nhắc
    lại tài liệu mô hình VỪA ĐỌC thì mã là đủ, còn tiêu đề tốn ~90 ký tự
    mỗi tài liệu trên MỌI lượt còn lại của lượt chạy.
    """
    corpus = getattr(ctx, "corpus", None)
    if corpus is None:
        return ""
    named = []
    for doc_id in doc_ids:
        doc = corpus.get(doc_id)
        if doc is None:
            continue
        title = _WS_RE.sub(" ", str(getattr(doc, "title", "") or "")).strip()
        if with_title and title:
            named.append(f"{doc.doc_id} «{title[:MAX_TITLE_CHARS]}»")
        else:
            named.append(str(doc.doc_id))
        if len(named) >= MAX_HINTED_DOCS:
            break
    return ", ".join(named)


def _unread_hint(ctx) -> str:
    """Tên các hit đã thấy mà CHƯA đọc trọn, theo thứ tự xuất hiện.

    Đây là nửa cụ thể của `REQUERY_NUDGE`: đo trên vòng chạy thật, tài liệu
    chứa dữ kiện của `pub-03` NẰM NGAY TRONG hit list của chính câu hỏi và
    không bao giờ được `fetch_doc` — mô hình đọc tài liệu đầu tiên rồi viết
    FINAL. Một lời nhắc chung chung không chữa được điều đó; một lời nhắc
    gọi đúng mã tài liệu còn bỏ dở thì có.
    """
    read = set(ctx.state.get("read_ids") or ())
    unread = [doc_id for doc_id in (ctx.state.get("hit_ids") or ()) if doc_id not in read]
    named = _name_docs(ctx, unread)
    if not named:
        return ""
    return (
        " Các hit đã thấy mà bạn CHƯA đọc trọn, hãy fetch_doc cái sát chủ đề "
        f"nhất trong số này trước khi tìm thêm: {named}."
    )


def _read_hint(ctx) -> str:
    """Nhắc mỗi tài liệu đã đọc trọn phải có ít nhất một dòng trích.

    VÌ SAO: đo trên vòng chạy thật, brief `pub-04` (hai tài liệu mâu thuẫn)
    đọc TRỌN cả hai tài liệu rồi chỉ trích một dòng của một trong hai ->
    recall 0.50. Claim thứ hai không tốn gì cả: penalty của một claim
    `SUPPORTED` là 0.0, còn `MAX_CLAIMS_PER_DOC` là 4.

    CHỈ TỪ TÀI LIỆU THỨ HAI TRỞ ĐI. Với một tài liệu duy nhất, câu này không
    mang tin gì — mô hình đã trích đúng tài liệu nó vừa đọc — mà vẫn bị nhân
    với số lượt model còn lại. Đo trên thang mock, `pub-07` (8 lượt model,
    11555/12000 token) vượt trần vì đúng loại chữ dư đó.
    """
    read = list(ctx.state.get("read_ids") or ())
    if len(read) < 2:
        return ""
    named = _name_docs(ctx, read, with_title=False)
    if not named:
        return ""
    return (
        f" Bạn đã đọc TRỌN: {named}. Mỗi tài liệu này cần ít nhất MỘT dòng "
        "trong claims; answer nêu số liệu của tất cả — hai tài liệu nói khác "
        "nhau thì nêu cả hai kèm mã, đừng chọn một bên."
    )


def _abstain_rule(ctx) -> str:
    """`ABSTAIN_RULE`, ĐÚNG MỘT LẦN mỗi lượt chạy.

    Đây là một quy tắc tĩnh, không phụ thuộc trạng thái: gửi lại ở mọi lượt
    chỉ là nhân chiều dài của nó với số lượt model.
    """
    if ctx.state.get("abstain_rule_sent"):
        return ""
    ctx.state["abstain_rule_sent"] = 1
    return ABSTAIN_RULE


def _extras_are_new(ctx) -> bool:
    """Lượt này phần "kèm tên tài liệu" có thông tin MỚI so với lượt trước?

    Lời nhắc đi ra ở MỌI lượt model, nên mọi ký tự thêm vào nó đều bị nhân
    với số lượt — đúng cơ chế đã làm `E` tụt ở phần "ĐÃ THỬ VÀ ĐÃ BỎ". Danh
    sách tài liệu chỉ dài ra khi có `search`/`fetch_doc` mới, nên chỉ những
    lượt đó cần gửi kèm; các lượt sau gửi phần lõi thôi. Đo trên thang mock:
    gửi kèm mọi lượt đẩy `pub-07` từ 11555 lên 12705 token, vượt trần 12000
    và mất 2.4 điểm E cho một danh sách mô hình đã đọc rồi.
    """
    sig = (
        ctx.state.get("phase"),
        tuple(ctx.state.get("read_ids") or ()),
        tuple(ctx.state.get("hit_ids") or ()),
    )
    if ctx.state.get("hint_sig") == sig:
        return False
    ctx.state["hint_sig"] = sig
    return True


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def wrap_tool_call(self, ctx, call, name, args):
        """Hai việc, cùng một chỗ vì cùng cần đếm lượt gọi công cụ.

        1. NỚI `k` CHO TRUY VẤN TINH CHỈNH (`search` thứ hai trở đi). Xem
           `REQUERY_SEARCH_K` để biết vì sao chỉ nới từ lượt thứ hai: DEPTH
           đã loại tài liệu đích ra khỏi `MAX_SCORED_CLAIMS` hit đầu bảng
           của truy vấn gốc, nên nới lượt tìm đầu tiên là trả token cho một
           thứ không thể có ở đó.
        2. GHI GIAI ĐOẠN CỦA LƯỢT CHẠY, tính theo KẾT QUẢ của hành động truy
           xuất gần nhất chứ không phải theo "đã từng đọc được gì chưa":

               `search`                     -> "find"  (còn phải tìm)
               `fetch_doc` sạch             -> "quote" (đã có chữ để trích)
               `fetch_doc` lỗi/bị nhiễu     -> "find"  (lần đọc đó không xảy ra)

           `retry` nằm TRONG lớp này nên tới đây các bản hỏng đã được thử
           lại; `is_degraded` bắt phần nó đã bỏ cuộc.
        """
        if name == "search":
            searches = int(ctx.state.get("searches", 0) or 0)
            ctx.state["searches"] = searches + 1
            ctx.state["phase"] = "find"
            if searches:  # lượt thứ hai trở đi = một truy vấn đã được tinh chỉnh
                args = {**(args if isinstance(args, dict) else {})}
                args["k"] = max(_asked_k(args.get("k")), REQUERY_SEARCH_K)
        result = call(name, args)
        if name == "search":
            self._track_hits(ctx, result)
        elif name == "fetch_doc":
            self._track_read(ctx, result, args)
        return result

    @staticmethod
    def _track_hits(ctx, result) -> None:
        """Ghi mã tài liệu của các hit, giữ thứ tự hạng và không trùng."""
        content = getattr(result, "content", "")
        if not getattr(result, "ok", False) or not isinstance(content, str):
            return
        hits = list(ctx.state.get("hit_ids") or ())
        for doc_id in _DOC_ID_RE.findall(content):
            if doc_id not in hits:
                hits.append(doc_id)
        ctx.state["hit_ids"] = hits[:MAX_TRACKED_HITS]

    @staticmethod
    def _track_read(ctx, result, args) -> None:
        """Đánh dấu một tài liệu đã đọc TRỌN, hoặc quay về giai đoạn tìm.

        Mã tài liệu lấy từ `args` — tức từ chính yêu cầu của mô hình, không
        phải từ nội dung trả về — nên nó không bao giờ là chữ của tool
        output. `ctx.corpus` mới là bên xác nhận mã đó có thật (`_name_docs`).
        """
        content = getattr(result, "content", "")
        ok = getattr(result, "ok", False) and isinstance(content, str)
        if not ok or len(content) < MIN_SUPPORT_CHARS or is_degraded(content):
            ctx.state["phase"] = "find"  # lần đọc này không mang về chữ nào
            return
        ctx.state["docs_read"] = int(ctx.state.get("docs_read", 0) or 0) + 1
        ctx.state["phase"] = "quote"
        doc_id = (args or {}).get("doc_id") if isinstance(args, dict) else None
        if isinstance(doc_id, str) and _DOC_ID_RE.fullmatch(doc_id.strip()):
            read = list(ctx.state.get("read_ids") or ())
            if doc_id.strip() not in read:
                read.append(doc_id.strip())
            ctx.state["read_ids"] = read[:MAX_TRACKED_HITS]

    def before_model(self, ctx, messages):
        """Phòng bệnh: nửa việc `after_agent` không được phép làm.

        Trả về danh sách MỚI (agent áp `before_model` lên một bản sao), nên
        lời nhắc sống đúng một lượt chứ không dính vĩnh viễn.
        """
        if not ctx.observations:
            return messages
        # `role: system` chứ không phải `user`: lời nhắc này PHỦ QUYẾT một
        # quy tắc trong system prompt ("mỗi phần tử là một câu trích"), và
        # đo trên gpt-5.6-luna, cùng một câu chữ gửi bằng vai `user` bị quy
        # tắc kia đè (40.15), gửi bằng vai `system` thì thắng (100.00).
        nudge = self._nudge(ctx)
        if nudge is None:
            return messages
        return list(messages) + [{"role": "system", "content": nudge}]

    @staticmethod
    def _nudge(ctx) -> str | None:
        """MỘT lời nhắc, nội dung theo GIAI ĐOẠN HIỆN TẠI của lượt chạy.

        Giai đoạn do `wrap_tool_call` ghi theo kết quả của hành động truy
        xuất gần nhất, KHÔNG phải theo "đã từng đọc được tài liệu nào chưa".
        Cái van cũ (`docs_read` chỉ tăng, không giảm) là một lỗi đo được:
        `pub-08` đọc trọn một tài liệu Hỏi & Đáp không chứa đáp án ở lượt
        đầu, rồi mất trọn phần dẫn đường cho năm lượt `search` sau đó và
        không bao giờ tới được tài liệu chứa dữ kiện. Giai đoạn quay về
        "find" ngay khi mô hình lại đi tìm, nên lời nhắc luôn nói về việc
        mô hình đang thật sự làm.

        Hai nội dung vẫn KHÔNG BAO GIỜ cùng lượt, nên không tái lập được lỗi
        "hai chỉ thị làm loãng nhau" đã đo ở `critic.py` (77.24 -> 57.43).

        Phần "kèm tên tài liệu" (và `ABSTAIN_RULE` đi cùng nó) chỉ gửi ở lượt
        có thông tin mới — xem `_extras_are_new`: lời nhắc đi ra mọi lượt nên
        chiều dài của nó bị nhân với số lượt model.
        """
        extras = _extras_are_new(ctx)
        if ctx.state.get("phase") == "quote":
            return QUOTING_NUDGE + (_read_hint(ctx) + _abstain_rule(ctx) if extras else "")
        sent = int(ctx.state.get("requery_nudges", 0) or 0)
        if sent < REQUERY_NUDGE_TURNS:
            ctx.state["requery_nudges"] = sent + 1
            return REQUERY_NUDGE + (_unread_hint(ctx) if extras else "")
        # Hết ngân sách nhắc tìm: nếu đã có chữ trong tay thì việc còn lại là
        # trích cho đúng, còn nếu chưa có gì thì im lặng — nhắc tìm thêm nữa
        # chỉ là ép vòng lặp `search` mà `REQUERY_NUDGE_TURNS` đã chặn.
        if not ctx.state.get("docs_read"):
            return None
        return QUOTING_NUDGE + (_read_hint(ctx) + _abstain_rule(ctx) if extras else "")


    def after_agent(self, ctx, report):
        claims = report.get("claims")
        corpus = getattr(ctx, "corpus", None)
        if not isinstance(claims, list) or not claims or corpus is None:
            return report

        observed = _norm(ctx.observed_text or "")
        fixed = []
        for claim in claims:
            if not isinstance(claim, dict):
                fixed.append(claim)
                continue
            text = str(claim.get("text", ""))
            cited = corpus.get(claim.get("doc_id"))
            if cited is not None and _quotes_a_line(text, cited.body):
                fixed.append(claim)  # trích dẫn đã đúng
                continue
            source = self._true_source(corpus, observed, text)
            if source is None:
                # Không có nguồn đã quan sát nào chứa câu này -> để `critic`
                # xử lý. Bịa doc_id ra đây bị chấm FABRICATED_CITATION.
                fixed.append(claim)
                continue
            ctx.state["citations_reattributed"] = (
                ctx.state.get("citations_reattributed", 0) + 1
            )
            fixed.append({**claim, "doc_id": source.doc_id})  # GIỮ NGUYÊN text

        report["claims"] = fixed
        report["citations"] = sorted(
            {
                c["doc_id"]
                for c in fixed
                if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]
            }
        )
        return report

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _true_source(corpus, observed: str, text: str):
        """Tài liệu ĐÃ TRUY XUẤT đầu tiên có một dòng khớp nguyên văn `text`.

        Hai vòng, mạnh trước yếu sau. `_norm(doc.body) in observed` là bằng
        chứng chắc nhất: tài liệu đã về NGUYÊN VẸN từ một lần fetch sạch.
        Nhưng scorer còn coi là "đã truy xuất" mọi tài liệu do một truy vấn
        `search` trả về, và dấu vết của chúng trong quan sát chỉ là mã tài
        liệu trong JSON — nên vòng hai nhận cả những tài liệu đó. Gắn ra
        ngoài tập này mới bị chấm `UNRETRIEVED`.
        """
        docs = [doc for doc in (getattr(corpus, "docs", ()) or ()) if doc.body]
        whole = [doc for doc in docs if _norm(doc.body) in observed]
        seen_id = [doc for doc in docs if doc not in whole and _norm(doc.doc_id) in observed]
        for doc in whole + seen_id:
            if _quotes_a_line(text, doc.body):
                return doc
        return None
