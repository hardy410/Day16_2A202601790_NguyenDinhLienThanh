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

    chưa đọc trọn tài liệu nào  -> `REQUERY_NUDGE` (tìm cho đúng)
    đã đọc trọn một tài liệu    -> `QUOTING_NUDGE` (trích cho đúng)

Trạng thái đó lấy từ `wrap_tool_call` của chính lớp này: một lượt
`fetch_doc` sạch là mốc chuyển giai đoạn.

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
REQUERY_NUDGE_TURNS = 2

_WS_RE = re.compile(r"\s+")

#: Nhắc về CÁCH TÌM, gửi ở giai đoạn 1 — khi chưa có tài liệu nào được đọc
#: trọn. Không bao giờ đi cùng lượt với `QUOTING_NUDGE`.
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

#: Nhắc ngắn, chỉ về CÁCH TRÍCH DẪN, gửi kèm mỗi lượt model SAU khi đã có
#: quan sát đầu tiên.
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
QUOTING_NUDGE = (
    "Nhắc về cách trích dẫn, ưu tiên cao hơn mọi quy tắc đã nêu trước đó về "
    "độ dài câu trích (kể cả quy tắc \"mỗi phần tử là một câu trích nguyên "
    "văn\"): đơn vị trích dẫn là MỘT DÒNG, không phải một câu. Mỗi phần tử "
    "claims phải chép TRỌN VẸN cả dòng trong tài liệu — từ ký tự đầu đến ký "
    "tự cuối của dòng đó, gồm mọi câu nằm trong dòng, kể cả câu trông như "
    "không liên quan — chứ không chỉ mệnh đề vừa đủ trả lời câu hỏi. Dừng ở "
    "dấu chấm đầu tiên là trích thiếu và bị chấm 0 điểm. Chép nguyên văn "
    "từng ký tự: không thêm dấu chấm, không đổi dấu nháy, không ghép hai dòng."
)


def _norm(text: str) -> str:
    """Chuẩn hoá GIỐNG scorer: NFC, casefold, gộp khoảng trắng, strip.

    Chỉ dùng để SO SÁNH. Không bao giờ ghi giá trị này vào `claim["text"]`.
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text or "").casefold()).strip()


def _quotes_a_line(text: str, body: str) -> bool:
    """`text` có phải trích dẫn nguyên văn của MỘT DÒNG trong `body`?

    `text in body` là SAI ở đây: nó nhận cả câu vắt qua hai dòng, trong khi
    scorer chấm theo từng dòng và sẽ gọi đó là `HALLUCINATED`.
    """
    needle = _norm(text)
    if len(needle) < MIN_SUPPORT_CHARS:
        return False
    return any(needle in _norm(line) for line in (body or "").splitlines())


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def wrap_tool_call(self, ctx, call, name, args):
        """Ghi mốc chuyển giai đoạn: một lượt `fetch_doc` SẠCH.

        Không sửa gì kết quả — chỉ đọc. `retry` nằm TRONG lớp này nên tới
        đây các bản hỏng đã được thử lại; `is_degraded` bắt phần nó đã bỏ
        cuộc, để một bản body bị nhiễu không bị tính là "đã đọc trọn".
        """
        result = call(name, args)
        if name == "fetch_doc" and getattr(result, "ok", False):
            content = result.content if isinstance(result.content, str) else ""
            if len(content) >= MIN_SUPPORT_CHARS and not is_degraded(content):
                ctx.state["docs_read"] = int(ctx.state.get("docs_read", 0) or 0) + 1
        return result

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
        """MỘT lời nhắc, nội dung theo giai đoạn của lượt chạy.

        Giai đoạn 1 (chưa đọc trọn tài liệu nào): tìm cho đúng. Giai đoạn 2
        (đã đọc): trích cho đúng. Hai nội dung không bao giờ cùng lượt, nên
        không tái lập được lỗi "hai chỉ thị làm loãng nhau" đã đo ở
        `critic.py` (77.24 -> 57.43).
        """
        if ctx.state.get("docs_read"):
            return QUOTING_NUDGE
        sent = int(ctx.state.get("requery_nudges", 0) or 0)
        if sent >= REQUERY_NUDGE_TURNS:
            return None
        ctx.state["requery_nudges"] = sent + 1
        return REQUERY_NUDGE


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
