import argparse
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, TextIO


@dataclass
class ParsedValue:
    answer_block: Optional[str]
    tail: str


_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)


def _iter_json_array(path: str, chunk_size: int = 4 * 1024 * 1024):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        started = False
        ended = False

        while True:
            if not ended and pos >= len(buf) - 1024:
                chunk = f.read(chunk_size)
                if chunk:
                    buf = buf[pos:] + chunk
                    pos = 0
                else:
                    ended = True

            while pos < len(buf) and buf[pos].isspace():
                pos += 1

            if not started:
                if pos >= len(buf):
                    if ended:
                        break
                    continue
                if buf[pos] != "[":
                    raise ValueError(f"{path} is not a JSON array.")
                started = True
                pos += 1
                continue

            while pos < len(buf) and buf[pos].isspace():
                pos += 1

            if pos < len(buf) and buf[pos] == "]":
                return

            if pos < len(buf) and buf[pos] == ",":
                pos += 1
                continue

            if pos >= len(buf):
                if ended:
                    raise ValueError(f"{path} ended unexpectedly while parsing JSON array.")
                continue

            try:
                obj, length = decoder.raw_decode(buf[pos:])
            except ValueError:
                if ended:
                    raise
                continue

            pos += length
            yield obj


class _JsonArrayWriter:
    def __init__(self, fp: TextIO) -> None:
        self._fp = fp
        self._first = True
        self._fp.write("[\n")

    def write_item(self, item: dict[str, Any]) -> None:
        if not self._first:
            self._fp.write(",\n")
        self._first = False
        self._fp.write(json.dumps(item, ensure_ascii=False))

    def close(self) -> None:
        self._fp.write("\n]\n")


def _get_gid(item: dict[str, Any]) -> Optional[str]:
    gid = item.get("gid", None)
    if gid is None:
        gid = item.get("GID", None)
    if gid is None:
        return None
    return str(gid)


def _find_last_gpt_message(conversations: Any) -> Optional[dict[str, Any]]:
    if not isinstance(conversations, list):
        return None
    last = None
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("from") == "gpt":
            last = msg
    return last


def _parse_gpt_value(value: Any) -> ParsedValue:
    text = "" if value is None else str(value)
    m = _ANSWER_RE.search(text)
    if not m:
        return ParsedValue(answer_block=None, tail=text.strip())
    answer = m.group(0).strip()
    tail = (text[m.end() :] or "").strip()
    return ParsedValue(answer_block=answer, tail=tail)


def _default_answer_block() -> str:
    return "<answer> 0 </answer>"


def _merge_answers(ansa: Optional[ParsedValue], sgc: Optional[ParsedValue]) -> str:
    ansa_answer = (ansa.answer_block if ansa and ansa.answer_block else _default_answer_block()).strip()
    sgc_answer = (sgc.answer_block if sgc and sgc.answer_block else _default_answer_block()).strip()

    tail = ""
    if ansa and ansa.tail:
        tail = ansa.tail
    elif sgc and sgc.tail:
        tail = sgc.tail

    if tail:
        return f"ansa：{ansa_answer}\n\nsgc:{sgc_answer}\n\n{tail}"
    return f"ansa：{ansa_answer}\n\nsgc:{sgc_answer}"


def _rename_fields(item: dict[str, Any], suffix: str) -> None:
    if "gt" in item:
        item[f"gt_{suffix}"] = item.pop("gt")
    if "policy" in item:
        item[f"policy_{suffix}"] = item.pop("policy")

def _ensure_last_gpt_value(conversations: Any) -> dict[str, Any]:
    if not isinstance(conversations, list):
        raise ValueError("conversations must be a list.")
    msg = _find_last_gpt_message(conversations)
    if msg is None:
        msg = {"from": "gpt", "value": ""}
        conversations.append(msg)
    if "value" not in msg:
        msg["value"] = ""
    return msg


def _extract_parsed_from_item(item: dict[str, Any]) -> ParsedValue:
    conv = item.get("conversations", None)
    gpt = _find_last_gpt_message(conv)
    if gpt is None:
        return ParsedValue(answer_block=None, tail="")
    return _parse_gpt_value(gpt.get("value"))


def _connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sgc ("
        "gid TEXT PRIMARY KEY,"
        "answer TEXT,"
        "tail TEXT,"
        "gt TEXT,"
        "policy TEXT,"
        "item_json TEXT,"
        "used INTEGER DEFAULT 0"
        ")"
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _insert_sgc(conn: sqlite3.Connection, gid: str, item: dict[str, Any], parsed: ParsedValue) -> None:
    gt = item.get("gt", None)
    policy = item.get("policy", None)
    item_json = json.dumps(item, ensure_ascii=False)
    conn.execute(
        "INSERT OR IGNORE INTO sgc(gid, answer, tail, gt, policy, item_json, used) VALUES(?,?,?,?,?,?,0)",
        (
            gid,
            parsed.answer_block,
            parsed.tail,
            None if gt is None else str(gt),
            None if policy is None else str(policy),
            item_json,
        ),
    )


def _get_sgc_row(conn: sqlite3.Connection, gid: str) -> Optional[dict[str, Any]]:
    cur = conn.execute("SELECT answer, tail, gt, policy FROM sgc WHERE gid=?", (gid,))
    row = cur.fetchone()
    if row is None:
        return None
    answer, tail, gt, policy = row
    return {"answer": answer, "tail": tail, "gt": gt, "policy": policy}


def _mark_sgc_used(conn: sqlite3.Connection, gid: str) -> None:
    conn.execute("UPDATE sgc SET used=1 WHERE gid=?", (gid,))


def _iter_unused_sgc(conn: sqlite3.Connection):
    cur = conn.execute("SELECT gid, answer, tail, gt, policy, item_json FROM sgc WHERE used=0")
    for row in cur:
        gid, answer, tail, gt, policy, item_json = row
        yield {
            "gid": gid,
            "answer": answer,
            "tail": tail,
            "gt": gt,
            "policy": policy,
            "item_json": item_json,
        }


def _merge_items(ansa_item: Optional[dict[str, Any]], sgc_item: Optional[dict[str, Any]]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    if ansa_item is not None:
        base.update(ansa_item)
    elif sgc_item is not None:
        base.update(sgc_item)

    if ansa_item is not None:
        _rename_fields(base, "ansa")
    if sgc_item is not None:
        sgc_copy = dict(sgc_item)
        _rename_fields(sgc_copy, "sgc")
        for k, v in sgc_copy.items():
            if k not in base:
                base[k] = v
            elif k in ("conversations", "system"):
                continue
            elif k in ("gt_sgc", "policy_sgc"):
                base[k] = v

    conv = base.get("conversations", None)
    gpt_msg = _find_last_gpt_message(conv)
    if gpt_msg is None:
        if isinstance(conv, list):
            conv.append({"from": "gpt", "value": ""})
            gpt_msg = conv[-1]
        else:
            base["conversations"] = [{"from": "gpt", "value": ""}]
            gpt_msg = base["conversations"][0]

    ansa_parsed = None
    sgc_parsed = None
    if ansa_item is not None:
        a_conv = ansa_item.get("conversations", None)
        a_gpt = _find_last_gpt_message(a_conv)
        if a_gpt is not None:
            ansa_parsed = _parse_gpt_value(a_gpt.get("value"))
    if sgc_item is not None:
        s_conv = sgc_item.get("conversations", None)
        s_gpt = _find_last_gpt_message(s_conv)
        if s_gpt is not None:
            sgc_parsed = _parse_gpt_value(s_gpt.get("value"))

    gpt_msg["value"] = _merge_answers(ansa_parsed, sgc_parsed)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ansa", required=True, help="json1 path (ansa)")
    parser.add_argument("--sgc", required=True, help="json2 path (sgc)")
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument("--allow_missing_gid", action="store_true", default=False)
    parser.add_argument("--chunk_size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--tmp_db", default=None)
    args = parser.parse_args()
    require_gid = not args.allow_missing_gid

    db_path = args.tmp_db
    if db_path is None:
        fd, db_path = tempfile.mkstemp(prefix="merge_ansa_sgc_", suffix=".sqlite3")
        os.close(fd)

    conn = _connect_db(db_path)
    without_gid = 0
    any_gid = False

    try:
        for raw in _iter_json_array(args.sgc, chunk_size=args.chunk_size):
            if not isinstance(raw, dict):
                continue
            gid = _get_gid(raw)
            if gid is None:
                without_gid += 1
                continue
            any_gid = True
            parsed = _extract_parsed_from_item(raw)
            _insert_sgc(conn, gid, raw, parsed)
        conn.commit()

        with open(args.out, "w", encoding="utf-8") as out_f:
            writer = _JsonArrayWriter(out_f)

            for raw in _iter_json_array(args.ansa, chunk_size=args.chunk_size):
                if not isinstance(raw, dict):
                    continue
                gid = _get_gid(raw)
                if gid is None:
                    without_gid += 1
                    continue
                any_gid = True

                ansa_parsed = _extract_parsed_from_item(raw)
                sgc_row = _get_sgc_row(conn, gid)
                sgc_parsed = None
                if sgc_row is not None:
                    sgc_parsed = ParsedValue(answer_block=sgc_row["answer"], tail=sgc_row["tail"] or "")
                    _mark_sgc_used(conn, gid)

                base = dict(raw)
                _rename_fields(base, "ansa")
                if sgc_row is not None:
                    if sgc_row.get("gt") is not None:
                        base["gt_sgc"] = sgc_row["gt"]
                    if sgc_row.get("policy") is not None:
                        base["policy_sgc"] = sgc_row["policy"]

                conv = base.get("conversations", None)
                if not isinstance(conv, list):
                    conv = []
                    base["conversations"] = conv
                gpt_msg = _ensure_last_gpt_value(conv)
                gpt_msg["value"] = _merge_answers(ansa_parsed, sgc_parsed)
                base["gid"] = gid
                writer.write_item(base)

            conn.commit()

            for row in _iter_unused_sgc(conn):
                gid = row["gid"]
                sgc_item = json.loads(row["item_json"])
                if not isinstance(sgc_item, dict):
                    continue
                sgc_parsed = ParsedValue(answer_block=row["answer"], tail=row["tail"] or "")
                ansa_parsed = None

                base = dict(sgc_item)
                _rename_fields(base, "sgc")

                conv = base.get("conversations", None)
                if not isinstance(conv, list):
                    conv = []
                    base["conversations"] = conv
                gpt_msg = _ensure_last_gpt_value(conv)
                gpt_msg["value"] = _merge_answers(ansa_parsed, sgc_parsed)
                base["gid"] = gid
                writer.write_item(base)

            writer.close()

    finally:
        conn.close()
        if args.tmp_db is None:
            try:
                os.remove(db_path)
            except OSError:
                pass

    if require_gid:
        if not any_gid:
            raise ValueError("No gid found in either input file.")
        if without_gid:
            raise ValueError(f"Found {without_gid} items without gid.")


if __name__ == "__main__":
    main()
