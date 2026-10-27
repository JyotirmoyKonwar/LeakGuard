"""Build the evaluation / training data from existing Hugging Face datasets only (no synthetic text).

Every example is normalised to one schema:
    {id, source, text, spans:[{start,end,entity}], direction, destination, role, tool_name, tool_target,
     gold_action, gold_protect, scenario, lang, split}

Gold actions come from a fixed, documented mapping of each dataset's *own* labels (see GOLD SPEC below and
docs/dataset.md). They are never produced by running LeakGuard, so the benchmark is not circular.
Rows may be re-used in another direction (prompt / response / tool_payload) by a deterministic hash-based
assignment; the text itself is never modified.

Run:  python -m leakguard.datagen.hf --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("datagen")

# ---------------------------------------------------------------------------------------------- label maps
AI4P_MAP = {  # ai4privacy/open-pii-masking-500k label -> LeakGuard entity (unlisted labels are ignored)
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "EMAIL": "EMAIL", "TELEPHONENUM": "PHONE",
    "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS", "CREDITCARDNUMBER": "CREDIT_CARD",
    "PASSPORTNUM": "PASSPORT", "IDCARDNUM": "GOV_ID", "TAXNUM": "GOV_ID", "SOCIALNUM": "GOV_ID",
    "DRIVERLICENSENUM": "GOV_ID",
}
AI4P_IGNORED = ["CITY", "TIME", "DATE", "TITLE", "AGE", "ZIPCODE", "SEX", "GENDER"]  # quasi-identifiers

PROWL_MAP = {  # Podric/prowl-secrets-corpus label_type -> entity ("neg_*" are negatives)
    "aws_access_key_id": "AWS_ACCESS_KEY", "aws_secret_access_key": "AWS_SECRET_KEY",
    "anthropic_api_key": "ANTHROPIC_KEY", "openai_api_key": "OPENAI_KEY", "azure_storage_key": "AZURE_STORAGE_KEY",
    "basic_auth_header": "BASIC_AUTH", "datadog_api_key": "DATADOG_KEY", "db_connection_string": "DB_CONNECTION_STRING",
    "digitalocean_token": "DIGITALOCEAN_TOKEN", "discord_bot_token": "DISCORD_TOKEN", "gcp_api_key": "GCP_API_KEY",
    "gcp_service_account": "PRIVATE_KEY", "generic_api_key": "GENERIC_SECRET", "generic_high_entropy": "GENERIC_SECRET",
    "generic_password": "PASSWORD", "github_pat_classic": "GITHUB_TOKEN", "github_pat_fine_grained": "GITHUB_TOKEN",
    "github_token_oauth": "GITHUB_TOKEN", "gitlab_pat": "GITLAB_TOKEN", "huggingface_token": "HF_TOKEN", "jwt": "JWT",
    "npm_token": "NPM_TOKEN", "private_key_pem": "PRIVATE_KEY", "pypi_token": "PYPI_TOKEN",
    "sendgrid_api_key": "SENDGRID_KEY", "shopify_token": "SHOPIFY_TOKEN", "slack_token": "SLACK_TOKEN",
    "slack_webhook": "SLACK_WEBHOOK", "stripe_secret_key": "STRIPE_KEY", "telegram_bot_token": "TELEGRAM_TOKEN",
    "twilio_api_key": "TWILIO_KEY",
}
PROWL_SKIP_NEG = {"neg_pii"}  # 'negative for secrets' but contains PII -> not a clean negative for a DLP guard

BLOCK_SECRETS = {"PRIVATE_KEY", "DB_CONNECTION_STRING"}
LOW_SEV_PII = {"PERSON", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "IP_ADDRESS"}
BULK_PII = 8

# ---------------------------------------------------------------------------------------------- GOLD SPEC
# The expected behaviour of an enterprise DLP guard, written as a function of the *dataset labels*.
#   prompt  -> external : secrets -> redact (private keys / DB URLs -> block); PII -> redact;
#                         >= 8 PII values -> escalate; nothing -> allow
#   prompt  -> internal : only low-risk PII -> allow; high-risk PII -> redact
#   response            : any secret -> block; any PII (not supplied by the user) -> redact; nothing -> allow
#   tool_payload ext.   : any secret -> block; high-risk PII -> block; only low-risk PII -> escalate; nothing -> allow
#   injection / jailbreak label -> block ; 'sensitive' label (no spans) -> protect (anything but allow)


def gold_for(spans: list[dict], direction: str, destination: str) -> str:
    ents = {s["entity"] for s in spans}
    from ..detection.types import SECRET_ENTITIES

    secrets = ents & SECRET_ENTITIES
    pii = ents - SECRET_ENTITIES
    n_pii = sum(1 for s in spans if s["entity"] not in SECRET_ENTITIES)
    if direction == "prompt":
        if secrets & BLOCK_SECRETS:
            return "block"
        if destination == "internal" and not secrets:
            return "allow" if pii <= LOW_SEV_PII else "redact"
        if n_pii >= BULK_PII:
            return "escalate"
        return "redact" if ents else "allow"
    if direction == "response":
        if secrets:
            return "block"
        return "redact" if pii else "allow"
    if direction == "tool_payload":
        if secrets or (pii - LOW_SEV_PII):
            return "block"
        return "escalate" if pii else "allow"
    raise ValueError(direction)


# ---------------------------------------------------------------------------------------------- helpers
def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def split_of(key: str, dev: float = 0.2, test: float = 0.2) -> str:
    r = (_h("split:" + key) % 10_000) / 10_000
    return "test" if r < test else "dev" if r < test + dev else "train"


def merge_adjacent(spans: list[dict], text: str) -> list[dict]:
    """Join e.g. GIVENNAME + SURNAME separated only by whitespace into one PERSON span."""
    spans = sorted(spans, key=lambda s: s["start"])
    out: list[dict] = []
    for s in spans:
        if out and out[-1]["entity"] == s["entity"] and text[out[-1]["end"]:s["start"]].strip(" ,") == "" \
                and s["start"] - out[-1]["end"] <= 2:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def _example(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"spans": [], "direction": "prompt", "destination": "external", "role": "employee", "tool_name": None,
            "tool_target": None, "lang": "en"}
    base.update(kw)
    base["gold_protect"] = base["gold_action"] != "allow"
    return base


# ---------------------------------------------------------------------------------------------- sources
def load_ai4privacy(n: int, seed: int) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=50_000)
    per_lang: Counter[str] = Counter()
    cap = max(1, n // 8) + n // 16  # roughly balance the 8 languages
    for r in ds:
        lang = r["language"]
        if per_lang[lang] >= cap:
            continue
        text = r["source_text"]
        spans = [{"start": m["start"], "end": m["end"], "entity": AI4P_MAP[m["label"]]}
                 for m in r["privacy_mask"] if m["label"] in AI4P_MAP]
        spans = merge_adjacent(spans, text)
        rid = f"ai4p-{r['uid']}"
        # deterministic direction / destination re-use (text untouched)
        bucket = _h("dir:" + rid) % 100
        if bucket < 12:
            direction, dest, scen = "response", "external", "pii_response"
        elif bucket < 22:
            direction, dest, scen = "prompt", "internal", "pii_internal_llm"
        else:
            direction, dest, scen = "prompt", "external", "pii_prompt"
        yield _example(id=rid, source="ai4privacy/open-pii-masking-500k", text=text, spans=spans,
                       direction=direction, destination=dest, gold_action=gold_for(spans, direction, dest),
                       scenario=scen, lang=lang, split=split_of(rid))
        per_lang[lang] += 1
        if sum(per_lang.values()) >= n:
            break


def load_prowl(n: int, seed: int) -> Iterator[dict]:
    """Rows are (text, one labelled value). The same text can appear in several rows with different values,
    so rows are grouped by text and their positive spans unioned."""
    from datasets import load_dataset

    ds = load_dataset("Podric/prowl-secrets-corpus", split="train")
    # 1) choose the texts to sample, 2) collect EVERY row for those texts. A text's secrets are spread over
    #    several rows; sampling rows instead of texts would leave real secrets unlabelled ("false negatives").
    texts = ds.unique("text")
    random.Random(seed).shuffle(texts)
    chosen = set(texts[: int(n * 1.3)])
    rows = ds.filter(lambda r: r["text"] in chosen)
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        lt = r["label_type"]
        if lt in PROWL_SKIP_NEG:
            groups.setdefault(r["text"], {"skip": True})["skip"] = True
            continue
        g = groups.setdefault(r["text"], {"spans": [], "types": set(), "source": r["source"], "skip": False})
        if g.get("skip"):
            continue
        g["types"].add(lt)
        if str(r["label_binary"]) == "1" and lt in PROWL_MAP:
            span = r["span"]
            if isinstance(span, str) and span:
                span = json.loads(span)
            if span:
                s, e = int(span[0]), int(span[1])
            else:  # some rows carry no offsets: locate the value in the text
                s = r["text"].find(r["value"])
                e = s + len(r["value"])
            if s >= 0 and r["text"][s:e] == r["value"]:
                g["spans"].append({"start": s, "end": e, "entity": PROWL_MAP[lt]})
    count = 0
    for text, g in groups.items():
        if g.get("skip"):
            continue
        uniq = sorted({(s["start"], s["end"], s["entity"]) for s in g["spans"]})
        spans = merge_adjacent([{"start": a, "end": b, "entity": c} for a, b, c in uniq], text)
        rid = f"prowl-{_h(text):012x}"
        bucket = _h("dir:" + rid) % 100
        if bucket < 15:
            direction, dest, tool, target, scen = "tool_payload", "external", "http_post", "https://paste.example.net/upload", "secret_tool_external"
        elif bucket < 25:
            direction, dest, tool, target, scen = "response", "external", None, None, "secret_response"
        else:
            direction, dest, tool, target, scen = "prompt", "external", None, None, "secret_prompt"
        if not spans:
            scen = scen.replace("secret", "negative")
        yield _example(id=rid, source="Podric/prowl-secrets-corpus", text=text, spans=spans, direction=direction,
                       destination=dest, tool_name=tool, tool_target=target,
                       gold_action=gold_for(spans, direction, dest), scenario=scen,
                       split=split_of(rid), meta={"prowl_source": g["source"], "label_types": sorted(g["types"])})
        count += 1
        if count >= n:
            break


def load_deepset() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("deepset/prompt-injections")
    for hf_split, rows in ds.items():
        for i, r in enumerate(rows):
            rid = f"deepset-{hf_split}-{i}"
            inj = int(r["label"]) == 1
            # keep deepset's own test split as test; carve dev out of its train split
            split = "test" if hf_split == "test" else ("dev" if _h(rid) % 5 == 0 else "train")
            yield _example(id=rid, source="deepset/prompt-injections", text=r["text"],
                           gold_action="block" if inj else "allow", scenario="injection" if inj else "benign_prompt",
                           lang="de" if re.search(r"[äöüß]|\b(und|nicht|ich|der|die|das)\b", r["text"]) else "en",
                           split=split)


def load_v1adam() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("v1adam/Prompt_injection_and_Sensitive_Data_exposure_detection", split="train")
    for i, r in enumerate(ds):
        rid = f"v1adam-{i}"
        lab = r["label"]
        gold = {"benign": "allow", "jailbreak": "block", "sensitive": "protect"}[lab]
        ex = _example(id=rid, source="v1adam/Prompt_injection_and_Sensitive_Data_exposure_detection", text=r["prompt"],
                      gold_action=gold, scenario=f"policy_{lab}", split=split_of(rid), meta={"template": r["template"]})
        ex["gold_protect"] = gold != "allow"
        yield ex


def load_code_secrets() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("nisaefendioglu/synthetic-sensitive-data-in-source-code-n300", split="train")
    secret_cats = {"api_key", "password_secret", "connection_string", "multi_secret", "private_key_keystore"}
    for r in ds:
        rid = f"code-{r['id']}"
        cat = r["category"]
        gold = "block" if cat in secret_cats else ("protect" if cat == "pii" else "allow")  # internal_url -> out of scope
        ex = _example(id=rid, source="nisaefendioglu/synthetic-sensitive-data-in-source-code-n300",
                      text=r["code_text"].replace("\\n", "\n"), direction="tool_payload", destination="external",
                      tool_name="http_post", tool_target="https://gist.example.net/api", gold_action=gold,
                      scenario=f"code_{cat}", split="test" if _h(rid) % 2 else "dev",
                      meta={"finding_types": r["finding_types"], "language": r["language"]})
        ex["gold_protect"] = gold != "allow"
        yield ex


# ---------------------------------------------------------------------------------------------- build
def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def build(out: Path, n_pii: int, n_secrets: int, seed: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    rows: list[dict] = []
    for name, it in [("ai4privacy", load_ai4privacy(n_pii, seed)), ("prowl", load_prowl(n_secrets, seed)),
                     ("deepset", load_deepset()), ("v1adam", load_v1adam()), ("code", load_code_secrets())]:
        before = len(rows)
        rows.extend(it)
        log.info("%s: %d examples", name, len(rows) - before)

    manifest: dict[str, Any] = {"seed": seed, "sources": {}, "splits": {}}
    for split in ("train", "dev", "test"):
        part = [r for r in rows if r["split"] == split]
        n = write_jsonl(out / f"{split}.jsonl", part)
        digest = hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()
        manifest["splits"][split] = {
            "n": n, "sha256": digest,
            "by_source": dict(Counter(r["source"] for r in part)),
            "by_scenario": dict(Counter(r["scenario"] for r in part)),
            "by_gold": dict(Counter(r["gold_action"] for r in part)),
            "by_lang": dict(Counter(r["lang"] for r in part)),
            "spans": sum(len(r["spans"]) for r in part),
        }
    manifest["entity_counts"] = dict(Counter(s["entity"] for r in rows for s in r["spans"]))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-pii", type=int, default=6000)
    ap.add_argument("--n-secrets", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    m = build(Path(a.out), a.n_pii, a.n_secrets, a.seed)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("n", "by_gold")} for k, v in m["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
