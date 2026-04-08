"""
Module 4: Curriculum Knowledge Point Extraction & Semantic Alignment
=====================================================================
Thesis Ch. 4-4

INPUT:
  transcript  — cleaned text (Module 3 output)
  syllabus    — structured text:
      章節標題 / Chapter Title
      學習目標 / Learning Objectives  (list)
      教學重點 / Key Points           (list, supports "main：sub1、sub2")

STEP 4-1 — parse_syllabus():
  Detect section headers (Chinese or English), extract
  chapter, objectives, key_points (with sub-point expansion).

STEP 4-2 — align():
  Encode sentences + key_points via
    multilingual Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2)
  Traditional matching (Thesis 4-4-2 Mode A):
    cosine_similarity(s_emb, kp_emb) ≥ threshold → direct match
  Intelligent matching (Mode B, enabled by default):
    If sim ≥ threshold×0.7:
      example indicator   (例如/比如/for example) → "example illustration"
      extension indicator (另外/此外/furthermore) → "extended discussion"
      application indic.  (應用/實際/practical)  → "practical application"
      context indicator   (這個/因此/therefore)  → "contextual link"
      character overlap with KP                  → "synonym match"

STEP 4-3 — extract_keywords():
  KeyBERT: top-5 keyphrases (n-gram 1–2)
  CKIP: word segmentation + POS filter (N/V/A/VH/VL) → merge, Top-3

STEP 4-4 — analyze_coverage():
  coverage_rate = covered_KPs / total_KPs, hit counts

OUTPUT:
  { syllabus: {chapter, objectives, key_points},
    alignment: [{sentence, matched_key_point, similarity, match_type, keywords}],
    coverage:  {coverage_rate, matched_sentences, covered_key_points, ...} }
"""

import re, logging
import numpy as np
from typing import List, Dict, Optional, Callable, Any
logger = logging.getLogger(__name__)

STOP = {
    "的","了","在","是","我","有","和","就","不","人","都","一","上","也","很","到","說",
    "要","去","你","會","著","沒有","看","好","自己","這","那","它","他","她","我們","你們",
    "他們","這個","那個","這些","那些","什麼","怎麼","為什麼","哪裡","可以","應該","能夠",
    "必須","需要","想要","希望","覺得","認為","知道","所以","但是","然而","雖然","如果",
    "the","a","an","is","are","was","be","in","of","to","and","or","for","with","that","it",
}
EX_IND  = {"例如","比如","舉例","假設","像是","譬如","for example","such as","e.g."}
EXT_IND = {"另外","此外","進一步","延伸","補充","furthermore","moreover","additionally"}
APP_IND = {"應用","實際","實務","實作","運用","使用","practical","application","implement"}
CTX_IND = {"這個","那個","因此","所以","然後","接著","此","該","therefore","thus","hence"}


class SyllabusAligner:
    def __init__(self):
        self._sent_model = self._keybert = self._ckip_ws = self._ckip_pos = None

    def _load_sent(self):
        if self._sent_model: return
        from sentence_transformers import SentenceTransformer
        self._sent_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def _load_keybert(self):
        if self._keybert: return
        from keybert import KeyBERT
        self._keybert = KeyBERT()

    def _load_ckip(self):
        if self._ckip_ws: return
        try:
            from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
            self._ckip_ws  = CkipWordSegmenter(model="bert-base")
            self._ckip_pos = CkipPosTagger(model="bert-base")
        except Exception as e:
            logger.warning(f"CKIP unavailable: {e}")

    # ── 4-1 parse ──────────────────────────────────────────
    def parse_syllabus(self, text: str) -> Dict[str, Any]:
        result: Dict[str,Any] = {"chapter":"","objectives":[],"key_points":[]}
        cur = None
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            if any(k in line for k in ("章節標題","Chapter Title","章節")):
                cur = "chapter"; continue
            elif any(k in line for k in ("學習目標","Learning Objectives","目標")):
                cur = "objectives"; continue
            elif any(k in line for k in ("教學重點","Key Points","重點","知識點")):
                cur = "key_points"; continue
            if cur == "chapter" and not result["chapter"]:
                result["chapter"] = line
            elif cur == "objectives":
                clean = re.sub(r"^[\d①②③]+[.\s)）]+","",line).strip()
                if clean: result["objectives"].append(clean)
            elif cur == "key_points":
                found_sep = False
                for sep in ("：",":"):
                    if sep in line:
                        parts = line.split(sep,1)
                        m = parts[0].strip()
                        if m: result["key_points"].append(m)
                        for sub in re.split(r"[、,，]", parts[1]):
                            if sub.strip(): result["key_points"].append(sub.strip())
                        found_sep = True; break
                if not found_sep:
                    cl = re.sub(r"^[\d•\-·]+\s*","",line).strip()
                    if cl: result["key_points"].append(cl)
        return result

    # ── 4-2 align ──────────────────────────────────────────
    def align(self, sentences: List[str], key_points: List[str],
              threshold: float = 0.6, smart: bool = True) -> List[Dict]:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self._load_sent()
        if not sentences or not key_points: return []
        s_emb  = self._sent_model.encode(sentences, show_progress_bar=False)
        kp_emb = self._sent_model.encode(key_points, show_progress_bar=False)
        results = []
        for i, sent in enumerate(sentences):
            sims    = cos_sim(s_emb[i:i+1], kp_emb)[0]
            bi      = int(np.argmax(sims))
            bs      = float(sims[bi])
            matched, score, mtype = None, 0.0, "no match"
            if bs >= threshold:
                matched, score, mtype = key_points[bi], bs, "direct match"
            elif smart:
                lo = threshold * 0.7
                if bs >= lo:
                    if   any(w in sent for w in EX_IND):
                        matched, score, mtype = key_points[bi], bs, "example illustration"
                    elif any(w in sent for w in EXT_IND):
                        matched, score, mtype = key_points[bi], bs, "extended discussion"
                    elif any(w in sent for w in APP_IND):
                        matched, score, mtype = key_points[bi], bs, "practical application"
                    elif any(w in sent for w in CTX_IND) and bs >= lo * 0.85:
                        matched, score, mtype = key_points[bi], bs, "contextual link"
                    else:
                        for j, kp in enumerate(key_points):
                            overlap = (set(kp)-STOP) & (set(sent)-STOP)
                            if len(overlap) >= 2:
                                matched = kp; score = float(sims[j]); mtype = "synonym match"; break
            results.append({"sentence":sent,"matched_key_point":matched,
                             "similarity":round(score,4),"match_type":mtype,"keywords":[]})
        hits = sum(1 for r in results if r["matched_key_point"])
        logger.info(f"Alignment: {hits}/{len(sentences)} matched")
        return results

    # ── 4-3 keywords ───────────────────────────────────────
    def extract_keywords(self, results: List[Dict]) -> List[Dict]:
        self._load_keybert(); self._load_ckip()
        POS = {"N","V","A","VH","VL","VA","VC"}
        for r in results:
            sent = r["sentence"]; kws: List[str] = []
            try:
                kws = [w for w,_ in self._keybert.extract_keywords(
                    sent, keyphrase_ngram_range=(1,2),
                    stop_words=list(STOP), top_n=5)
                    if w not in STOP and len(w) > 1]
            except Exception: pass
            if self._ckip_ws:
                try:
                    ws = self._ckip_ws([sent])
                    ps = self._ckip_pos(ws) if self._ckip_pos else [[]]
                    for w, p in zip(ws[0], ps[0]):
                        if any(p.startswith(m) for m in POS) and len(w)>1 and w not in STOP and w not in kws:
                            kws.append(w)
                except Exception: pass
            r["keywords"] = kws[:3]
        return results

    # ── 4-4 coverage ───────────────────────────────────────
    def analyze_coverage(self, results: List[Dict], key_points: List[str]) -> Dict:
        kpc = {kp:0 for kp in key_points}
        for r in results:
            if r["matched_key_point"] in kpc: kpc[r["matched_key_point"]] += 1
        cov = sum(1 for v in kpc.values() if v > 0)
        mat = sum(1 for r in results if r["matched_key_point"])
        return {
            "total_sentences":    len(results),
            "matched_sentences":  mat,
            "unmatched_sentences":len(results)-mat,
            "total_key_points":   len(key_points),
            "covered_key_points": cov,
            "uncovered_key_points":len(key_points)-cov,
            "coverage_rate":      round(cov/max(len(key_points),1),4),
            "key_point_counts":   kpc,
        }

    # ── Full pipeline ───────────────────────────────────────
    def full_pipeline(self, transcript: str, syllabus_text: str,
                       threshold: float = 0.6, smart: bool = True,
                       progress_cb: Optional[Callable] = None) -> Dict:
        def _cb(p,m):
            if progress_cb: progress_cb(p,m)
        _cb(0.05, "Parsing syllabus …")
        syllabus = self.parse_syllabus(syllabus_text)
        _cb(0.20, "Splitting transcript …")
        sents = [s.strip() for s in re.split(r"[。！？\n]",transcript) if len(s.strip())>3]
        _cb(0.40, "Sentence-BERT encoding + alignment …")
        results = self.align(sents, syllabus["key_points"], threshold, smart)
        _cb(0.75, "Keyword extraction (KeyBERT + CKIP) …")
        results = self.extract_keywords(results)
        _cb(0.92, "Coverage analysis …")
        coverage = self.analyze_coverage(results, syllabus["key_points"])
        _cb(1.0, f"Done — coverage {coverage['coverage_rate']*100:.1f}%")
        return {"syllabus":syllabus,"alignment":results,"coverage":coverage}
