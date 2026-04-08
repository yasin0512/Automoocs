"""Module 3 — Text Cleaning (LLM post-processing) — Thesis Ch. 4-3"""
# ─── text_cleaner.py ──────────────────────────────────────────
import os, time, logging, re
from typing import Optional, Dict, Callable
logger = logging.getLogger(__name__)

NOISE_WORDS = [
    "喂喂喂","呃","有齁","然後咧","對不對","我們現在繼續",
    "那個","其實","基本上","就是說","這樣子","你知道嗎",
    "來喔","同學看這邊","嗯嗯","啊啊","齁齁","對對對",
    "好好好","欸欸","等等","等一下","先等一下","暫停一下",
    "有沒有","聽得到嗎","看得到嗎","清楚嗎","懂嗎",
    "麥克風","投影機","螢幕","黑板","白板","看這邊",
]
_SYS = "You are a professional text editor specializing in speech transcript cleaning. Always respond in Traditional Chinese."
_TMPL = """Clean and optimize the following speech recognition transcript:
1. Remove excessive filler words (嗯、那個、就是、然後、有齁、對不對 etc.) while preserving rhythm
2. Fix typos, grammatical errors, punctuation mistakes
3. Add appropriate sentence breaks and paragraph formatting
4. Preserve the speaker's tone, meaning and teaching style
5. Output ONLY Traditional Chinese (繁體中文)
6. Output ONLY the cleaned text — no explanations

Original transcript:
\"\"\"
{text}
\"\"\"
Cleaned text:"""


class TextCleaner:
    def __init__(self, openai_key: str = "", ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3", openai_model: str = "gpt-4o"):
        self.openai_key   = openai_key or os.getenv("OPENAI_API_KEY","")
        self.ollama_url   = ollama_url
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self._ollama_ok: Optional[bool] = None

    def check_ollama(self) -> bool:
        import requests
        try:
            self._ollama_ok = requests.get(f"{self.ollama_url}/api/tags", timeout=4).status_code == 200
        except Exception:
            self._ollama_ok = False
        return self._ollama_ok

    def available_backends(self) -> Dict[str, bool]:
        return {"ollama": self.check_ollama(), "openai": bool(self.openai_key)}

    def _ollama_call(self, text: str) -> str:
        import requests
        r = requests.post(f"{self.ollama_url}/api/chat", json={
            "model": self.ollama_model,
            "messages": [{"role":"system","content":_SYS},
                         {"role":"user","content":_TMPL.format(text=text)}],
            "stream": False, "options":{"temperature":0.3},
        }, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _openai_call(self, text: str) -> str:
        import openai
        c = openai.OpenAI(api_key=self.openai_key)
        r = c.chat.completions.create(model=self.openai_model,
            messages=[{"role":"system","content":_SYS},
                      {"role":"user","content":_TMPL.format(text=text)}],
            temperature=0.3, max_tokens=4000)
        return r.choices[0].message.content.strip()

    @staticmethod
    def _rule_clean(text: str) -> str:
        for w in NOISE_WORDS: text = text.replace(w,"")
        text = re.sub(r"[，,]{2,}","，",text)
        text = re.sub(r"[。.]{2,}","。",text)
        return re.sub(r"\s+"," ",text).strip()

    def clean(self, text: str, prefer: str = "ollama",
              progress_cb: Optional[Callable] = None) -> Dict:
        def _cb(p,m):
            if progress_cb: progress_cb(p,m)
        t0 = time.time()
        result = {"original":text,"cleaned":text,"method":"none","elapsed":0.0,"success":False}
        order = ([("ollama",self._ollama_call),("openai",self._openai_call)]
                 if prefer == "ollama" else
                 [("openai",self._openai_call),("ollama",self._ollama_call)])
        for backend, fn in order:
            if backend == "ollama":
                if self._ollama_ok is None: self.check_ollama()
                if not self._ollama_ok: continue
            elif backend == "openai":
                if not self.openai_key: continue
            _cb(0.4, f"Cleaning via {backend} ({self.ollama_model if backend=='ollama' else self.openai_model})…")
            try:
                cleaned = fn(text)
                result.update({"cleaned":cleaned,"method":f"{backend}/{self.ollama_model if backend=='ollama' else self.openai_model}","success":True})
                _cb(1.0,"Cleaning complete"); break
            except Exception as e:
                logger.warning(f"{backend} failed: {e}")
        if not result["success"]:
            _cb(0.8,"Applying rule-based fallback…")
            result.update({"cleaned":self._rule_clean(text),"method":"rule-based","success":True})
            _cb(1.0,"Rule-based cleaning done")
        result["elapsed"] = round(time.time()-t0,2)
        return result
