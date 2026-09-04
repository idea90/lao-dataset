"""
Lao Grammar & Linguistic Knowledge Web Service / API for LLMs & AI Agents.

Provides:
- An LLM-optimized REST & JSON API for fetching Lao grammar rules, classifiers, tone charts, and vocabulary.
- A /llms.txt and /llms-full.txt endpoint following the official LLM markdown scraping standard.
- Semantic search across Lao grammar rules and classifiers.
- Live grammar validation helper (e.g. checking classifier usage and word order).
- A clean, beautiful responsive web interface for human users.
- OpenAPI / Swagger interactive documentation at /docs.
"""

import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows console UTF-8 setup
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel

from core.knowledge_base import (
    LAO_GRAMMAR_RULES,
    LAO_CLASSIFIERS_TABLE,
    LAO_CONSONANT_CLASSES,
)
from core.unicode_utils import (
    contains_lao,
    lao_character_ratio,
    normalize_lao_text,
    analyze_lao_script,
)

LAO_TONE_RULES_SUMMARY = """
### Tone Rules Breakdown (Standard Vientiane Dialect - 6 Tones):
1. **Live Syllables (Unmarked)**:
   - High Consonant -> Rising tone (24 / ˩˦)
   - Middle Consonant -> Low tone (11 / ˩)
   - Low Consonant -> High-falling or High tone (35 / ˦˥)

2. **Tone Mark 1 (ໄມ້ເອກ ່)**:
   - High Consonant -> Mid-falling tone (31 / ˧˩)
   - Middle Consonant -> Mid-falling tone (31 / ˧˩)
   - Low Consonant -> Mid tone (33 / ˧)

3. **Tone Mark 2 (ໄມ້ໂທ ້)**:
   - High Consonant -> Low-falling tone (52 / ˥˨)
   - Middle Consonant -> High-falling tone (53 / ˥˧)
   - Low Consonant -> High-falling / Rising tone (51 / ˥˩)

4. **Dead Syllables (Checked Stops -p, -t, -k)**:
   - Short Vowel: High/Mid consonant -> High tone (55); Low consonant -> Mid tone (33)
   - Long Vowel: High/Mid consonant -> Low tone (11); Low consonant -> High-falling (52)
"""

app = FastAPI(
    title="🇱🇦 Lao Language & Grammar Knowledge API for LLMs",
    description="An AI-ready semantic knowledge service serving authoritative Lao grammar rules, classifiers (ລັກສະນະນາມ), tone determination matrices, and syntax guides for LLMs, RAG pipelines, and AI agents.",
    version="1.0.0",
)

# Enable CORS for external agents or web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "final"


# -----------------------------------------------------------------------------
# LLM Standards: /llms.txt & /llms-full.txt
# -----------------------------------------------------------------------------

@app.get("/llms.txt", response_class=PlainTextResponse, summary="LLM Standard Documentation Index")
def get_llms_txt():
    """
    Standard llms.txt index describing the Lao grammar knowledge base
    for AI crawlers, perplexity-style search engines, and LLMs.
    """
    return """# Lao Language & Grammar Knowledge Service (ພາສາລາວ)

> An authoritative, structured linguistic reference engine providing rules, classifiers, tone determination matrices, and syntactic templates for the Lao language.

## Quick Links for AI Agents
- Full LLM Context: /llms-full.txt (All grammar rules, tone charts, and classifiers in markdown)
- All Grammar Rules (JSON): /api/rules
- All Classifiers (JSON): /api/classifiers
- Tone Determination Matrix (JSON): /api/tones
- Consonant Classes (JSON): /api/consonants
- Semantic Search: /api/search?q={query}
- Grammar Validator: /api/validate (POST)

## Core Linguistic Characteristics of Lao
1. **Word Order**: Strict canonical Subject-Verb-Object (SVO). Head-initial noun phrases: modifiers (adjectives, possessives) always follow nouns (`Noun + Adjective`).
2. **Classifiers (ລັກສະນະນາມ)**: Mandatory for counting (`Noun + Numeral + Classifier`) and specification (`Noun + Classifier + Demonstrative`). Numbers cannot directly modify nouns.
3. **Tones**: 6 distinct lexical tones (in standard Vientiane dialect) determined by an interplay of:
   - Initial Consonant Class (High, Middle, Low)
   - Syllable Ending (Live / Sonorant vs Dead / Checked stop -p, -t, -k)
   - Vowel Length (Short vs Long)
   - Tone Marks (ໄມ້ເອກ ່, ໄມ້ໂທ ້, etc.)
4. **Negation**: The particle 'ບໍ່' (bor) is placed directly *before* the main verb.
5. **Questions**: Yes/No questions are formed by placing 'ບໍ່?' (bor?) or 'ຫວາ?' (waa?) at the end of the sentence.
6. **Spacing**: No spaces between words. Whitespace functions as clause/sentence punctuation.
"""


@app.get("/llms-full.txt", response_class=PlainTextResponse, summary="Full Lao Grammar Context for LLM Ingestion")
def get_llms_full_txt():
    """
    Returns the complete Lao grammar manual in dense, well-structured Markdown,
    ideal for direct prompt injection, system prompts, or RAG context ingestion.
    """
    lines = [
        "# Complete Lao Language & Grammar Specification (ພາສາລາວ)\n",
        "## 1. Overview & Typology",
        "- **Family**: Kra–Dai -> Southwestern Tai (closely related to Isan and Thai, written in Lao script).",
        "- **Typology**: Analytic/Isolating, Tonogenesis, Canonical SVO, Head-Initial Noun Phrases.\n",
        "## 2. Core Grammar Rules\n",
    ]

    for rule in LAO_GRAMMAR_RULES:
        lines.append(f"### {rule['category']}: {rule['rule_name']} ({rule['lao_name']})")
        lines.append(f"**Explanation:** {rule['explanation']}")
        lines.append(f"**Formula:** `{rule['formula']}`")
        lines.append("**Examples:**")
        for eg in rule["examples"]:
            lines.append(f"- Lao: **{eg['lao']}** ({eg['transcription']})")
            lines.append(f"  - Gloss: {eg['gloss']}")
            lines.append(f"  - Translation: \"{eg['translation']}\"")
        lines.append("")

    lines.append("## 3. Mandatory Numerical Classifiers (ລັກສະນະນາມ)\n")
    lines.append("Lao requires classifiers when counting or specifying nouns. Pattern: `Noun + Number + Classifier`\n")
    lines.append("| Classifier | Transcription | Primary Usage Category |")
    lines.append("| :--- | :--- | :--- |")
    for clf in LAO_CLASSIFIERS_TABLE:
        lines.append(f"| **{clf['classifier']}** | {clf['transcription']} | {clf['usage']} |")
    lines.append("")

    lines.append("## 4. Tone Determination Rules (6 Tones in Vientiane Dialect)\n")
    lines.append(LAO_TONE_RULES_SUMMARY)
    lines.append("")

    lines.append("## 5. Consonant Classification (High, Middle, Low)\n")
    for group, chars in LAO_CONSONANT_CLASSES.items():
        lines.append(f"- **{group.capitalize()} Class**: {', '.join(chars)}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# REST API Endpoints (JSON for Tools, Function Calling & RAG)
# -----------------------------------------------------------------------------

@app.get("/api/rules", summary="Get all formal Lao grammar rules")
def get_grammar_rules(category: Optional[str] = Query(None, description="Filter rules by category")):
    """Returns structured Lao grammar rules, formulas, and verified bilingual examples."""
    if category:
        filtered = [r for r in LAO_GRAMMAR_RULES if category.lower() in r["category"].lower()]
        return {"category": category, "count": len(filtered), "rules": filtered}
    return {"count": len(LAO_GRAMMAR_RULES), "rules": LAO_GRAMMAR_RULES}


@app.get("/api/classifiers", summary="Get Lao classifier catalog (ລັກສະນະນາມ)")
def get_classifiers(q: Optional[str] = Query(None, description="Search for a noun or semantic category")):
    """Returns the Lao classifiers catalog and finds the appropriate classifier for a noun."""
    if q:
        q_clean = q.strip().lower()
        matched = []
        for clf in LAO_CLASSIFIERS_TABLE:
            if (q_clean in clf["usage"].lower() or
                q_clean in clf["classifier"] or
                q_clean in clf["transcription"].lower()):
                matched.append(clf)
        return {"query": q, "count": len(matched), "results": matched}
    return {"count": len(LAO_CLASSIFIERS_TABLE), "classifiers": LAO_CLASSIFIERS_TABLE}


@app.get("/api/tones", summary="Get tone determination matrix")
def get_tones():
    """Returns the tone determination matrix for live/dead syllables and tone marks."""
    return {
        "dialect": "Vientiane Standard",
        "tone_count": 6,
        "consonant_classes": LAO_CONSONANT_CLASSES,
        "rules_summary": LAO_TONE_RULES_SUMMARY.strip(),
    }


@app.get("/api/consonants", summary="Get consonant class distribution")
def get_consonants():
    """Returns High, Middle, and Low consonant letter mappings."""
    return {
        "classes": LAO_CONSONANT_CLASSES,
        "total_letters": sum(len(v) for v in LAO_CONSONANT_CLASSES.values())
    }


@app.get("/api/search", summary="Search across all Lao grammar rules and knowledge")
def search_knowledge(q: str = Query(..., min_length=2, description="Keywords in English or Lao")):
    """
    Semantic search across all grammar rules, classifiers, tone guidelines,
    and syntactic descriptions.
    """
    query = q.lower().strip()
    matched_rules = []
    for r in LAO_GRAMMAR_RULES:
        haystack = f"{r['category']} {r['rule_name']} {r['lao_name']} {r['explanation']} {r['formula']}".lower()
        for eg in r["examples"]:
            haystack += f" {eg['lao']} {eg['translation']} {eg['gloss']}".lower()
        if query in haystack:
            matched_rules.append(r)

    matched_classifiers = []
    for c in LAO_CLASSIFIERS_TABLE:
        haystack = f"{c['classifier']} {c['transcription']} {c['usage']}".lower()
        if query in haystack:
            matched_classifiers.append(c)

    return {
        "query": q,
        "matched_rules_count": len(matched_rules),
        "matched_rules": matched_rules,
        "matched_classifiers_count": len(matched_classifiers),
        "matched_classifiers": matched_classifiers
    }


class ValidationRequest(BaseModel):
    sentence: str
    target_rule: Optional[str] = None


@app.post("/api/validate", summary="Validate Lao sentence structure & check classifier usage")
def validate_lao_sentence(req: ValidationRequest):
    """
    Analyzes a Lao sentence for common grammatical patterns, script composition,
    and detected classifier usage.
    """
    text = req.sentence.strip()
    analysis = analyze_lao_script(text)
    has_lao = analysis["total_lao_chars"] > 0

    # Check for classifier pattern: Number + Classifier
    detected_classifiers = []
    num_clf_pattern = re.compile(r"([໐-໙0-9]+|ໜຶ່ງ|ສອງ|ສາມ|ສີ່|ຫ້າ|ຫົກ|ເຈັດ|ແປດ|ເກົ້າ|ສິບ)\s*([ກ-ໝ]{1,5})")
    for m in num_clf_pattern.finditer(text):
        num, clf_candidate = m.groups()
        for clf in LAO_CLASSIFIERS_TABLE:
            if clf["classifier"] == clf_candidate:
                detected_classifiers.append({
                    "matched": m.group(0),
                    "number": num,
                    "classifier": clf["classifier"],
                    "usage": clf["usage"]
                })

    # Check negation placement
    has_negation = "ບໍ່" in text
    has_question_particle = text.endswith("ບໍ່?") or text.endswith("ບໍ່") or "ຫວາ" in text

    feedback = []
    if detected_classifiers:
        feedback.append(f"Detected {len(detected_classifiers)} classifier construct(s) following standard [Noun + Numeral + Classifier] structure.")
    if has_negation:
        feedback.append("Contains negation particle 'ບໍ່' (ensure it precedes the main verb).")
    if has_question_particle:
        feedback.append("Contains interrogative sentence structure ending with question particle.")

    return {
        "sentence": text,
        "script_analysis": analysis,
        "detected_classifiers": detected_classifiers,
        "syntactic_feedback": feedback,
        "valid_lao_script": has_lao,
    }


# -----------------------------------------------------------------------------
# User-Friendly Web Interface
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Interactive Web Interface")
def home_page():
    """Renders an elegant, modern web interface with search, rule cards, and LLM documentation."""
    classifiers_rows = "".join(
        f"""
        <tr class="border-b border-slate-700 hover:bg-slate-800/50 transition">
            <td class="px-4 py-3 font-semibold text-emerald-400 text-lg">{c['classifier']}</td>
            <td class="px-4 py-3 text-slate-300 font-mono text-sm">{c['transcription']}</td>
            <td class="px-4 py-3 text-slate-200">{c['usage']}</td>
        </tr>
        """
        for c in LAO_CLASSIFIERS_TABLE
    )

    rules_cards = "".join(
        f"""
        <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-5 hover:border-emerald-500/50 transition shadow-lg">
            <div class="flex justify-between items-start mb-2">
                <span class="text-xs uppercase tracking-wider px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 font-semibold">{r['category']}</span>
                <span class="text-slate-400 font-mono text-xs">{r['lao_name']}</span>
            </div>
            <h3 class="text-lg font-bold text-white mb-2">{r['rule_name']}</h3>
            <p class="text-slate-300 text-sm mb-3">{r['explanation']}</p>
            <div class="bg-slate-900/80 rounded-lg p-2.5 mb-3 font-mono text-xs text-amber-300 border border-slate-800">
                Formula: {r['formula']}
            </div>
            <div class="space-y-1.5 text-xs">
                <div class="font-semibold text-slate-400">Example:</div>
                <div class="text-emerald-300 font-medium text-sm">{r['examples'][0]['lao']} <span class="text-slate-400 font-normal">({r['examples'][0]['transcription']})</span></div>
                <div class="text-slate-400 italic">"{r['examples'][0]['translation']}"</div>
            </div>
        </div>
        """
        for r in LAO_GRAMMAR_RULES
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lao Language & Grammar API for LLMs (ພາສາລາວ)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Noto+Sans+Lao:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', 'Noto Sans Lao', sans-serif; }}
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
    <!-- Navigation -->
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-3xl">🇱🇦</span>
                <div>
                    <h1 class="text-lg font-bold text-white">Lao Grammar API for LLMs</h1>
                    <p class="text-xs text-slate-400">Knowledge Server for AI Agents & Models</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <a href="/llms.txt" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-emerald-400 border border-slate-700 transition">/llms.txt</a>
                <a href="/llms-full.txt" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-amber-400 border border-slate-700 transition">/llms-full.txt</a>
                <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white transition shadow-sm">Swagger Docs</a>
            </div>
        </div>
    </header>

    <!-- Hero Section -->
    <main class="max-w-6xl mx-auto px-4 py-8 space-y-12">
        <section class="text-center space-y-4 max-w-3xl mx-auto pt-4">
            <div class="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                <span>⚡ Built for LLMs, RAG & Function Calling</span>
            </div>
            <h2 class="text-4xl font-extrabold text-white tracking-tight sm:text-5xl">
                Fetch and Understand <span class="text-emerald-400">Lao Grammar Rules</span>
            </h2>
            <p class="text-slate-400 text-base">
                Zero GPU overhead, zero fan noise. AI models (ChatGPT, Claude, Gemma, LLaMA) can query this live server via REST endpoints or ingest standard <code class="text-emerald-400">/llms.txt</code> markdown documentation.
            </p>

            <!-- Search Bar -->
            <div class="pt-4 max-w-xl mx-auto">
                <div class="relative">
                    <input id="searchInput" type="text" placeholder="Search rules (e.g. classifier, SVO, negation, tone, ໂຕ)..." 
                           class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 text-slate-100 placeholder-slate-500 shadow-inner">
                    <button onclick="performSearch()" class="absolute right-2 top-2 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-medium transition">
                        Search
                    </button>
                </div>
                <div id="searchResults" class="mt-3 text-left space-y-2 hidden"></div>
            </div>
        </section>

        <!-- LLM Agent Integration Info -->
        <section class="bg-gradient-to-r from-slate-900 to-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 class="text-lg font-bold text-white mb-3 flex items-center space-x-2">
                <span>🤖 How Any LLM Can Fetch From This Website</span>
            </h3>
            <div class="grid md:grid-cols-3 gap-4 text-xs">
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
                    <div class="font-bold text-emerald-400 mb-1">1. LLM Context Ingestion</div>
                    <p class="text-slate-400 leading-relaxed">
                        LLMs like Perplexity, Claude, or ChatGPT can curl <code class="text-slate-200">/llms-full.txt</code> to load the entire Lao grammar manual into system prompt or RAG context.
                    </p>
                </div>
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
                    <div class="font-bold text-emerald-400 mb-1">2. Function / Tool Calling</div>
                    <p class="text-slate-400 leading-relaxed">
                        Configure your AI agent with a tool pointing to <code class="text-slate-200">GET /api/classifiers?q=animal</code> to resolve classifiers dynamically.
                    </p>
                </div>
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
                    <div class="font-bold text-emerald-400 mb-1">3. Grammar Verification</div>
                    <p class="text-slate-400 leading-relaxed">
                        Call <code class="text-slate-200">POST /api/validate</code> with any Lao sentence to verify classifier syntax and detect negation and interrogative particles.
                    </p>
                </div>
            </div>
        </section>

        <!-- Grammar Rules Section -->
        <section class="space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <h3 class="text-xl font-bold text-white">Lao Grammar Rule Knowledge Base</h3>
                    <p class="text-xs text-slate-400">Formal linguistic rules and syntactic constraints</p>
                </div>
                <a href="/api/rules" target="_blank" class="text-xs text-emerald-400 hover:underline font-mono">View raw JSON &rarr;</a>
            </div>
            <div class="grid md:grid-cols-2 gap-4">
                {rules_cards}
            </div>
        </section>

        <!-- Numerical Classifiers Section -->
        <section class="space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <h3 class="text-xl font-bold text-white">Lao Numerical Classifiers (ລັກສະນະນາມ)</h3>
                    <p class="text-xs text-slate-400">Required semantic counters: Noun + Number + Classifier</p>
                </div>
                <a href="/api/classifiers" target="_blank" class="text-xs text-emerald-400 hover:underline font-mono">View raw JSON &rarr;</a>
            </div>
            <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-xl">
                <table class="w-full text-left text-sm">
                    <thead class="bg-slate-900 text-xs uppercase tracking-wider text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="px-4 py-3">Classifier (ລັກສະນະນາມ)</th>
                            <th class="px-4 py-3">Romanization</th>
                            <th class="px-4 py-3">Semantic Usage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {classifiers_rows}
                    </tbody>
                </table>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800 mt-16 py-8 text-center text-xs text-slate-500">
        <p>Lao Language & Grammar AI Service • Zero GPU Requirements • Ready for AI Agents</p>
    </footer>

    <script>
        async function performSearch() {{
            const query = document.getElementById('searchInput').value.trim();
            const container = document.getElementById('searchResults');
            if (!query) return;

            try {{
                const res = await fetch(`/api/search?q=${{encodeURIComponent(query)}}`);
                const data = await res.json();
                container.innerHTML = '';
                container.classList.remove('hidden');

                if (data.matched_rules_count === 0 && data.matched_classifiers_count === 0) {{
                    container.innerHTML = `<div class="p-3 bg-slate-900 border border-slate-800 rounded-lg text-slate-400 text-xs">No matching rules or classifiers found.</div>`;
                    return;
                }}

                let html = `<div class="p-3 bg-slate-900 border border-slate-800 rounded-xl space-y-2 text-xs">`;
                html += `<div class="font-bold text-emerald-400">Search Results for "${{data.query}}":</div>`;

                data.matched_rules.forEach(r => {{
                    html += `<div class="p-2 bg-slate-950 rounded border border-slate-800">
                                <span class="text-amber-400 font-semibold">[Rule] ${{r.rule_name}}</span> - ${{r.explanation}}
                             </div>`;
                }});

                data.matched_classifiers.forEach(c => {{
                    html += `<div class="p-2 bg-slate-950 rounded border border-slate-800">
                                <span class="text-emerald-400 font-semibold">[Classifier] ${{c.classifier}} (${{c.transcription}})</span>: ${{c.usage}}
                             </div>`;
                }});
                html += `</div>`;
                container.innerHTML = html;
            }} catch (err) {{
                console.error(err);
            }}
        }}

        document.getElementById('searchInput').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') performSearch();
        }});
    </script>
</body>
</html>
"""
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
