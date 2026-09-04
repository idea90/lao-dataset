"""
Static Site & Static API Generator for GitHub Pages.

Builds a fully static deployment in the `docs/` folder compatible with GitHub Pages:
- docs/index.html (Interactive search, classifier tables, rule cards, client-side JS)
- docs/llms.txt (LLM web standard summary)
- docs/llms-full.txt (Full Lao grammar manual for LLMs in markdown)
- docs/api/rules.json (All formal grammar rules)
- docs/api/classifiers.json (All Lao numerical classifiers)
- docs/api/tones.json (Tone determination rules & consonant matrix)
- docs/api/consonants.json (High, Middle, Low consonant breakdown)
- docs/.nojekyll (Ensures GitHub Pages serves all files and directories without Jekyll processing)
"""

import json
import os
import shutil
import sys
from pathlib import Path

from core.knowledge_base import (
    LAO_GRAMMAR_RULES,
    LAO_CLASSIFIERS_TABLE,
    LAO_CONSONANT_CLASSES,
)
from server import LAO_TONE_RULES_SUMMARY, get_llms_txt, get_llms_full_txt

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
API_DIR = DOCS_DIR / "api"


def build_github_pages():
    print(f"[*] Building GitHub Pages site in: {DOCS_DIR}...")

    # Create directories
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Create .nojekyll (important for GitHub Pages)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")

    # 2. Build llms.txt and llms-full.txt
    (DOCS_DIR / "llms.txt").write_text(get_llms_txt(), encoding="utf-8")
    (DOCS_DIR / "llms-full.txt").write_text(get_llms_full_txt(), encoding="utf-8")
    print("  + Created docs/llms.txt and docs/llms-full.txt")

    # 3. Export Static JSON API Endpoints
    with open(API_DIR / "rules.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(LAO_GRAMMAR_RULES), "rules": LAO_GRAMMAR_RULES}, f, ensure_ascii=False, indent=2)

    with open(API_DIR / "classifiers.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(LAO_CLASSIFIERS_TABLE), "classifiers": LAO_CLASSIFIERS_TABLE}, f, ensure_ascii=False, indent=2)

    with open(API_DIR / "tones.json", "w", encoding="utf-8") as f:
        json.dump({
            "dialect": "Vientiane Standard",
            "tone_count": 6,
            "consonant_classes": LAO_CONSONANT_CLASSES,
            "rules_summary": LAO_TONE_RULES_SUMMARY.strip()
        }, f, ensure_ascii=False, indent=2)

    with open(API_DIR / "consonants.json", "w", encoding="utf-8") as f:
        json.dump({
            "classes": LAO_CONSONANT_CLASSES,
            "total_letters": sum(len(v) for v in LAO_CONSONANT_CLASSES.values())
        }, f, ensure_ascii=False, indent=2)
    print("  + Created static API endpoints in docs/api/*.json")

    # 4. Generate Interactive index.html
    rules_cards = "".join(
        f"""
        <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-5 hover:border-emerald-500/50 transition shadow-lg rule-card" data-category="{r['category'].lower()}" data-search="{r['rule_name'].lower()} {r['explanation'].lower()} {r['lao_name']}">
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

    classifiers_rows = "".join(
        f"""
        <tr class="border-b border-slate-700 hover:bg-slate-800/50 transition classifier-row" data-search="{c['classifier']} {c['transcription'].lower()} {c['usage'].lower()}">
            <td class="px-4 py-3 font-semibold text-emerald-400 text-lg">{c['classifier']}</td>
            <td class="px-4 py-3 text-slate-300 font-mono text-sm">{c['transcription']}</td>
            <td class="px-4 py-3 text-slate-200">{c['usage']}</td>
        </tr>
        """
        for c in LAO_CLASSIFIERS_TABLE
    )

    index_html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lao Language & Grammar Knowledge Portal for LLMs (ພາສາລາວ)</title>
    <meta name="description" content="Structured linguistic knowledge base, grammar rules, classifiers, and tone matrices for Lao language. Ready for AI agents, LLM context ingestion, and developers.">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Noto+Sans+Lao:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', 'Noto Sans Lao', sans-serif; }}
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 py-4 flex flex-wrap items-center justify-between gap-3">
            <div class="flex items-center space-x-3">
                <span class="text-3xl">🇱🇦</span>
                <div>
                    <h1 class="text-lg font-bold text-white">Lao Grammar Portal for LLMs</h1>
                    <p class="text-xs text-slate-400">Open Linguistic Knowledge Base for AI Models</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                <a href="llms.txt" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-emerald-400 border border-slate-700 transition">/llms.txt</a>
                <a href="llms-full.txt" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-amber-400 border border-slate-700 transition">/llms-full.txt</a>
                <a href="api/rules.json" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-cyan-400 border border-slate-700 transition">rules.json</a>
                <a href="api/classifiers.json" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-pink-400 border border-slate-700 transition">classifiers.json</a>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-6xl mx-auto px-4 py-8 space-y-12">
        <!-- Hero -->
        <section class="text-center space-y-4 max-w-3xl mx-auto pt-4">
            <div class="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                <span>⚡ Hosted on GitHub Pages • Free Static API for LLMs</span>
            </div>
            <h2 class="text-4xl font-extrabold text-white tracking-tight sm:text-5xl">
                Fetch and Understand <span class="text-emerald-400">Lao Grammar</span>
            </h2>
            <p class="text-slate-400 text-base">
                An authoritative linguistic resource designed for AI models (ChatGPT, Claude, Gemma, DeepSeek, Cursor) to fetch, reference, and understand Lao syntactic rules, tone matrices, and numerical classifiers.
            </p>

            <!-- Instant Client-Side Search -->
            <div class="pt-4 max-w-xl mx-auto">
                <div class="relative">
                    <input id="searchInput" type="text" placeholder="Search rules & classifiers (e.g. classifier, SVO, animal, negation, ໂຕ)..." 
                           oninput="filterContent()"
                           class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 text-slate-100 placeholder-slate-500 shadow-inner">
                </div>
            </div>
        </section>

        <!-- LLM Agent Integration Cards -->
        <section class="bg-gradient-to-r from-slate-900 to-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 class="text-lg font-bold text-white mb-3 flex items-center space-x-2">
                <span>🤖 How Any LLM Fetches From This GitHub Page</span>
            </h3>
            <div class="grid md:grid-cols-3 gap-4 text-xs">
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
                    <div class="font-bold text-emerald-400 mb-1">1. Read /llms-full.txt</div>
                    <p class="text-slate-400 leading-relaxed">
                        Provide this site's <code class="text-slate-200">llms-full.txt</code> URL in your custom instructions, system prompt, or RAG pipeline for full linguistic rules.
                    </p>
                </div>
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
                    <div class="font-bold text-emerald-400 mb-1">2. Static REST JSON APIs</div>
                    <p class="text-slate-400 leading-relaxed">
                        Fetch directly from <code class="text-slate-200">api/classifiers.json</code>, <code class="text-slate-200">api/rules.json</code>, or <code class="text-slate-200">api/tones.json</code> via curl or fetch.
                    </p>
                </div>
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
                    <div class="font-bold text-emerald-400 mb-1">3. 100% Free & Fast</div>
                    <p class="text-slate-400 leading-relaxed">
                        Hosted on GitHub's global CDN. Never spins up your computer fans, requires zero backend servers, and has 99.99% uptime.
                    </p>
                </div>
            </div>
        </section>

        <!-- Grammar Rules -->
        <section class="space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <h3 class="text-xl font-bold text-white">Lao Grammar Rule Knowledge Base</h3>
                    <p class="text-xs text-slate-400">Formal linguistic rules and syntactic constraints</p>
                </div>
                <a href="api/rules.json" target="_blank" class="text-xs text-emerald-400 hover:underline font-mono">View raw rules.json &rarr;</a>
            </div>
            <div id="rulesContainer" class="grid md:grid-cols-2 gap-4">
                {rules_cards}
            </div>
        </section>

        <!-- Numerical Classifiers -->
        <section class="space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <h3 class="text-xl font-bold text-white">Lao Numerical Classifiers (ລັກສະນະນາມ)</h3>
                    <p class="text-xs text-slate-400">Mandatory counters: Noun + Number + Classifier</p>
                </div>
                <a href="api/classifiers.json" target="_blank" class="text-xs text-emerald-400 hover:underline font-mono">View raw classifiers.json &rarr;</a>
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
                    <tbody id="classifiersBody">
                        {classifiers_rows}
                    </tbody>
                </table>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800 mt-16 py-8 text-center text-xs text-slate-500">
        <p>Lao Language & Grammar AI Portal • Hosted on GitHub Pages • Open Source</p>
    </footer>

    <script>
        function filterContent() {{
            const query = document.getElementById('searchInput').value.toLowerCase().trim();

            // Filter rule cards
            document.querySelectorAll('.rule-card').forEach(card => {{
                const text = card.getAttribute('data-search') || '';
                card.style.display = (!query || text.includes(query)) ? 'block' : 'none';
            }});

            // Filter classifier rows
            document.querySelectorAll('.classifier-row').forEach(row => {{
                const text = row.getAttribute('data-search') || '';
                row.style.display = (!query || text.includes(query)) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print("  + Created docs/index.html")
    print("[+] Static GitHub Pages build complete!")


if __name__ == "__main__":
    build_github_pages()
