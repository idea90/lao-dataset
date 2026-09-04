"""
Model Context Protocol (MCP) Server for Lao Language & Grammar Knowledge.

Provides tools and resources to Claude Desktop, Cursor, Claude Code, and any MCP-compliant LLM:
- Tools:
    1. get_lao_classifier: Look up the correct classifier (ລັກສະນະນາມ) for any noun.
    2. get_grammar_rules: Retrieve formal syntax and word order rules.
    3. get_tone_rules: Get the 6-tone determination rules for High, Middle, and Low consonants.
    4. validate_lao_sentence: Analyze a Lao sentence for classifier syntax and grammatical patterns.
- Resources:
    - lao://grammar/rules: Complete formal grammar rules.
    - lao://grammar/classifiers: Full table of 15+ numerical classifiers.
    - lao://grammar/tones: Tone determination matrix.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from core.knowledge_base import (
    LAO_GRAMMAR_RULES,
    LAO_CLASSIFIERS_TABLE,
    LAO_CONSONANT_CLASSES,
)
from core.unicode_utils import analyze_lao_script
from server import LAO_TONE_RULES_SUMMARY

# Initialize MCP Server
mcp = MCPServer("Lao-Grammar-Knowledge-Server")


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def get_lao_classifier(noun_or_category: str) -> str:
    """
    Look up the required Lao numerical classifier (ລັກສະນະນາມ) for counting or specifying an object.
    
    Args:
        noun_or_category: English or Lao noun (e.g. 'car', 'animal', 'book', 'person', 'dog', 'ແມວ').
    """
    query = noun_or_category.lower().strip()
    matched = []
    for clf in LAO_CLASSIFIERS_TABLE:
        if (query in clf["usage"].lower() or 
            query in clf["classifier"] or 
            query in clf["transcription"].lower()):
            matched.append(clf)

    if not matched:
        return f"No exact classifier found for '{noun_or_category}'. Note that general objects without a specific classifier often use 'ອັນ' (an), and round objects use 'ໜ່ວຍ' (nuay)."

    res = f"Found {len(matched)} matching classifier(s) for '{noun_or_category}':\n\n"
    for c in matched:
        res += f"- **{c['classifier']}** (Romanization: *{c['transcription']}*)\n"
        res += f"  - Usage: {c['usage']}\n"
        res += f"  - Counting pattern: [Noun] + [Numeral] + **{c['classifier']}**\n\n"
    return res.strip()


@mcp.tool()
def get_grammar_rules(category: Optional[str] = None) -> str:
    """
    Retrieve formal Lao grammar rules, sentence structure formulas, and bilingual examples.
    
    Args:
        category: Optional category filter (e.g. 'Syntax', 'Noun', 'Classifiers', 'Tense', 'Negation', 'Interrogatives').
    """
    rules = LAO_GRAMMAR_RULES
    if category:
        cat_lower = category.lower()
        rules = [r for r in rules if cat_lower in r["category"].lower()]
        if not rules:
            return f"No grammar rules found matching category '{category}'."

    output = []
    for r in rules:
        output.append(f"### {r['category']}: {r['rule_name']} ({r['lao_name']})")
        output.append(f"**Explanation**: {r['explanation']}")
        output.append(f"**Formula**: `{r['formula']}`")
        output.append("**Examples**:")
        for eg in r["examples"]:
            output.append(f"- Lao: **{eg['lao']}** ({eg['transcription']})")
            output.append(f"  - Gloss: {eg['gloss']}")
            output.append(f"  - Translation: \"{eg['translation']}\"")
        output.append("")
    return "\n".join(output)


@mcp.tool()
def get_tone_rules(consonant: Optional[str] = None) -> str:
    """
    Get Lao tone determination rules (Standard Vientiane 6 Tones).
    
    Args:
        consonant: Optional specific Lao consonant letter to check its class (High, Mid, Low).
    """
    res = []
    if consonant:
        c_clean = consonant.strip()
        found_class = None
        for cls_name, letters in LAO_CONSONANT_CLASSES.items():
            if c_clean in letters:
                found_class = cls_name
                break
        if found_class:
            res.append(f"Consonant **{c_clean}** belongs to: **{found_class}**\n")
        else:
            res.append(f"Letter '{c_clean}' was not recognized in standard Lao consonant classes.\n")

    res.append("### Lao Tone Determination Matrix (Vientiane Standard):")
    res.append(LAO_TONE_RULES_SUMMARY.strip())
    return "\n".join(res)


@mcp.tool()
def validate_lao_sentence(sentence: str) -> str:
    """
    Validate a Lao sentence: checks for Lao script ratio, classifier structures,
    negation placement ('ບໍ່'), and question particles.
    
    Args:
        sentence: The Lao sentence text to analyze.
    """
    text = sentence.strip()
    analysis = analyze_lao_script(text)

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

    has_negation = "ບໍ່" in text
    has_question = text.endswith("ບໍ່?") or text.endswith("ບໍ່") or "ຫວາ" in text

    feedback = []
    feedback.append(f"**Lao Characters**: {analysis['total_lao_chars']} (Consonants: {analysis['consonants']}, Vowels: {analysis['vowels']}, Tones: {analysis['tones']})")

    if detected_classifiers:
        for c in detected_classifiers:
            feedback.append(f"- ✓ Detected classifier pattern: `{c['matched']}` (Classifier **{c['classifier']}** for {c['usage']})")
    else:
        feedback.append("- ℹ️ No explicit [Numeral + Classifier] pattern detected.")

    if has_negation:
        feedback.append("- ✓ Contains negation particle 'ບໍ່' (verify it immediately precedes the main verb).")
    if has_question:
        feedback.append("- ✓ Contains interrogative sentence structure ending with question particle.")

    return "\n".join(feedback)


# =============================================================================
# MCP RESOURCES
# =============================================================================

@mcp.resource("lao://grammar/rules")
def resource_rules() -> str:
    """All formal Lao grammar rules in structured JSON."""
    return json.dumps(LAO_GRAMMAR_RULES, ensure_ascii=False, indent=2)


@mcp.resource("lao://grammar/classifiers")
def resource_classifiers() -> str:
    """Catalog of Lao numerical classifiers (ລັກສະນະນາມ) in JSON."""
    return json.dumps(LAO_CLASSIFIERS_TABLE, ensure_ascii=False, indent=2)


@mcp.resource("lao://grammar/tones")
def resource_tones() -> str:
    """Lao tone determination matrix and consonant classes."""
    return json.dumps({
        "consonant_classes": LAO_CONSONANT_CLASSES,
        "rules": LAO_TONE_RULES_SUMMARY.strip()
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
