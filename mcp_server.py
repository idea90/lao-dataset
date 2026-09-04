"""
Model Context Protocol (MCP) Server for Lao Language & Grammar Knowledge.

Configures any connected LLM (Claude, Cursor, etc.) to act as a fluent, native Lao speaker:
- System Instructions: Enforces STRICT EXCLUSIVE LAO RESPONSE POLICY.
  Even if the user writes or speaks in English, the AI NEVER switches back to English!
- Tools:
    1. get_lao_classifier: Look up the correct classifier (ລັກສະນະນາມ) for any noun.
    2. get_grammar_rules: Retrieve formal syntax and word order rules.
    3. get_tone_rules: Get the 6-tone determination rules for High, Middle, and Low consonants.
    4. validate_lao_sentence: Analyze a Lao sentence for classifier syntax and grammatical patterns.
    5. fluent_lao_phrasing: Provides natural, native conversational phrasing and honorifics.
- Prompts:
    - fluent_lao_persona: One-click prompt that switches the AI into a fluent Lao assistant.
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

# Core Native Lao Fluency Instructions provided to any connected LLM
FLUENT_LAO_INSTRUCTIONS = """
You are a fluent, polite, culturally natural native Lao speaker (ພາສາລາວ) and expert linguistic assistant.

CRITICAL LANGUAGE POLICY (MANDATORY AND ABSOLUTE):
- ALWAYS RESPOND EXCLUSIVELY IN FLUENT LAO (ພາສາລາວ).
- EVEN IF THE USER SPEAKS, ASKS, WRITES, OR PROMPTS IN ENGLISH, NEVER SWITCH BACK TO ENGLISH!
- You MUST answer all English questions, user queries, instructions, or coding questions entirely in Lao.
- If specific English technical terms, programming keywords, or proper nouns are required, place them inside parentheses or quotes (e.g. "Python" ຫຼື "Database"), but explain and write all sentences in natural, fluent Lao.

Follow these strict rules when speaking Lao:
1. **Natural Lao Orthography (No Spaces Between Words)**:
   - Write Lao script smoothly WITHOUT spaces between words (e.g. "ຂ້ອຍກິນເຂົ້າແລ້ວ", NEVER "ຂ້ອຍ ກິນ ເຂົ້າ ແລ້ວ").
   - Spaces are used ONLY as punctuation to separate clauses, complete thoughts, or sentence pauses.
2. **Polite Particles & Natural Register**:
   - For acknowledgment / affirmative: always begin with "ໂດຍ" (doi) or "ເຈົ້າ" (jao).
   - Softening particles at end of sentences: use "ເດີ" (doe - for requests/reminders), "ນໍ" (nor - seeking agreement), "ເນາະ" (noh).
   - Friendly greeting: "ສະບາຍດີ" (sabai dee).
   - Thank you: "ຂອບໃຈຫຼາຍໆ" (khop jai lai lai).
3. **Mandatory Classifiers (ລັກສະນະນາມ)**:
   - When counting or quantifying items, ALWAYS use the structure: [Noun] + [Number] + [Classifier].
   - Examples:
     * Dogs/Cats/Cars: "ໝາ 2 ໂຕ" (dogs 2 [clf]), "ລົດ 1 ຄັນ" (car 1 [clf]).
     * Books: "ປຶ້ມ 3 ຫົວ" (books 3 [clf]).
     * Flat things: "ເຈ້ຍ 5 ແຜ່ນ" (paper 5 [clf]).
4. **Natural Sentence Flow**:
   - Avoid awkward literal English translations. Use idiomatic Lao phrasing (e.g. use "ບໍ່ເປັນຫຍັງ" for you're welcome / no problem; "ກິນເຂົ້າແລ້ວບໍ່?" for how are you / have you eaten?).
5. **Honorifics & Pronouns**:
   - Self (polite): "ຂ້ອຍ" (khoy).
   - You (polite/respectful): "ເຈົ້າ" (jao) or "ທ່ານ" (than - formal).
   - Third person: "ລາວ" (lao) or "ເພິ່ນ" (phoen - respectful).
"""

# Initialize MCP Server with explicit instructions
mcp = MCPServer(
    name="Lao-Grammar-Knowledge-Server",
    instructions=FLUENT_LAO_INSTRUCTIONS
)


# =============================================================================
# MCP PROMPTS (For 1-Click Fluent Lao Persona)
# =============================================================================

@mcp.prompt()
def fluent_lao_persona(topic: str = "general conversation") -> str:
    """Prompt template that activates fluent, native Lao conversational mode."""
    return f"""ເຈົ້າເປັນຜູ້ຊ່ວຍ AI ທີ່ເວົ້າພາສາລາວໄດ້ຢ່າງຄ່ອງແຄ້ວ, ສຸພາບ ແລະ ເປັນທຳມະຊາດ (Fluent Native Lao Speaker).

ກົດລະບຽບສຳຄັນທີ່ສຸດ:
- ຕອບ ແລະ ສົນທະນາເປັນ "ພາສາລາວ" ພຽງຢ່າງດຽວເທົ່ານັ້ນ!
- ເຖິງແມ່ນວ່າຜູ້ໃຊ້ຈະຖາມ ຫຼື ເວົ້າເປັນພາສາອັງກິດ (English), ເຈົ້າກໍຕ້ອງຕອບກັບເປັນພາສາລາວສະເໝີ, ຫ້າມປ່ຽນກັບໄປເວົ້າພາສາອັງກິດຢ່າງເດັດຂາດ.
- ຂຽນພາສາລາວໃຫ້ຖືກຕ້ອງຕາມຫຼັກໄວຍາກອນ, ບໍ່ຍະຫວ່າງລະຫວ່າງຄຳ, ແລະ ໃຊ້ລັກສະນະນາມຢ່າງຖືກຕ້ອງ.

ຫົວຂໍ້ການສົນທະນາ: {topic}
"""


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def fluent_lao_phrasing(intent: str) -> str:
    """
    Look up natural, idiomatic, fluent Lao expressions and polite phrasing for various intents.
    
    Args:
        intent: The conversational intent (e.g. 'greeting', 'thanks', 'apology', 'farewell', 'agreement', 'ordering_food').
    """
    phrases = {
        "greeting": [
            {"lao": "ສະບາຍດີ", "meaning": "Hello / Good day (standard polite greeting)"},
            {"lao": "ສະບາຍດີຕອນເຊົ້າ", "meaning": "Good morning"},
            {"lao": "ກິນເຂົ້າແລ້ວບໍ່?", "meaning": "Have you eaten yet? (Common friendly informal greeting)"},
        ],
        "thanks": [
            {"lao": "ຂອບໃຈຫຼາຍໆເດີ", "meaning": "Thank you very much (warm and polite)"},
            {"lao": "ຂອບໃຈເດີ້", "meaning": "Thanks! (casual, friendly)"},
            {"lao": "ຍິນດີຮັບໃຊ້", "meaning": "Happy to serve / At your service"},
        ],
        "apology": [
            {"lao": "ຂໍໂທດຫຼາຍໆເດີ", "meaning": "I'm very sorry (sincere apology)"},
            {"lao": "ບໍ່ເປັນຫຍັງ", "meaning": "No problem / That's okay / You're welcome"},
        ],
        "agreement": [
            {"lao": "ໂດຍ, ແມ່ນແລ້ວ", "meaning": "Yes, that's correct (polite)"},
            {"lao": "ເຈົ້າ, ຖືກຕ້ອງ", "meaning": "Yes, exactly"},
            {"lao": "ເຫັນດີນຳ", "meaning": "I agree with you"},
        ],
        "farewell": [
            {"lao": "ໂຊກດີເດີ", "meaning": "Good luck / Take care"},
            {"lao": "ແລ້ວພົບກັນໃໝ່ເດີ", "meaning": "See you again soon"},
            {"lao": "ໄປກ່ອນເດີ", "meaning": "I'm heading out now (casual)"},
        ],
        "ordering_food": [
            {"lao": "ຂໍສັ່ງອາຫານແດ່ເດີ", "meaning": "May I order food please?"},
            {"lao": "ເອົາເຝີງົວ 1 ຖ້ວຍແດ່", "meaning": "I'd like 1 bowl of beef pho please (uses classifier 'ຖ້ວຍ')"},
            {"lao": "ຄິດເງິນແດ່", "meaning": "Check please / Bill please"},
        ]
    }

    intent_clean = intent.lower().strip()
    matched = None
    for k, v in phrases.items():
        if k in intent_clean or intent_clean in k:
            matched = (k, v)
            break

    if not matched:
        out = f"### Native Lao Phrasing Guidelines:\n"
        for cat, list_p in phrases.items():
            out += f"\n**{cat.capitalize()}**:\n"
            for p in list_p:
                out += f"- **{p['lao']}**: {p['meaning']}\n"
        return out

    out = f"### Fluent Lao Phrasing for '{matched[0]}':\n\n"
    for p in matched[1]:
        out += f"- **{p['lao']}** — *{p['meaning']}*\n"
    return out


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
        res += f"  - Counting pattern: [Noun] + [Numeral] + **{c['classifier']}** (e.g. ສອງ{c['classifier']})\n\n"
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
