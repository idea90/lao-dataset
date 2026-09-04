"""
Authoritative Lao linguistic & grammar knowledge base.
Provides structured rules for Lao phonology, tone determination,
classifiers, syntax, tense/aspect, particles, and AI dataset generation.
"""

from typing import Any, Dict, List

# Core Lao Grammar Knowledge System
LAO_GRAMMAR_RULES: List[Dict[str, Any]] = [
    {
        "category": "Syntax & Word Order",
        "rule_name": "Canonical SVO Word Order",
        "lao_name": "ໂຄງສ້າງປະໂຫຍກ (ປະທານ + ກິລິຍາ + ກຳ)",
        "explanation": "Lao is strictly an SVO (Subject-Verb-Object) language. Modifiers follow the head word they modify.",
        "formula": "Subject + Verb + Object",
        "examples": [
            {
                "lao": "ຂ້ອຍກິນເຂົ້າ",
                "transcription": "Khoy kin khao",
                "gloss": "I [Subject] + eat [Verb] + rice/food [Object]",
                "translation": "I eat rice."
            },
            {
                "lao": "ລາວອ່ານປຶ້ມ",
                "transcription": "Lao aan puem",
                "gloss": "He/She [Subject] + read [Verb] + book [Object]",
                "translation": "He/She reads a book."
            }
        ]
    },
    {
        "category": "Noun Phrase Structure",
        "rule_name": "Head-Initial Noun Phrases",
        "lao_name": "ໂຄງສ້າງວະລີນາມ",
        "explanation": "In Lao, adjectives and possessives follow the noun they modify.",
        "formula": "Noun + Adjective / Possessive",
        "examples": [
            {
                "lao": "ຫມາໃຫຍ່",
                "transcription": "Ma nyai",
                "gloss": "Dog + big",
                "translation": "Big dog"
            },
            {
                "lao": "ປຶ້ມຂອງຂ້ອຍ",
                "transcription": "Puem khong khoy",
                "gloss": "Book + of + me",
                "translation": "My book"
            },
            {
                "lao": "ລົດສີດຳ",
                "transcription": "Lot si dam",
                "gloss": "Car + color + black",
                "translation": "Black car"
            }
        ]
    },
    {
        "category": "Classifiers (ລັກສະນະນາມ)",
        "rule_name": "Counting and Specification with Classifiers",
        "lao_name": "ການໃຊ້ລັກສະນະນາມກັບຈຳນວນນັບ",
        "explanation": "Numbers cannot modify nouns directly in Lao. They require a classifier matching the semantic category of the noun. Pattern: [Noun] + [Numeral] + [Classifier]. For demonstratives: [Noun] + [Classifier] + [Demonstrative].",
        "formula": "Noun + Number + Classifier  OR  Noun + Classifier + Demonstrative (ນີ້/ນັ້ນ)",
        "examples": [
            {
                "lao": "ຫມາ ສອງ ໂຕ",
                "transcription": "Ma song to",
                "gloss": "Dog + two + [animal classifier]",
                "translation": "Two dogs"
            },
            {
                "lao": "ປຶ້ມ ສາມ ເຫຼັ້ມ",
                "transcription": "Puem sam lem",
                "gloss": "Book + three + [book/blade classifier]",
                "translation": "Three books"
            },
            {
                "lao": "ຄົນ ຜູ້ ນີ້",
                "transcription": "Khon phu ni",
                "gloss": "Person + [person classifier] + this",
                "translation": "This person"
            },
            {
                "lao": "ລົດ ຄັນ ນັ້ນ",
                "transcription": "Lot khan nan",
                "gloss": "Car + [vehicle classifier] + that",
                "translation": "That car"
            }
        ]
    },
    {
        "category": "Tense & Aspect Markers",
        "rule_name": "Aspectual Particles (Past, Continuous, Future)",
        "lao_name": "ຄຳຊ່ວຍບອກການ ແລະ ລັກສະນະ",
        "explanation": "Lao verbs do not conjugate for tense. Temporal aspects are marked with pre-verbal or post-verbal particles: ໄດ້ (past/achieved), ແລ້ວ (already/completed), ກຳລັງ/ພວມ (progressive), ຢູ່ (continuous), ຈະ/ຊິ (future).",
        "formula": "Past: ໄດ້ + Verb (or Verb + ແລ້ວ) | Progressive: ກຳລັງ + Verb (+ ຢູ່) | Future: ຈະ/ຊິ + Verb",
        "examples": [
            {
                "lao": "ຂ້ອຍໄດ້ໄປວຽງຈັນ",
                "transcription": "Khoy dai pai Viengchan",
                "gloss": "I + [past marker] + go + Vientiane",
                "translation": "I went to Vientiane."
            },
            {
                "lao": "ຂ້ອຍກິນເຂົ້າແລ້ວ",
                "transcription": "Khoy kin khao laeo",
                "gloss": "I + eat + rice + [completed marker]",
                "translation": "I already ate."
            },
            {
                "lao": "ລາວກຳລັງຮຽນໜັງສືຢູ່",
                "transcription": "Lao kam lang hian nang sue yu",
                "gloss": "He + [progressive] + study + book + [continuous]",
                "translation": "He is currently studying."
            },
            {
                "lao": "ພວກເຮົາຈະໄປຕະຫຼາດມື້ອື່ນ",
                "transcription": "Phuok hao ja pai talat mue uen",
                "gloss": "We + will + go + market + tomorrow",
                "translation": "We will go to the market tomorrow."
            }
        ]
    },
    {
        "category": "Negation",
        "rule_name": "Negation Particle ບໍ່ (Bor)",
        "lao_name": "ການປະຕິເສດ",
        "explanation": "To negate a predicate, place ບໍ່ (bor) directly before the verb or adjective. For imperative prohibitions, use ຢ່າ (ya).",
        "formula": "ບໍ່ + Verb / Adjective  OR  ຢ່າ + Verb",
        "examples": [
            {
                "lao": "ຂ້ອຍບໍ່ຮູ້",
                "transcription": "Khoy bor hu",
                "gloss": "I + not + know",
                "translation": "I don't know."
            },
            {
                "lao": "ອາຫານນີ້ບໍ່ແຊບ",
                "transcription": "Ahan ni bor saeb",
                "gloss": "Food + this + not + delicious",
                "translation": "This food is not delicious."
            },
            {
                "lao": "ຢ່າໄປ",
                "transcription": "Ya pai",
                "gloss": "Don't + go",
                "translation": "Don't go!"
            }
        ]
    },
    {
        "category": "Interrogatives (Questions)",
        "rule_name": "Question Particle Formation",
        "lao_name": "ປະໂຫຍກຄຳຖາມ",
        "explanation": "Yes/no questions place ບໍ່ (bor) at the end. Question words (wh-words) remain in situ (in the natural position of the information requested).",
        "formula": "Statement + ບໍ່?  OR  [SVO with wh-word in place]",
        "examples": [
            {
                "lao": "ເຈົ້າສະບາຍດີບໍ່?",
                "transcription": "Jao sabai dee bor?",
                "gloss": "You + comfortable/well + [question particle]?",
                "translation": "How are you? / Are you well?"
            },
            {
                "lao": "ເຈົ້າຊື່ຫຍັງ?",
                "transcription": "Jao xue nyang?",
                "gloss": "You + name + what?",
                "translation": "What is your name?"
            },
            {
                "lao": "ຫ້ອງນ້ຳຢູ່ໃສ?",
                "transcription": "Hong nam yu sai?",
                "gloss": "Bathroom + located + where?",
                "translation": "Where is the restroom?"
            },
            {
                "lao": "ລາວແມ່ນໃຜ?",
                "transcription": "Lao maen phai?",
                "gloss": "He/She + is + who?",
                "translation": "Who is he/she?"
            }
        ]
    },
    {
        "category": "Nominalization (ການສ້າງຄຳນາມ)",
        "rule_name": "Prefixes ການ- (Kan-) and ຄວາມ- (Khuam-)",
        "lao_name": "ການປ່ຽນຄຳກິລິຍາ ແລະ ຄຸນນາມ ເປັນຄຳນາມ",
        "explanation": "Verbs of action take the prefix ການ- (kan-) to form nouns (actions/activities). Adjectives, states, or emotions take the prefix ຄວາມ- (khuam-) to form abstract nouns.",
        "formula": "ການ + Action Verb  |  ຄວາມ + Adjective/State",
        "examples": [
            {
                "lao": "ການຮຽນ",
                "transcription": "Kan hian",
                "gloss": "Prefix:action + learn",
                "translation": "Education / Study / Learning"
            },
            {
                "lao": "ການເດີນທາງ",
                "transcription": "Kan doenthang",
                "gloss": "Prefix:action + travel",
                "translation": "Travel / Journey"
            },
            {
                "lao": "ຄວາມຮັກ",
                "transcription": "Khuam hak",
                "gloss": "Prefix:state + love",
                "translation": "Love (noun)"
            },
            {
                "lao": "ຄວາມສຸກ",
                "transcription": "Khuam suk",
                "gloss": "Prefix:state + happy",
                "translation": "Happiness"
            }
        ]
    },
    {
        "category": "Phonology & Tone Determination",
        "rule_name": "Consonant Class Tone Matrix",
        "lao_name": "ກົດເກນການຜັນສຽງວັນນະຍຸດ",
        "explanation": "In Lao, the tone of a syllable is determined by 3 factors: (1) Consonant Class (High, Mid, Low), (2) Syllable type (Live vs Dead, Vowel length), and (3) Tone mark (ໄມ້ເອກ, ໄມ້ໂທ).",
        "formula": "Consonant Class + Syllable Ending + Tone Mark -> Resulting Tone (1 of 6 tones)",
        "examples": [
            {
                "lao": "ກາ (Middle + Live + No mark)",
                "transcription": "Kaa",
                "gloss": "Mid class + long vowel live syllable",
                "translation": "Tone: Low tone"
            },
            {
                "lao": "ຂາ (High + Live + No mark)",
                "transcription": "Khaa",
                "gloss": "High class + long vowel live syllable",
                "translation": "Tone: Rising tone"
            },
            {
                "lao": "ຄາ (Low + Live + No mark)",
                "transcription": "Khaa",
                "gloss": "Low class + long vowel live syllable",
                "translation": "Tone: High/Mid tone"
            }
        ]
    }
]

# Comprehensive Table of Lao Classifiers
LAO_CLASSIFIERS_TABLE = [
    {"classifier": "ໂຕ", "transcription": "to", "usage": "Animals, items with legs/arms (shirts, pants, chairs, tables), letters/characters"},
    {"classifier": "ຄົນ", "transcription": "khon", "usage": "General people, individuals"},
    {"classifier": "ອົງ", "transcription": "ong", "usage": "Monks, royalty, Buddha statues, deities"},
    {"classifier": "ເຫຼັ້ມ", "transcription": "lem", "usage": "Books, notebooks, knives, needles, candles"},
    {"classifier": "ໃບ", "transcription": "bai", "usage": "Leaves, sheets of paper, tickets, documents, fruits, bags, glasses/cups"},
    {"classifier": "ຄັນ", "transcription": "khan", "usage": "Vehicles (cars, bicycles), umbrellas, spoons, forks"},
    {"classifier": "ຫຼັງ", "transcription": "lang", "usage": "Houses, buildings, mosquito nets"},
    {"classifier": "ເສັ້ນ", "transcription": "sen", "usage": "Linear items: roads, hairs, threads, strings, noodles"},
    {"classifier": "ໜ່ວຍ", "transcription": "nuay", "usage": "Round or three-dimensional objects: fruits, eggs, balls, organs, mountains"},
    {"classifier": "ດວງ", "transcription": "duang", "usage": "Sources of light/luminaries (sun, moon, stars), stamps, seals, hearts"},
    {"classifier": "ແຜ່ນ", "transcription": "phaen", "usage": "Flat sheets: planks of wood, paper boards, CD discs, mats"},
    {"classifier": "ຄູ່", "transcription": "khu", "usage": "Pairs: shoes, socks, earrings, chopsticks"},
    {"classifier": "ສາຍ", "transcription": "sai", "usage": "Rivers, ropes, belts, necklaces, bus routes"},
    {"classifier": "ຮູບ", "transcription": "hup", "usage": "Pictures, photographs, portrait images"},
    {"classifier": "ດອກ", "transcription": "dok", "usage": "Flowers, keys"}
]

# Consonant Class Mapping
LAO_CONSONANT_CLASSES = {
    "High (ອັກສອນສູງ)": ["ຂ", "ສ", "ຖ", "ຜ", "ຝ", "ຫ", "ໝ", "ໜ", "ຫລ", "ຫຼ"],
    "Middle (ອັກສອນກາງ)": ["ກ", "ຈ", "ດ", "ຕ", "ບ", "ປ", "ຢ", "ອ"],
    "Low (ອັກສອນຕ່ຳ)": ["ຄ", "ງ", "ຊ", "ຍ", "ທ", "ນ", "ພ", "ຟ", "ມ", "ຣ", "ລ", "ວ", "ຮ"]
}
