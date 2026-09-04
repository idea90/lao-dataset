"""
Synthetic Grammar Task & Dataset Augmenter for Lao Language.
Generates:
1. Grammar Error Correction (GEC) training pairs.
2. Fill-in-the-blank (Cloze) classifier and particle drills.
3. Consonant class & tone determination challenges.
4. Syntactic word order and sentence transformation tasks.
"""

import random
from typing import Any, Dict, List
from core.knowledge_base import LAO_CLASSIFIERS_TABLE, LAO_CONSONANT_CLASSES, LAO_GRAMMAR_RULES


class SyntheticGrammarGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.classifiers = LAO_CLASSIFIERS_TABLE
        self.consonants = LAO_CONSONANT_CLASSES

    def generate_classifier_cloze_tasks(self, count: int = 15) -> List[Dict[str, Any]]:
        """Generate fill-in-the-blank classifier selection tasks."""
        tasks = []
        templates = [
            ("ຂ້ອຍມີຫມາ 2 {blank}", "ໂຕ", "animal / creature", ["ຄົນ", "ຄັນ", "ເຫຼັ້ມ"]),
            ("ລາວຊື້ລົດໃຫຍ່ 1 {blank}", "ຄັນ", "vehicle", ["ໂຕ", "ໜ່ວຍ", "ໃບ"]),
            ("ນັກຮຽນອ່ານປຶ້ມ 3 {blank}", "ເຫຼັ້ມ", "book", ["ແຜ່ນ", "ຄັນ", "ຫຼັງ"]),
            ("ແມ່ຊື້ໄຂ່ໄກ່ 10 {blank}", "ໜ່ວຍ", "round object (egg)", ["ໂຕ", "ເສັ້ນ", "ດອກ"]),
            ("ຢູ່ເທິງໂຕະມີເຈ້ຍ 5 {blank}", "ໃບ", "sheet of paper", ["ຄົນ", "ຄັນ", "ອົງ"]),
            ("ຄູບາ 3 {blank} ກຳລັງສັນເຂົ້າ", "ອົງ", "revered monk", ["ຄົນ", "ໂຕ", "ຫຼັງ"]),
            ("ຂ້ອຍໃສ່ເກີບ 1 {blank}", "ຄູ່", "pair of shoes", ["ໜ່ວຍ", "ຄັນ", "ແຜ່ນ"]),
            ("ເຮືອນ 2 {blank} ນີ້ງາມຫຼາຍ", "ຫຼັງ", "house/building", ["ໂຕ", "ເຫຼັ້ມ", "ຄັນ"]),
            ("ມີຄົນ 4 {blank} ລໍຖ້າຢູ່", "ຄົນ", "person", ["ໂຕ", "ອົງ", "ໃບ"]),
            ("ລາວຕັດຜົມ 1 {blank}", "ເສັ້ນ", "hair strand", ["ໜ່ວຍ", "ດວງ", "ຄູ່"]),
            ("ດາວ 1 {blank} ສ່ອງແສງໃນທ້ອງຟ້າ", "ດວງ", "star / celestial light", ["ໂຕ", "ຄັນ", "ໃບ"]),
            ("ຂ້ອຍມີເສື້ອ 3 {blank}", "ໂຕ", "shirt / clothing", ["ຄົນ", "ໜ່ວຍ", "ຫຼັງ"]),
            ("ດອກໄມ້ 5 {blank} ມີກິ່ນຫອມ", "ດອກ", "flower", ["ໂຕ", "ຄັນ", "ເຫຼັ້ມ"]),
            ("ຖະໜົນ 1 {blank} ນີ້ຍາວຫຼາຍ", "ສາຍ", "linear road/route", ["ໜ່ວຍ", "ໃບ", "ອົງ"]),
            ("ລາວຊື້ໝາກກ້ຽງ 4 {blank}", "ໜ່ວຍ", "fruit", ["ຫຼັງ", "ຄັນ", "ຄູ່"]),
        ]

        for idx, (sentence, correct_clf, context, distractors) in enumerate(templates[:count], start=1):
            options = distractors + [correct_clf]
            random.shuffle(options)
            options_str = ", ".join(options)

            instruction = (
                f"Choose the correct Lao classifier (ລັກສະນະນາມ) to fill in the blank for the sentence:\n"
                f"Sentence: \"{sentence.format(blank='[___]')}\"\n"
                f"Options: {options_str}"
            )
            output = (
                f"**Correct Classifier:** **{correct_clf}**\n\n"
                f"**Completed Sentence:** {sentence.format(blank=correct_clf)}\n\n"
                f"**Grammatical Explanation:**\n"
                f"The noun in this sentence represents a {context}. "
                f"In Lao grammar, the classifier **'{correct_clf}'** is specifically required for this semantic class."
            )

            tasks.append({
                "id": f"synth_cloze_{idx:03d}",
                "category": "classifier_cloze",
                "instruction": instruction,
                "input": "",
                "output": output,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ]
            })
        return tasks

    def generate_error_correction_tasks(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate grammar error correction (GEC) training samples."""
        tasks = []
        scenarios = [
            {
                "incorrect": "ຂ້ອຍເຫັນຫມາ ສາມ ຄົນ",
                "correct": "ຂ້ອຍເຫັນຫມາ ສາມ ໂຕ",
                "error_type": "Wrong Classifier (ລັກສະນະນາມບໍ່ຖືກຕ້ອງ)",
                "explanation": "The classifier 'ຄົນ' (khon) is only used for humans. For animals like 'ຫມາ' (dog), the correct classifier is 'ໂຕ' (to)."
            },
            {
                "incorrect": "ໃຫຍ່ຫມາ ໂຕນີ້ແລ່ນໄວ",
                "correct": "ຫມາໃຫຍ່ ໂຕນີ້ແລ່ນໄວ",
                "error_type": "Incorrect Word Order (ໂຄງສ້າງວະລີນາມ)",
                "explanation": "In Lao noun phrases, adjectives follow the noun (Noun + Adjective). 'ຫມາ' (dog) must precede 'ໃຫຍ່' (big)."
            },
            {
                "incorrect": "ຂ້ອຍໄປບໍ່ຕະຫຼາດມື້ນີ້",
                "correct": "ຂ້ອຍບໍ່ໄປຕະຫຼາດມື້ນີ້",
                "error_type": "Misplaced Negation (ການວາງຄຳປະຕິເສດ)",
                "explanation": "The negation marker 'ບໍ່' (bor) must be placed immediately before the main verb 'ໄປ' (go), not after it."
            },
            {
                "incorrect": "ລາວກິນເຂົ້າແລ້ວບໍ່ແມ່ນ",
                "correct": "ລາວກິນເຂົ້າແລ້ວບໍ່?",
                "error_type": "Question Particle Error",
                "explanation": "To form a standard yes/no question in Lao, place the question particle 'ບໍ່?' at the end of the sentence."
            },
            {
                "incorrect": "ລາວຊື້ປຶ້ມ ສອງ ໂຕ",
                "correct": "ລາວຊື້ປຶ້ມ ສອງ ເຫຼັ້ມ",
                "error_type": "Wrong Classifier for Book",
                "explanation": "Books take the classifier 'ເຫຼັ້ມ' (lem), not 'ໂຕ' (to) which is reserved for animals and clothing."
            },
            {
                "incorrect": "ພວກເຮົາຮຽນກຳລັງພາສາລາວ",
                "correct": "ພວກເຮົານກຳລັງຮຽນພາສາລາວ",
                "error_type": "Progressive Aspect Particle Position",
                "explanation": "The aspectual marker 'ກຳລັງ' (kam lang) indicating continuous action must precede the verb 'ຮຽນ' (study)."
            },
            {
                "incorrect": "ລົດສອງນັ້ນຄັນ",
                "correct": "ລົດສອງຄັນນັ້ນ",
                "error_type": "Demonstrative Modifier Ordering",
                "explanation": "The canonical order is: Noun + Numeral + Classifier + Demonstrative ([Noun] + [Number] + [Classifier] + [ນັ້ນ])."
            },
            {
                "incorrect": "ຂ້ອຍຢາກໄດ້ການຮັກ",
                "correct": "ຂ້ອຍຢາກໄດ້ຄວາມຮັກ",
                "error_type": "Incorrect Nominalization Prefix",
                "explanation": "'ຮັກ' (love) is an emotional state/feeling, so it takes the prefix 'ຄວາມ-' to form 'ຄວາມຮັກ' (love), not 'ການ-' which is only for physical action verbs."
            }
        ]

        for idx, item in enumerate(scenarios[:count], start=1):
            instruction = (
                f"Identify and correct the grammatical error in the following Lao sentence:\n"
                f"Sentence: \"{item['incorrect']}\""
            )
            output = (
                f"**Corrected Sentence:** {item['correct']}\n\n"
                f"**Error Identified:** {item['error_type']}\n\n"
                f"**Explanation:**\n{item['explanation']}"
            )
            tasks.append({
                "id": f"synth_gec_{idx:03d}",
                "category": "grammar_error_correction",
                "instruction": instruction,
                "input": item["incorrect"],
                "output": output,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ]
            })
        return tasks

    def generate_tone_and_phonology_tasks(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate consonant class and tone determination drills."""
        tasks = []
        consonant_challenges = [
            ("ກ", "Middle (ອັກສອນກາງ)", "Unmarked live syllable produces Low tone."),
            ("ຂ", "High (ອັກສອນສູງ)", "Unmarked live syllable produces Rising tone."),
            ("ຄ", "Low (ອັກສອນຕ່ຳ)", "Unmarked live syllable produces High/Mid tone."),
            ("ຈ", "Middle (ອັກສອນກາງ)", "Unmarked live syllable produces Low tone."),
            ("ສ", "High (ອັກສອນສູງ)", "Unmarked live syllable produces Rising tone."),
            ("ທ", "Low (ອັກສອນຕ່ຳ)", "Unmarked live syllable produces High/Mid tone."),
            ("ດ", "Middle (ອັກສອນກາງ)", "Unmarked live syllable produces Low tone."),
            ("ຫ", "High (ອັກສອນສູງ)", "Unmarked live syllable produces Rising tone."),
            ("ນ", "Low (ອັກສອນຕ່ຳ)", "Unmarked live syllable produces High/Mid tone."),
            ("ບ", "Middle (ອັກສອນກາງ)", "Unmarked live syllable produces Low tone."),
        ]

        for idx, (cons, cons_class, tone_rule) in enumerate(consonant_challenges[:count], start=1):
            instruction = (
                f"Which consonant class does the Lao letter '{cons}' belong to, and what base tone does it yield in an unmarked live syllable?"
            )
            output = (
                f"**Consonant:** '{cons}'\n"
                f"**Class:** **{cons_class}**\n\n"
                f"**Tone Rule:**\n{tone_rule}"
            )
            tasks.append({
                "id": f"synth_tone_{idx:03d}",
                "category": "phonology_tone_drill",
                "instruction": instruction,
                "input": "",
                "output": output,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ]
            })
        return tasks
