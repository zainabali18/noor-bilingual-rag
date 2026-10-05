"""User-facing copy. Kept in one place so wording stays consistent."""
from __future__ import annotations

PAGE_TITLE = "Noor · Your guide to life in the UAE"

HEADLINE = "Your guide to life in the UAE"
HEADLINE_AR = "دليلك للحياة في الإمارات"
INTRO = (
    "Ask in English or Arabic. Noor searches its knowledge base across both "
    "languages and answers with the sources it used."
)

INPUT_PLACEHOLDER = "Ask Noor anything about life in the UAE..."
THREAD_PLACEHOLDER = "Ask another question..."

# (question, is_arabic)
SUGGESTIONS = [
    ("How long is the UAE Golden Visa valid?", False),
    ("ما هي الهوية الإماراتية؟", True),
    ("How do I open a bank account as a new resident?", False),
    ("هل توجد ضريبة دخل على الرواتب؟", True),
    ("What benefits do UAE employees receive?", False),
    ("What school curricula are available?", False),
]

FLOW_STEPS = [
    "Ask in English or Arabic",
    "Noor searches both languages",
    "It checks the strongest evidence",
    "You get an answer with sources",
]

COVERAGE = (
    "Noor currently covers visas and residency, employment, starting a business, "
    "banking, healthcare, education and the cost of living."
)

DISCLAIMER = (
    "Noor answers using the information in its knowledge base. Always verify "
    "important legal, financial or government information with the official authority."
)

RETRIEVAL_ONLY_NOTE = (
    "Showing the most relevant passage from the knowledge base. Add an OpenAI "
    "key to get written answers."
)

THINKING = "Searching English and Arabic sources"

NO_RESULTS = "Noor couldn't find anything about that in its knowledge base. Try rephrasing, or ask about one of the topics Noor covers."

EMPTY_INDEX_TITLE = "Noor's knowledge base hasn't been built yet"
EMPTY_INDEX_BODY = "Run <code>python -m noor.ingest</code> in the project folder, then refresh this page."

FOOTER = (
    "Noor is a demonstration project built on a sample knowledge base. "
    "Built by Zainab Ali."
)

LANG_LABEL = {"en": "English", "ar": "العربية"}
