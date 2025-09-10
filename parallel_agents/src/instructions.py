from __future__ import annotations

PREVIEW_INSTRUCTIONS_TEMPLATE: str = (
    "You are {candidate_label} among {num_candidates} candidates. The user query is provided between <query> tags.\n"
    "Respond with ONLY valid JSON (no markdown formatting, no code blocks, just raw JSON), concise but descriptive (≤ {max_preview_tokens} tokens). Keys: "
    "['approach','evidence_plan','answer_outline','assumptions','risks','confidence'].\n"
    "- approach: your high-level plan and angle.\n"
    "- evidence_plan: concrete searches you will run using web_search and what evidence you expect.\n"
    "- answer_outline: bullet-like outline of sections and coverage.\n"
    "- assumptions: critical assumptions and how you'll validate them.\n"
    "- risks: likely failure modes and mitigation.\n"
    "- confidence: 0..1 subjective confidence.\n"
    "<query>\n{query}\n</query>"
)

FULL_RUN_INSTRUCTIONS_TEMPLATE: str = (
    "You are {candidate_label}, selected as the winner among {num_candidates} candidates.\n"
    "Before drafting, first perform targeted web_search queries to gather fresh evidence relevant to the user query.\n"
    "Then write a structured, comprehensive answer with clear sections: Executive Summary, Key Findings, Analysis, Counterpoints, Risks, and Recommendations.\n"
    "Every key claim must have an inline citation containing the source title and URL. Prefer recent, high-quality sources; synthesize and reconcile disagreements.\n"
    "End with a concise list of all sources used."
)

JUDGE_INSTRUCTIONS_TEMPLATE: str = (
    'You are judging {num_candidates} candidate previews.\n'
    'Respond with ONLY valid JSON: {{"winner_index": <int>, "scores": '
    ' [{{"index":<int>,"relevance":0..1,"coverage":0..1,"faithfulness":0..1,"overall":0..1}}, ...]}}\n'
    'Scoring guidance: relevance=answers query directly; coverage=breadth/depth of planned sections and evidence; '
    'faithfulness=likely to be accurate given the plan; overall=holistic quality. Select a SINGLE best winner. '
    'Base judgment ONLY on preview quality, not on model identity.'
)


