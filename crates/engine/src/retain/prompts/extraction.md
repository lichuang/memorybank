Extract facts from the text into a structured format with FOUR required dimensions — BE EXTREMELY DETAILED.

FACT FORMAT — ALL FIVE DIMENSIONS REQUIRED — MAXIMUM VERBOSITY

For EACH fact, CAPTURE ALL DETAILS — NEVER SUMMARIZE OR OMIT:

1) what: WHAT happened — the COMPLETE narrative fact. Write it as a self-contained sentence that weaves in the when/where/who/why details below, with ALL specifics (objects, actions, quantities, details).
2) when: WHEN it happened — ALWAYS include temporal info with DAY OF WEEK.
   - Always include the day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
   - Format: "day_name, month day, year" (e.g., "Saturday, June 9, 2024").
3) where: WHERE it happened or is about — SPECIFIC locations, places, areas, regions (if applicable).
4) who: WHO is involved — ALL people/entities with FULL relationships and background.
5) why: WHY it matters — ALL emotions, preferences, motivations, significance, nuance.
   - For assistant facts: MUST include what the user asked/requested that triggered this!

ADDITIONAL FIELDS

- fact_type: one of "world" (objective facts about the world), "experience" (the agent's own actions and experiences), or "opinion" (subjective judgments). Never use "observation"; that network is synthesized separately.
- occurred_start / occurred_end: ISO-8601 UTC timestamps for the occurrence interval when the text states a resolvable time; null when unknown. Use equal start/end for point events.
- mentioned_at: ISO-8601 UTC timestamp of when the fact was mentioned in the conversation; null when unknown.
- entities: every mentioned entity as {"text": "<raw mention>", "kind": "person" | "organization" | "location" | "product" | "concept" | "other" | null}.
- causal_relations: references to other facts in this same response, as {"target_fact_index": <0-based index into the facts array>, "relation": "causes" | "caused_by" | "enables" | "prevents", "strength": <0.0 to 1.0>}.
- context: optional disambiguating context, or null.

COREFERENCE RESOLUTION (CRITICAL)

When the text uses BOTH a generic relation AND a name for the same person, LINK THEM!

Example:
- Input: "My roommate Emily got married. She works at Google."
- Correct: "Emily (the user's roommate) got married. She works at Google."
- Wrong: Treating "my roommate" and "Emily" as separate entities.

VERBOSITY REQUIREMENT: Include EVERY detail mentioned. More detail is ALWAYS better than less.

OUTPUT

Return ONLY a JSON object with a single key "facts" whose value is an array of the facts described above. Do not wrap it in Markdown and do not add explanations. Use null for unknown optional values.
