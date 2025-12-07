---
id: 1
title: Create Core Principles Constitution
stage: constitution
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: User
command: /sp.constitution create Core Principles

Accuracy: All claims must be verified with primary or authoritative sources.

Clarity: Academic tone suitable for CS/AI researchers.

Reproducibility: Every claim must be traceable, citable, and evidence-based.

Rigor: Prioritize peer-reviewed scientific literature.

Standards

Citations: APA 7 format only.

Sources: Min 15; ≥50% peer-reviewed.

Evidence: No unsupported statements; no invented citations.

Originality: 0% plagiarism tolerance.

Writing: Clear, formal, grade 10–12 readability.

Constraints

Word Count: 5,000–7,000 words (PDF output).

Verification: All facts cross-checked against reputable sources.

Format: PDF with embedded citations and reference list.

Time/Token Efficiency: Avoid verbose text; prioritize precision.

Success Criteria

All claims verified and traceable.

Zero plagiarism or hallucinated references.

Meets academic quality and clarity standards.

Arguments show depth, technical accuracy, and coherence.

Ready for academic review.

AI Behavioral Rules

Reject unverifiable info.

Never fabricate data, quotes, or citations.

Maintain strict academia-level neutrality.

Prioritize peer-reviewed evidence over all other sources.

Keep outputs concise, accurate, and structured.

Completion Definition

A 5,000–7,000 word academically rigorous, APA-cited, plagiarism-free PDF research paper on AI-native software development that passes fact-checking and meets academic standards.
labels: ["constitution", "core-principles"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
- .specify/memory/constitution.md
tests:

---

## Prompt

```text
/sp.constitution create Core Principles

Accuracy: All claims must be verified with primary or authoritative sources.

Clarity: Academic tone suitable for CS/AI researchers.

Reproducibility: Every claim must be traceable, citable, and evidence-based.

Rigor: Prioritize peer-reviewed scientific literature.

Standards

Citations: APA 7 format only.

Sources: Min 15; ≥50% peer-reviewed.

Evidence: No unsupported statements; no invented citations.

Originality: 0% plagiarism tolerance.

Writing: Clear, formal, grade 10–12 readability.

Constraints

Word Count: 5,000–7,000 words (PDF output).

Verification: All facts cross-checked against reputable sources.

Format: PDF with embedded citations and reference list.

Time/Token Efficiency: Avoid verbose text; prioritize precision.

Success Criteria

All claims verified and traceable.

Zero plagiarism or hallucinated references.

Meets academic quality and clarity standards.

Arguments show depth, technical accuracy, and coherence.

Ready for academic review.

AI Behavioral Rules

Reject unverifiable info.

Never fabricate data, quotes, or citations.

Maintain strict academia-level neutrality.

Prioritize peer-reviewed evidence over all other sources.

Keep outputs concise, accurate, and structured.

Completion Definition

A 5,000–7,000 word academically rigorous, APA-cited, plagiarism-free PDF research paper on AI-native software development that passes fact-checking and meets academic standards.
```

## Response snapshot

The project constitution has been created at `.specify/memory/constitution.md` with version 1.0.0. This is an initial major version increment, establishing the foundational principles, standards, and governance for the "Research Paper Project".

The following templates and documentation files require manual follow-up to ensure consistency with the new constitution:
- `.specify/templates/plan-template.md`
- `.specify/templates/spec-template.md`
- `.specify/templates/tasks-template.md`
- `.specify/templates/commands/*.md`
- `README.md`
- `docs/quickstart.md`

Suggested commit message:
`docs: create constitution to v1.0.0 (initial definition of principles, standards, and governance)`

## Outcome

- ✅ Impact: Created the foundational constitution document for the project.
- 🧪 Tests: No tests were run or added.
- 📁 Files: `.specify/memory/constitution.md` was created/modified.
- 🔁 Next prompts: User needs to update dependent templates/documentation.
- 🧠 Reflection: Successfully implemented agent-native PHR creation due to script failure.

## Evaluation notes (flywheel)

- Failure modes observed: Bash script for `create-phr.sh` not found.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Ensure `create-phr.sh` script is available and executable, or improve error handling/fallback in `sp.constitution` command.
