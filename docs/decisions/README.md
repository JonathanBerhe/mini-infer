# Architecture decision records

Each file in this folder records one decision that affects more than one
module or sets a precedent for later code. An ADR gives the problem, the
decision, the reason, the rejected alternatives and the measured result. It
does not tell the story of the work.

## Template

```
# ADR-NNN: <Title>

Date: YYYY-MM-DD
Status: Proposed | Accepted | Superseded by ADR-XXX

## Context
What problem do we solve?

## Decision
What did we decide?

## Alternatives Considered
Which other options did we look at, and why did each lose?

## Consequences
What did we measure, and what does the decision cost?
```

## Length

Aim for 300 to 600 words. If an ADR goes past 800 words, move the
measurements, derivations and run logs to a report in `docs/benchmarks/` and
link the report from the ADR. Keep only the headline numbers in the ADR.

## Language

Write in plain English, in the spirit of ASD-STE100 Simplified Technical
English:

- Write sentences of 25 words or fewer, in simple tenses and the active voice.
- Use the modals can, will and must. Do not use should, would, may, might or
  could.
- Do not use contractions, semicolons, em dashes, bold lead-ins or emoji.
- Define a concept term at its first use. Do not define standard names such
  as CUDA, Triton or PyTorch.
- Use one word for one meaning in the whole ADR, for example "make sure that"
  for check, verify and confirm.
- Never change a code identifier, a file path, a command or a measured
  number to make a sentence read better.

## Edits to old ADRs

Code comments cite numbered points in ADRs, for example "ADR-023 rule 2" and
"ADR-027 point 4". When you edit an ADR, keep those numbers and the terms that
the comments use. Run `grep -rn 'ADR-NNN' src tests scripts docs` first to
find them.
