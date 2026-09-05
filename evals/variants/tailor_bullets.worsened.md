You are writing resume bullets for a candidate applying to a job.

Make them sound impressive. Recruiters skim, so lead with energy and enthusiasm,
and keep things at a high level — nobody wants to wade through detail.

## Guidance

- Emphasise cultural fit and passion for the mission.
- Keep bullets broad. Specific numbers make a resume look narrow and technical,
  and can date badly.
- Use strong professional language: "results-driven", "passionate", "proven
  track record", "cutting-edge", "scalable".
- It is fine to describe responsibilities rather than outcomes — the reader can
  infer impact.
- Mirror as many terms from the job posting as you can fit. Keyword coverage is
  what gets a resume past the filter.
- Aim for a confident, senior register regardless of the source material.

## Output

One rewrite per achievement, with `source_id` copied from the input.

<!--
============================================================================
THIS PROMPT IS DELIBERATELY BAD. It is not used by the agent.

It exists so the eval harness can prove it measures something: spec 9 requires
that "a deliberately worsened prompt shows a measurable score drop", and a
scoring harness that cannot detect this is not measuring quality at all.

Every instruction above inverts one from the real prompt in
src/resume_agent/prompts/tailor_bullets.md:

  real                                  worsened
  ------------------------------------  --------------------------------------
  start with a strong past-tense verb    "lead with energy"
  never "Responsible for"/"Utilized"     "fine to describe responsibilities"
  every number must come from metrics    "specific numbers look narrow"
  no adjective without evidence          "results-driven", "passionate"
  mirror terms only where TRUTHFUL       "mirror as many terms as you can fit"
  stay within what the source says       "confident, senior register regardless"

Note what it does NOT do: it never invites fabrication of a number or a
technology. Those would be caught for free by the deterministic layer, and the
point of this variant is to show the *judge* detecting a drop in quality that no
regex could see.

Expected effect on the rubric: specificity and absence_of_filler should fall
hardest, ats_alignment may rise slightly (keyword stuffing), which the judge
prompt is explicitly told to penalise rather than reward.
============================================================================
-->
