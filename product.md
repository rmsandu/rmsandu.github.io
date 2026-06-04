# Raluca Portfolio — Product Design Doc

**Author:** Codex
**Status:** Draft v0.1
**Last updated:** 2026-06-04
**One-liner:** A polished personal portfolio that helps visitors understand Raluca's AI/ML expertise, contact her quickly, and scan her strongest projects from the first page.

---

## 1. The user & the moment

- **Who:** A hiring manager, technical lead, collaborator, or potential client who has just opened Raluca's site from GitHub, LinkedIn, a CV, or a shared link.
- **When:** They have 30-90 seconds to decide whether Raluca's background matches an AI/ML, computer vision, medical imaging, generative AI, or applied research opportunity.
- **Why now:** The current site has strong material, but it looks generic and makes visitors work too hard to find the signal: who Raluca is, how to contact her, and what she has built.

## 2. The contract (I/O)

- **Input:** The visitor lands on the homepage or one of the top navigation pages.
- **Output:** A clear first-page profile using the existing portrait photo, top-of-page GitHub/LinkedIn/email icons, personal/professional summary, brief timeline summary, blog-post project cards, and routes into publications and resume content.
- **The loop:** Land on the homepage -> see the existing profile photo -> immediately see GitHub/LinkedIn/email icons -> understand Raluca's professional and human positioning -> scan the brief timeline -> open featured blog/project cards or deeper publication/resume content.

## 3. The magical moment

> "I can tell in half a minute what she does, what she has built, and how to contact her."

## 4. Scope: what we ARE building (v1)

- A redesigned homepage ordered as existing profile photo, GitHub/LinkedIn/email icons, summary, brief Karpathy-style timeline summary, then featured cards.
- A project-forward section that treats the four most recent blog posts as scannable project cards.
- A compact timeline inspired by karpathy.ai, using dates, concise career/research milestones, graduation-cap icons for education, and tools/work icons for professional experience.
- Refined blog and publication listing styles that are calmer, more readable, and less dependent on text over images.
- A cleaner resume page that reads like a timeline and keeps the CV PDF link prominent.
- A Lovable Fresh Greens visual theme applied through the existing SCSS/Bulma setup, using tokenized `hsl(var(--...))` colors from `BRIEF.md`, while keeping the page background white.
- A more readable, polished font system that improves both homepage scanning and long-form blog readability.

## 5. Scope: what we are NOT building

- **No full rewrite to a new framework** — the current Flask/Jinja/Frozen-Flask setup is small and sufficient.
- **No complex CMS or admin UI** — static content files are enough for this portfolio.
- **No new project data backend** — project entries can live in templates or lightweight data files.
- **No heavy animation system** — the design should feel alive through layout, color, rhythm, and hover states.
- **No dark generic AI aesthetic** — the brief explicitly wants green with warm accents, not default tech gradients.
- **No exhaustive case-study pages in v1** — homepage project summaries can link to existing blog/publication content first.
- **No contact form** — direct email, GitHub, LinkedIn, and Scholar links are faster and lower maintenance.

## 6. The signature detail

The signature detail is a compact "field notebook" timeline: small icons, dates, and sharply written milestone entries that connect Raluca's technical work with her human context. It should feel like a researcher's annotated path rather than a corporate resume. The Fresh Greens theme carries the visual personality, while the page background stays white and saturated green tokens are used for structure, emphasis, and actions.

## 7. Success: how we know it worked

- **Primary:** A first-time visitor can identify Raluca's role, location, human context, GitHub/LinkedIn/email icons, and at least three relevant project/blog cards without scrolling past the first major project section.
- **Secondary:** The homepage places GitHub, email, and LinkedIn icons at the top and links clearly into CV, publications, and blog content.
- **Secondary:** The site visually stops reading as default Bulma while staying fast and maintainable.
- **What we're NOT measuring:** Pageview volume, animation complexity, number of sections, or novelty of implementation.

## 8. Open questions

- [x] Featured project cards: use the four most recent blog posts by posted date.
- [ ] Should project entries link only to existing blog/publication pages, or should any get dedicated case-study pages later?
- [ ] Is the primary visitor a hiring manager, freelance client, research collaborator, or all three equally?

## 9. Handoff

- **For UX:** The hardest design question is making the first page feel personal and project-rich without becoming cluttered.
- **For Eng:** Add the Lovable Fresh Greens tokens in `custom-styles.scss` as the root color system, then map existing semantic classes to those tokens while keeping the Flask/Jinja/Bulma pipeline simple.
