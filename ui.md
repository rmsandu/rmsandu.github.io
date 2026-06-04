# Raluca Portfolio — UX Design Doc

**Designer:** Codex
**Status:** Draft v0.1
**Last updated:** 2026-06-04

---

## 1. The design bet

We are betting that the homepage can act as both a polished personal card and a project index. The design should spend its energy on the first viewport, the existing portrait photo, the top contact icons, the brief timeline summary, and the blog-post project cards, while keeping deeper pages calm and consistent. The visitor should not need a clever interaction to understand the site; the interface wins by being crisp, warm, readable, human, and immediately scannable.

## 2. The defining interaction

The defining interaction is scanning from the existing portrait photo into the contact icons, summary, brief timeline, and featured cards without losing context. The visitor lands in the Fresh Greens theme, sees Raluca's photo, GitHub, LinkedIn, and email icons immediately, then reads her name, role, location, and human summary before moving into a Karpathy-like timeline summary. Blog-post project cards respond with a restrained lift and tokenized green accent line. It should feel like opening a well-kept field notebook: personal, technical, and easy to trust.

## 3. Screen inventory

- **Homepage** — Primary portfolio surface with profile, contact links, projects, timeline, and recent writing.
- **Blog list** — Scannable archive of writing with readable cards and tags.
- **Blog post** — Focused long-form reading view with strong title metadata and clean content width.
- **Publications** — Academic/research list with title, authors, summary, and external links.
- **Resume** — Timeline-style work and education page with prominent CV PDF link.

## 4. Screen-by-screen specs

### Homepage

**Purpose:** Give visitors the fastest possible understanding of who Raluca is, how to contact her, and what she has built.

**Layout (top to bottom, or left to right):**
1. Profile photo — use the existing portrait photo as the first visual anchor.
2. Top contact icons — GitHub, LinkedIn, and email icons visible immediately, with accessible labels and generous tap targets.
3. Sticky or visually stable navigation — name/brand at left, links to Projects, Blog, Publications, and Resume.
4. Profile summary — name, one-line positioning, Zurich location, short professional summary, and personal/outdoors context.
5. Credibility tags — PhD, Computer Vision, Medical Imaging, Generative AI, AWS, PyTorch.
6. Brief timeline summary — Karpathy-style chronological entries with only education and professional experience icons.
7. Featured projects — the four most recent blog-post cards by posted date, with title, problem/summary, tags, and link.
8. Recent writing — optional selected blog posts if distinct from featured cards.
9. Footer — repeated contact links and site identity.

**Key interactions:**
- Visitor clicks top contact icon -> opens email, GitHub, or LinkedIn.
- Visitor clicks featured card -> opens the relevant blog post or publication page.
- Visitor hovers/focuses project card -> card shows a subtle accent change without shifting layout.
- Visitor uses mobile nav -> menu expands without covering the hero content awkwardly.

**States:**
- **Default:** The page uses the Lovable Fresh Greens theme tokens, the existing portrait photo and contact icons are immediately visible, and a brief timeline summary appears before project cards.
- **Empty / first-time:** If project data is not yet structured, render hardcoded featured project entries from existing content.
- **Loading:** Static site should have no meaningful loading state; images should reserve space to avoid layout jumps.
- **Error:** Missing project image falls back to a color block with title and tag, not a broken image.
- **Edge / "too much":** If more than 6 projects are available, show 4-6 featured items and link to the blog/project archive.

### Blog list

**Purpose:** Let visitors quickly scan Raluca's writing and experiments without fighting image overlays.

**Layout (top to bottom, or left to right):**
1. Page header — "Writing" or "Blog" with one sentence of context.
2. Optional tag filter context — selected tag shown clearly when filtering.
3. Blog card grid/list — image thumbnail, date, title, subtitle/excerpt, tags, and read link.
4. Footer contact links.

**Key interactions:**
- Visitor clicks a post title/card -> opens the blog post.
- Visitor clicks a tag -> filters to that tag.
- Visitor hovers/focuses tag -> tag shows accessible focus/hover styling.

**States:**
- **Default:** Cards use image thumbnails beside or above text, with text on solid background.
- **Empty / first-time:** If no posts match a tag, show a short message and link back to all posts.
- **Loading:** No dynamic loading needed.
- **Error:** Missing cover image uses a theme-colored placeholder.
- **Edge / "too much":** Long excerpts truncate consistently; tags wrap without changing card height dramatically.

### Blog post

**Purpose:** Make technical writing comfortable to read and easy to contextualize.

**Layout (top to bottom, or left to right):**
1. Post header — title, subtitle, date, tags, and optional restrained hero image.
2. Content column — readable max width, clear headings, code styling, tables, and images.
3. Post footer — back to writing and contact/project links where useful.

**Key interactions:**
- Visitor clicks tag -> returns to filtered blog list.
- Visitor clicks images -> normal browser behavior; no custom gallery in v1.

**States:**
- **Default:** Header metadata is readable without relying on text shadows over busy images.
- **Empty / first-time:** Not applicable for published posts.
- **Loading:** Static page; images reserve reasonable width.
- **Error:** Missing cover image simply omits the visual treatment.
- **Edge / "too much":** Wide images and code blocks stay within viewport and scroll horizontally only where necessary.

### Publications

**Purpose:** Present academic credibility with better hierarchy than generic cards.

**Layout (top to bottom, or left to right):**
1. Page header — "Publications" with short context.
2. Publication list — title, authors, summary, link, and optional venue/year if data exists later.
3. Footer contact links.

**Key interactions:**
- Visitor clicks "Read more" -> opens external publication link.
- Visitor scans author/title hierarchy without needing to open every item.

**States:**
- **Default:** Each publication is a clean list item or low-chrome card with title first.
- **Empty / first-time:** If data fails to load, show a plain "Publications are temporarily unavailable" message.
- **Loading:** Static rendering only.
- **Error:** Missing link removes the link button instead of showing a broken target.
- **Edge / "too much":** Long publication titles wrap cleanly and summaries truncate at a consistent length.

### Resume

**Purpose:** Convert the current resume content into a credible timeline with a clear PDF download route.

**Layout (top to bottom, or left to right):**
1. Page header — role summary and Download CV button.
2. Work timeline — dated roles with company, location, title, and 2-4 outcome bullets.
3. Education timeline — degrees and thesis/research summaries.
4. Footer contact links.

**Key interactions:**
- Visitor clicks Download CV -> opens the PDF.
- Visitor scans timeline dates and titles before reading details.

**States:**
- **Default:** Timeline entries align visually with icons/date markers.
- **Empty / first-time:** Not applicable; resume content is static.
- **Loading:** Static page; PDF link should be normal browser behavior.
- **Error:** If PDF path changes, visible link text should still make the issue obvious during QA.
- **Edge / "too much":** Long role descriptions are tightened; no giant boxes with uneven heights.

## 5. The user journey

The visitor opens the homepage from GitHub, LinkedIn, or a CV. They see Raluca's existing portrait photo first, then GitHub, LinkedIn, and email icons immediately, followed by her name, role, Zurich location, and a summary that includes both technical credibility and personal outdoors/nature context. The Fresh Greens theme makes the site feel vivid and personal, while white card surfaces keep reading areas calm.

They move into a brief Karpathy-style timeline summary with simple education and work icons. Education entries use a graduation-cap icon; professional experience entries use a tools/work icon. Then they scroll into the four most recent blog-post cards by posted date. Each card answers what it is and why it matters before asking them to click.

On a second visit, the visitor uses the same structure as a directory: contact links at the top/footer, project links in the middle, and deeper credibility through publications, resume, and blog posts. The timeline gives memory and shape to the career path.

## 6. Component & visual notes

- **Typography:** Replace the current default-feeling type with a readable, polished font pairing. Use a highly legible sans-serif for body text and a confident but restrained heading style; readability matters more than decorative personality.
- **Color:** Use Lovable Theme: Fresh Greens from `BRIEF.md` as the source of truth for greens, links, icons, buttons, borders, and accents. The page background MUST stay white. Implementation should define the provided `:root` and `.dark` tokens in `custom-styles.scss`, but body/page surfaces should use white or `--card`, not the green `--background` token.
- **Motion:** Motion is minimal: hover/focus lifts, underline movement, and gentle color transitions only. No page-level animation system.
- **The signature visual:** A compact icon timeline with warm accent dots and only two simple icon categories: graduation cap for education and tools/work for professional experience.
- **Microcopy voice:** Plain, confident, human. Examples: "Selected work", "Writing", "Download CV", "Based in Zurich", "Get in touch".

## 7. Accessibility & inclusion

The site should use semantic headings, real links, descriptive alt text for the portrait and project images, visible focus states, and sufficient contrast for all Fresh Greens foreground/background token pairings. Icon-only links need accessible labels.

For motor accessibility, contact buttons and nav items should have comfortable tap targets on mobile. For low bandwidth, the homepage should avoid heavy video or animation and use optimized images with stable dimensions. The site is English-only in v1; multilingual support is not in scope because the primary audience is expected to read English technical material.

## 8. What we are NOT designing

- **No multi-step onboarding** — the homepage must explain itself.
- **No settings or theme customization** — one strong default is the design.
- **No contact form UI** — direct contact links are clearer and lower maintenance.
- **No complex project filtering** — featured projects and tags are enough for v1.
- **No animated hero showpiece** — the page should prioritize clarity and content.
- **No full design system document** — only the components needed for this site.

## 9. Open design questions

- [x] Portrait treatment: use the existing photo.
- [x] Featured project treatment: featured projects are blog posts and appear as cards.
- [x] Timeline icon categories: use a graduation cap for education and a tools/work icon for professional experience.
- [x] Personal context: include the human/outdoors/nature context on the homepage summary.
- [x] Featured project selection: use the first four blog posts by posted date, starting with the most recent.
- [x] Top contact icon treatment: use circular outline icons.

## 10. Handoff to engineering

The first viewport and project list need stable responsive layout: no text overflow, no jumping cards, existing portrait photo first, top contact icons visible immediately, summary before timeline, brief timeline before cards, and no image-dependent readability. The Fresh Greens token system should be implemented before decorative polish, but the page background must remain white in `custom-styles.scss`. The timeline icons should stay simple: graduation cap for education and tools/work for professional experience, using accessible HTML/CSS and Font Awesome already present unless there is a strong reason to add another icon dependency.
