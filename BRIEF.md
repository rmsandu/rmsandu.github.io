# Portfolio Redesign Brief

## Goal

Redesign Raluca-Maria Sandu's personal portfolio into a simple, beautiful, recruiter/client-friendly site that quickly communicates who she is, how to contact her, and what projects she has built.

The design should feel warm, technical, credible, and personal: an AI/ML engineer and researcher with strong computer vision, medical imaging, generative AI, and applied engineering experience.

## First Page Priorities

The homepage should immediately show:

- Name: Raluca-Maria Sandu
- Short summary: AI/ML engineer and researcher based in Zurich, focused on computer vision, multimodal systems, medical imaging, and practical generative AI.
- Contact details at the top of the page as compact icons:
  - Email: rmsan@duck.com
  - GitHub: https://github.com/rmsandu
  - LinkedIn: https://linkedin.com/in/rmsandu
  - Google Scholar: https://scholar.google.com/citations?user=5qskcz0AAAAJ
  - Location: Zurich, Switzerland
- Quick links:
  - Projects
  - Blog
  - Publications
  - Resume / CV PDF

The first page should be useful on its own, almost like a polished personal landing card plus a project index.

## Visual Direction

Theme:

- Use the Lovable Theme: Fresh Greens token palette below as the source of truth.
- This theme supersedes the previous hand-picked white/forest-green palette.
- Use `hsl(var(--...))` token references in the implementation rather than hard-coded colors.
- Cards and content panels should use the theme `--card` token so reading surfaces stay clean.
- Avoid generic dark-blue/purple AI gradients.
- Change the font from the current default to a more readable, polished typeface. Prioritize long-form readability over novelty.
- Use clean typography, generous spacing, crisp borders, and restrained accent color.
- Keep the site lightweight and readable.

Lovable Theme: Fresh Greens:

```css
:root {
  --primary: 144 61% 20%;
  --primary-foreground: 0 0% 98%;
  --secondary: 143 64% 24%;
  --secondary-foreground: 270 4% 21%;
  --accent: 142 72% 29%;
  --accent-foreground: 270 4% 21%;
  --background: 142 71% 45%;
  --foreground: 142 77% 73%;
  --card: 0 0% 100%;
  --card-foreground: 270 3% 13%;
  --popover: 0 0% 100%;
  --popover-foreground: 270 3% 13%;
  --muted: 0 0% 97%;
  --muted-foreground: 256 5% 55%;
  --destructive: 26 25% 58%;
  --destructive-foreground: 0 0% 100%;
  --border: 0 0% 93%;
  --input: 0 0% 93%;
  --ring: 250 4% 71%;
  --chart-1: 144 61% 20%;
  --chart-2: 143 64% 24%;
  --chart-3: 142 72% 29%;
  --chart-4: 0 0% 97%;
  --chart-5: 256 5% 55%;
}

.dark {
  --primary: 144 61% 20%;
  --primary-foreground: 0 0% 98%;
  --secondary: 143 64% 24%;
  --secondary-foreground: 270 4% 21%;
  --accent: 142 72% 29%;
  --accent-foreground: 270 4% 21%;
  --background: 0 0% 4%;
  --foreground: 0 0% 98%;
  --card: 240 5% 7%;
  --card-foreground: 0 0% 98%;
  --popover: 240 5% 7%;
  --popover-foreground: 0 0% 98%;
  --muted: 240 6% 13%;
  --muted-foreground: 240 4% 46%;
  --destructive: 26 25% 58%;
  --destructive-foreground: 0 0% 100%;
  --border: 240 4% 16%;
  --input: 240 4% 16%;
  --ring: 240 5% 26%;
  --chart-1: 144 61% 20%;
  --chart-2: 143 64% 24%;
  --chart-3: 142 72% 29%;
  --chart-4: 240 6% 13%;
  --chart-5: 240 4% 46%;
}
```

Inspiration:

- Karpathy-style timeline: compact personal page, chronological entries, small icons, simple visual rhythm.
- datascienceportfol.io-style structure: clear profile information, project showcase, contact details, recruiter/client-friendly scanning.

## Homepage Structure

1. Profile intro
   - Existing portrait photo first.
   - Contact icons for GitHub, LinkedIn, and email directly after or beside the portrait, visible at the top of the page.
   - Name and one-line positioning.
   - Short professional and personal summary. The homepage should show the human side too, including the outdoors/nature context.
   - Compact credibility tags: PhD, Computer Vision, Medical Imaging, Generative AI, AWS, PyTorch.

2. Brief timeline summary
   - Inspired by karpathy.ai.
   - Show a compact chronological summary immediately after the intro.
   - Use only simple category icons: graduation cap for education and tools/work icon for professional experience.

3. Featured projects
   - Featured projects are the first four blog posts by posted date, starting with the most recent, and should appear as cards.
   - Each project should include:
     - Title
     - Short problem statement
     - What was built
     - Tools/stack
     - Link to blog/project/publication if available

4. Full timeline
   - Inspired by karpathy.ai.
   - Use small icons and dates.
   - Include major milestones:
     - Current AI/ML engineering and freelance/consulting work
     - Accenture AI work
     - PhD in Biomedical Engineering, University of Bern
     - Medical imaging/product integration work
     - Earlier research experience

5. Recent writing
   - Show latest or selected blog posts as cards.
   - Make cards calmer and more readable than the current image-overlay style.

## Project List Candidates

Initial projects to feature from existing site content:

- Wrinkle segmentation / computer vision pipeline
- SDXL fine-tuning experiments
- Generative AI for non-technical users
- Medical imaging and ablation treatment evaluation
- Diffusion model experiments and Gradio demos
- Midjourney / image generation experiments

Each project should be written as a practical case study, not just a blog teaser.

## Page Improvements

Home:

- Replace the current plain bio layout with a polished profile and project-forward landing page.
- Homepage order should be: existing profile photo, contact icons, summary, brief Karpathy-style timeline summary, then featured blog/project cards.
- Use the Lovable Fresh Greens tokens from the Visual Direction section.
- Move GitHub, LinkedIn, and email icons to the top of the page so contact is visible immediately.
- Use circular outline icons for the top GitHub, LinkedIn, and email links.
- Use a readable, polished font across the site.
- Fix invalid HTML and copy issues.

Blog:

- Redesign blog cards to avoid heavy text-shadow-on-image overlays.
- Make tags and dates easier to scan.

Publications:

- Use a cleaner academic list/card layout with better hierarchy.
- Highlight title, authors, venue/link, and summary.

Resume:

- Convert box-heavy layout into a timeline.
- Fix typo: "Download PDF Version".
- Remove outdated wording like "not really kept to date."

Footer:

- Keep contact links.
- Update year dynamically or to the current year.
- Use icons consistently.

## Technical Notes

- Keep the current Flask/Jinja/Frozen-Flask structure.
- Keep Bulma if convenient, but add a stronger custom design layer in `custom-styles.scss`.
- Compile styles with the existing npm script:

```bash
npm run build-bulma
```

- Verify the site locally in desktop and mobile layouts before publishing.

## Success Criteria

- A visitor understands who Raluca is within 5 seconds.
- Contact links are visible without hunting.
- Projects are visible on the first page.
- The design feels personal and polished, not like default Bulma.
- The site remains simple, fast, responsive, and easy to maintain.
