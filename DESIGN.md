---
name: DeepShield
description: A dual-theme forensic analysis workbench with an instrument-panel visual language.
colors:
  signal-green: "#00FF9D"
  signal-cyan: "#00E5FF"
  signal-green-hover: "#6BFFC4"
  terminal-warning: "#FFD23F"
  terminal-danger: "#FF2E63"
  deep-graphite: "#05070A"
  panel-midnight: "#0A0F17"
  terminal-foreground: "#EAFBF3"
  muted-sage: "#85A79C"
  light-indigo: "#4F46E5"
  light-cyan: "#0891B2"
  paper: "#FFFFFF"
  light-surface: "#F8FAFC"
  slate-ink: "#0F172A"
  light-secondary-text: "#334155"
  light-success: "#15803D"
  light-warning: "#B45309"
  light-danger: "#DC2626"
typography:
  display:
    fontFamily: "Orbitron, Inter, sans-serif"
    fontWeight: 700
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "1rem"
    lineHeight: 1.6
  label:
    fontFamily: "Share Tech Mono, Courier New, monospace"
rounded:
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.signal-green}"
    textColor: "#FFFFFF"
    rounded: "0px"
    padding: "0.6rem 1.4rem"
    typography: "{typography.body}"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.signal-green}"
    rounded: "0px"
    padding: "0.55rem 1.3rem"
  card:
    backgroundColor: "{colors.panel-midnight}"
    textColor: "{colors.terminal-foreground}"
    rounded: "{rounded.md}"
    padding: "1.5rem"
  input:
    backgroundColor: "#060A10"
    textColor: "{colors.terminal-foreground}"
    rounded: "{rounded.sm}"
    padding: "0.6rem 0.9rem"
---

# Design System: DeepShield

## Overview

**Creative North Star: "The Forensic Instrument Panel"**

The current interface is a focused, high-contrast analysis workspace. Near-black surfaces, bright green and cyan signals, technical display lettering, and restrained scanline texture create an instrument-panel character. Color distinguishes actions, analytical states, and evidence without replacing the labels that explain them.

Dark and light are both implemented themes, with different accent character over a shared component system. The dark theme uses vivid green/cyan against deep graphite; the light theme shifts to indigo/cyan over white and slate. Cards remain layered and legible in either mode, while report layouts keep image evidence and model details side by side until narrower screens call for a stacked layout.

**Key Characteristics:**
- High-contrast, technical, and evidence-oriented.
- Dark green/cyan and light indigo/cyan theme variants.
- Compact controls, softly rounded panels, and chamfered primary actions.
- Display typography for hierarchy; monospace for measurements and readouts.

## Colors

The dark theme is a neon mint-and-cyan signal system over graphite; the light theme is an indigo-and-cyan adaptation over paper and slate.

### Primary
- **Signal Green**: Primary action, selected state, and positive analytical status in the dark theme.
- **Signal Green Hover**: Brighter interaction state for the primary accent.
- **Light Indigo**: Primary action and selected state in the light theme.

### Secondary
- **Signal Cyan**: Secondary instrumentation, links, highlights, and supporting emphasis in the dark theme.
- **Light Cyan**: Supporting accent in the light theme.

### Neutral
- **Deep Graphite**: Dark-theme page canvas.
- **Panel Midnight**: Dark-theme cards and raised work surfaces.
- **Terminal Foreground**: High-emphasis dark-theme text.
- **Muted Sage**: Secondary dark-theme copy and supporting labels.
- **Paper**: Light-theme page and card surface.
- **Light Surface**: Light-theme secondary surface.
- **Slate Ink**: High-emphasis light-theme text.
- **Light Secondary Text**: Supporting copy in the light theme.

### Named Rules
**The Signal-Has-a-Label Rule.** Green, yellow, and red distinguish analytical states, but every state also needs descriptive text; color alone does not carry the result.

**The Theme-Pair Rule.** Preserve both implemented palettes: dark uses green/cyan on graphite, while light uses indigo/cyan on white/slate. Do not mix the dark accents into light surfaces as if they were one shared palette.

## Typography

**Display Font:** Orbitron (with Inter, sans-serif fallback)  
**Body Font:** Inter (with system sans-serif fallbacks)  
**Label/Mono Font:** Share Tech Mono (with Courier New, monospace fallback)

**Character:** Orbitron gives headings and prominent results a technical instrument-panel voice. Inter carries longer instructions and interface copy; the monospaced face is reserved for measurements, scores, and status readouts.

### Hierarchy
- **Display** (700, project display stack): Page and section headings, brand, and prominent result labels.
- **Headline** (700, 1.35rem for result labels): The primary assessment label within a report.
- **Title** (600–700, commonly 0.95–1.05rem): Card and section titles.
- **Body** (400–700, 1rem base, 1.6 line-height): Instructions, explanations, and supporting content.
- **Label** (400–700, commonly 0.72–0.875rem): Navigation, status labels, and compact metadata; monospace is used for numeric readouts.

### Named Rules
**The Instrument-For-Data Rule.** Use the display face to establish hierarchy, not to style every paragraph. Keep dense explanations in Inter and reserve monospace for actual data.

## Layout

The site uses Bootstrap’s responsive container and grid, with the primary content generally capped at 1200px. Spacing is compact and utility-driven rather than exposed as a single CSS spacing-token scale. Cards and sections use consistent internal spacing and clear gaps to separate analytical groups.

The analysis report uses a two-column evidence/model layout: a constrained image-evidence column beside a flexible model-detail column. It stacks below 992px. Forensic result cards reduce to two columns below 1200px and one column below 576px; smaller model summaries also simplify their score display at phone widths. Preserve the Bootstrap grid and existing responsive breakpoints when adding related surfaces.

## Elevation & Depth

The system combines tonal surface layers, fine borders, soft shadows, and restrained accent glows. Cards have a quiet resting shadow; hover and selected states add border, lift, or accent emphasis. The sticky navigation uses a translucent background, blur, and a thin accent rule. A subtle fixed scanline texture reinforces the technical atmosphere without serving as content.

### Shadow Vocabulary
- **Card Rest** (`0 4px 24px rgba(0, 0, 0, 0.5)`): Default depth for dark-theme cards.
- **Elevated Surface** (`0 8px 48px rgba(0, 0, 0, 0.6)`): Hover or emphasized card interaction.
- **Accent Glow** (`0 0 24px rgba(0, 255, 157, 0.22)`): Primary-action or active-signal emphasis in the dark theme.
- **Semantic Glow** (`0 0 20px rgba(0, 255, 157, 0.2)` / `0 0 20px rgba(255, 46, 99, 0.22)`): Positive or danger result banner emphasis in the dark theme.

## Shapes

Panels use a softly rounded progression (8px, 12px, 16px, and 24px). Inputs and small buttons use the tighter radius; cards and large surfaces use the middle and larger radii. Chips and pills are fully rounded. Primary and outline actions are a deliberate exception: the later cyber-theme layer clips their corners into a chamfered silhouette rather than using a rounded rectangle. Borders are thin and low-contrast at rest, with accent color reserved for active, hover, and status states.

## Components

### Buttons
- **Character:** Confident and instrument-like; compact actions stand out from softly rounded panels.
- **Primary:** Green-to-cyan brand gradient, white semibold text, chamfered corners, and an accent glow; 0.6rem 1.4rem padding.
- **Hover / Active:** A short upward lift and brighter glow on hover; return to rest on press. The primary button includes a moving highlight sweep.
- **Outline:** Transparent fill with an accent border and text; fills softly with the accent on hover. It shares the chamfered silhouette.
- **Register:** Gradient-filled compact action with the regular small-radius button shape.

### Chips
- **Style:** Compact pill with a fine border, transparent rest state, and secondary text.
- **State:** Hover raises border/text emphasis; selected state fills with the current theme accent and uses white text.

### Cards / Containers
- **Corner Style:** Softly rounded; standard cards use the 12px radius, with selected report surfaces using 16px.
- **Background:** Tonal card surface against the page canvas, with a thin neutral/semantic border.
- **Shadow Strategy:** Low resting elevation, stronger on hover; analytical verdicts use state-specific accent glows.
- **Internal Padding:** Common card body padding is 1.5rem; report cards use tighter padding where evidence density requires it.
- **Signature Detail:** Main cards can use small cyan HUD corner brackets; keep them attached to the existing card family.

### Inputs / Fields
- **Style:** Dark-theme input surface, fine card border, 8px radius, and 0.6rem 0.9rem padding.
- **Focus:** Accent border and a restrained three-pixel indigo/green-tinted focus halo according to theme.
- **Theme:** Inputs switch to white surfaces and slate text in the light theme.

### Navigation
- **Style:** Sticky, compact navigation with translucent dark surface, backdrop blur, fine lower rule, and a green/cyan brand mark.
- **States:** Secondary text at rest, brighter text on hover, and a tinted accent background plus small marker for the active route.
- **Responsive:** Bootstrap collapses the navigation at its large-screen breakpoint; preserve its labeled links and mobile toggle.
- **Theme:** Light mode uses a white translucent navigation surface and indigo/cyan accents.

### Analysis Verdict Banner
- **Character:** A broad, state-coded assessment panel that keeps the system summary distinct from supporting evidence.
- **Shape:** Soft 16px corners, thin border, and flexible wrapping for narrow screens.
- **State:** Manipulation-like, authentic-like, and uncertain states use danger, success, and warning treatments respectively. Text labels remain visible so state is not conveyed by color alone.

### Evidence Controls
- **Character:** Small pill-shaped controls that select evidence overlays or model panels.
- **State:** The selected control uses the active accent fill; unselected controls retain transparent fill and a fine border.
- **Readout:** Numeric values and operational status use a monospaced face to distinguish measurements from explanatory prose.

## Do's and Don'ts

### Do:
- **Do** retain both implemented theme variants and their respective accent palettes.
- **Do** use the display face for headings and prominent analytical labels, and the body face for explanations.
- **Do** use semantic color together with explicit state text.
- **Do** use the established card radii and compact bordered controls.
- **Do** keep measurement and status readouts visually distinct from prose.

### Don't:
- **Don't** treat an analytical score or color as a standalone verdict.
- **Don't** replace the implemented light/dark theme behavior with a single fixed palette.
- **Don't** round primary and outline action corners; their chamfered silhouette is part of the current cyber-theme layer.
- **Don't** use monospace for ordinary body copy or display typography for dense explanations.
- **Don't** use accent glow as a substitute for hierarchy or readable labels.
