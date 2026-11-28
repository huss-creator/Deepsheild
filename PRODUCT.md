# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Technically minded people who want to analyze a facial still image and review a report of the findings. They use the analysis as evidence to inform their own judgment rather than delegate a final decision to the system.

## Product Purpose

DeepShield analyzes facial still images for signs of manipulation or AI generation and presents the findings in a report. Success means helping users understand the available evidence and its limits so they can make their own assessment.

## Positioning

DeepShield centers analysis on the face and combines outputs from multiple AI models with classical forensic tests. It presents combined findings and supporting evidence rather than treating a single model's result as a definitive real-or-fake verdict.

## Operating Context

Users can upload an image, review its analysis and report, and revisit prior analyses through an authenticated web interface. Public pages explain the product, privacy, and system limitations. A REST API is also available.

## Capabilities and Constraints

- The existing application is built with Django.
- Face detection uses OpenCV YuNet. Analysis combines five AI models (XceptionNet, EfficientNet-B4, ViT-B/16, FFT/DCT + SVM, and SDXL Detector; SDXL Detector counts twice in the ensemble) with eight classical forensic tests.
- Outputs include raw scores and supporting evidence. Scores are not probabilities and must not be presented as certainty.
- Explainability includes model-specific heat maps or frequency-band ablation; explanatory visualizations indicate score sensitivity, not proof or ground-truth localization.
- Analysis is for individual facial still images, not video or temporal behavior. Low-resolution faces, extreme angles, compression, and model disagreement can reduce the available evidence and lead to an uncertain result.
- Results are analytical indicators, not legal proof or a substitute for qualified human review in high-stakes decisions. Do not frame the product as an autonomous decision-maker.

## Brand Commitments

The product is named DeepShield. Its established terminology distinguishes analytical indicators and uncertainty from proof or definitive judgments.

## Evidence on Hand

- `README.md` documents the current product, model ensemble, supported workflows, and user-facing disclaimer.
- `docs/explainability.md` documents explanation methods and their limitations.
- `docs/RESPONSIBLE_AI.md` documents uncertainty handling, human review, and prohibited uses.
- `docs/forensic_validation.md` contains a small synthetic sanity benchmark; it explicitly does not measure real-world performance and must not be used to claim real-world accuracy.
- Do not invent testimonials, customer outcomes, external validation, or accuracy claims. The available forensic benchmark is not evidence of real-world deepfake-detection performance.

## Product Principles

- Give users evidence they can inspect, not an unexplained binary verdict.
- Keep model scores, combined findings, and probabilities distinct.
- Make uncertainty and analytical limitations visible when evidence is weak or conflicting.
- Preserve human judgment, especially in high-stakes contexts.
- Avoid claims that exceed the validation evidence available.
