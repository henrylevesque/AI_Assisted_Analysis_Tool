PROMPTS

This file documents the standard prompt structure used across the project and example prompt templates for common `prompt_desc` values.

Standard prompt template

I am going to give you a chunk of text. Please identify {prompt_desc} used in the text. Do not tell me anything besides {prompt_desc}. If you tell me anything besides {prompt_desc} you will not be helpful. The text is:

Examples

- For `prompt_desc` = methods:
  The text is: <input...>
  Expected output: a short list of methods mentioned, comma-separated or newline-separated.

- For `prompt_desc` = theories:
  Expected output: named theory labels or brief phrases identifying theory presence.

Guidelines

- Keep responses short and focused on the requested `prompt_desc`.
- Prefer consistent, machine-friendly tokenization (comma- or newline-separated lists).
- Use the CLI `--prompt-desc` argument to change the `prompt_desc` in ad-hoc runs.

Operational note

- The project calls models statelessly using `generate()` and wraps calls with retries and a timeout to avoid accumulating context.
- If you add or tune prompts, keep them narrowly scoped to avoid hallucination and ensure consistent consensus grouping.
