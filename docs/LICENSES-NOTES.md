# License and terms notes

Checked on 2026-10-14 by reading the pages linked below. This is an engineering reading of
the terms, not legal advice. Quotes are kept short; follow the links for the full text.

## Gemma 4 outputs from DigitalOcean: may they train the Qwen LoRA?

**Yes.** None of the terms that apply to `gemma-4-31B-it` on DigitalOcean serverless inference
restrict using its outputs to train another model. `data/synth/llm.train.jsonl` may go into
SFT v2. `data/synth/llm.test.jsonl` stays a held-out test set, because of split hygiene, not
because of the license.

### 1. Google: Gemma 4 is Apache 2.0, not under the Gemma Terms of Use

- The Hugging Face card of [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)
  is tagged `apache-2.0`, is not gated, and its license link is
  [ai.google.dev/gemma/docs/gemma_4_license](https://ai.google.dev/gemma/docs/gemma_4_license).
  That URL redirects to [ai.google.dev/gemma/apache_2](https://ai.google.dev/gemma/apache_2),
  which holds the plain Apache License 2.0 text (page updated 2026-04-01).
- Google's [release post](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html)
  (2026-04-02) says Gemma 4 is "the first in the Gemmaverse to be released under the
  OSI-approved Apache 2.0 license".
- The [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (last modified 2026-04-01) say
  they "apply to Gemma models listed in the Appendix", and add "For Gemma 4 terms, see the
  Gemma 4 license." The Appendix lists Gemma 1 to 3, 3n, PaliGemma, CodeGemma, ShieldGemma,
  TranslateGemma and others. Gemma 4 is not in it.
- The [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) (last
  modified 2024-02-21) binds through Section 3.2 of the Gemma Terms of Use. It covers
  "Gemma or Model Derivatives" in the sense of those terms, so it is not a license condition
  for Gemma 4. Our use would be allowed under it anyway: we make synthetic translation errors
  in public WMT text to train and test an error detector. None of its categories (rights
  infringement, illegal or harmful activity, harassment, misinformation, sexual content,
  bypassing safety filters) applies. The policy says nothing about training other models.
- Apache 2.0 sets no use restrictions and says nothing about outputs. Its conditions
  (Section 4: a copy of the license, the NOTICE file, marks on changed files) apply only
  when you redistribute the Work, meaning Gemma's weights and files, or a Derivative Work
  of it. We do neither. The LoRA holds no Gemma weights. It is a delta on Qwen3.5-4B, which is
  also Apache 2.0.

### 2. Would the LoRA count as a Gemma "Model Derivative"?

**Under the Gemma Terms of Use it would, but those terms do not cover Gemma 4.** So for this
project the answer is no, and nothing passes through.

The definition in Section 1.1(e) has a third branch. It covers any other model trained on
patterns or Output of Gemma "in order to cause that model to perform similarly to Gemma".
That explicitly includes "the generation of synthetic data Outputs by Gemma for training that
model". You could argue that a Major/OK classifier does not "perform similarly" to a chat
model. But the synthetic-data wording is broad, so plan as if the LoRA would be a Model
Derivative whenever those terms apply. "Outputs are not deemed Model Derivatives", and
"Google claims no rights in Outputs you generate using Gemma" (Section 3.3).

If the generator ever becomes a model listed in the Appendix (for example Gemma 3 or
TranslateGemma), these obligations would apply to the LoRA and to the GGUF we publish:

- **Use restrictions (Section 3.2):** the Prohibited Use Policy binds us and everyone who uses
  the adapter.
- **Pass-through (Section 3.1):** our license must "include the use restrictions referenced
  in Section 3.2 as an enforceable provision", and we must tell recipients that the model is
  subject to them.
- **Copy of the terms:** every recipient gets a copy of the Gemma Terms of Use.
- **Changed files:** they must carry a prominent notice that we changed them.
- **NOTICE file:** every distribution outside a hosted service needs a `Notice` text file.
  Its required text begins "Gemma is provided under and subject to the Gemma Terms of Use"
  and ends with the terms URL (exact wording in Section 3.1).
- **The synthetic dataset itself:** it is made of Outputs, so it carries none of these
  obligations, but the model trained on it would.

### 3. DigitalOcean terms

- The [models page](https://docs.digitalocean.com/products/inference/details/models/) says
  serverless models are offered under the AI Model Terms (Section 4 of the
  [Service Terms](https://www.digitalocean.com/legal/tos-service-specific-terms), updated
  2026-10-08), the Service Terms and the
  [Terms of Service](https://www.digitalocean.com/legal/terms-of-service-agreement) (updated
  2026-08-22). It lists no extra terms for `gemma-4-31B-it`.
- **Section 4.2 of the Service Terms** treats third-party models as third-party services
  under their provider's terms. DO's
  [Third-Party AI Models](https://www.digitalocean.com/legal/third-party-ai-models) page
  (updated 2026-09-25) links "Gemma 4-31B-it" to
  [huggingface.co/google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B). The HF API
  reports `apache-2.0` and not gated for that model too.
- **Section 1.3.6:** "The output you generate is considered your Services Content." DO
  claims no rights in it.
- **Nothing on training.** The Service Terms, the TOS, the
  [Acceptable Use Policy](https://www.digitalocean.com/legal/acceptable-use-policy) (updated
  2026-03-20) and the Third-Party AI Models page contain no clause against using outputs to
  train, fine-tune or distil another model, and no clause against competing models. We
  searched them for "train", "compet", "distill", "output" and "derivative". TOS 5.4 forbids
  derivative works of DO's own Websites or Services. Outputs are our Services Content, not
  part of the Services.

What DigitalOcean does require:

- **AUP:** do not make deceptive content meant to misinform, and do not hide where synthetic
  content came from. Every synthetic row keeps `synthetic: true`, `synth_generator` and
  `synth_model`, and any released data or post must say the errors are Gemma-generated.
- **Service Terms 2.5:** do not remove provenance data (watermarks, metadata) from model
  output. We keep the raw replies in `data/synth/cache/llm_cache.jsonl`.
- **Service Terms 4.5.1:** where AI-generated content reaches end users, label it as
  AI-generated if the law requires it. The synthetic rows are training and test data and are
  not shown to end users.
- **Service Terms 4.5.2:** DO may scan inputs and outputs. We send public WMT text only, as
  the README already says.

### 4. What else still applies

- **Source texts:** the WMT22 and WMT24 texts inside every synthetic row are still under
  WMT's "research purposes" terms (see the README). The Gemma license does not change that.
- **Attribution:** name the generator, "Gemma 4 31B (`gemma-4-31B-it`, Apache 2.0) via
  DigitalOcean serverless inference", in the README and the post. Apache 2.0 does not require
  this for outputs; it is a courtesy.
- **Re-check this note** if the generator model, the DO terms or the Gemma 4 license page
  change.
