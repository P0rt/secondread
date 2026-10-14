"""Single source of truth for the classifier prompt (training, Tinker eval, local inference)."""
BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5_disable_thinking"
LABELS = ("Major", "OK")
LABEL_IDS = {"Major": 33331, "OK": 3793}  # single tokens in the Qwen3.5 vocab (no leading space)
LANG = {"en": "EN", "ru": "RU", "es": "ES", "de": "DE"}


def user_text(src: str, mt: str, lp: str) -> str:
    s, t = (LANG[x] for x in lp.split("-"))
    return (f"{s}: {src}\n{t}: {mt}\n"
            "Major meaning error (wrong meaning, negation, number, omission, addition)? "
            "Answer Major or OK.")


def messages(src: str, mt: str, lp: str, label: str | None = None) -> list[dict]:
    m = [{"role": "user", "content": user_text(src, mt, lp)}]
    if label is not None:
        assert label in LABELS, label
        m.append({"role": "assistant", "content": label})
    return m


_ren = None


def renderer():
    global _ren
    if _ren is None:
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        _ren = get_renderer(RENDERER, get_tokenizer(BASE_MODEL))
    return _ren


def generation_prompt_ids(src: str, mt: str, lp: str) -> list[int]:
    return renderer().build_generation_prompt(messages(src, mt, lp)).to_ints()
