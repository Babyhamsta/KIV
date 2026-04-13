"""Evaluation helpers for long-context prompting and generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch



FILLER_PARAGRAPHS = [
    "The history of maritime navigation spans thousands of years, from the earliest Polynesian wayfinders who read ocean swells and star patterns to the invention of the magnetic compass in medieval China. European sailors of the Age of Exploration relied on astrolabes and cross-staffs to determine latitude, though longitude remained elusive until John Harrison built his marine chronometer in the eighteenth century. Modern vessels use satellite-based GPS systems that provide accuracy within a few meters, a dramatic improvement over the celestial methods that guided ships for millennia.",
    "Photosynthesis is the biochemical process by which green plants, algae, and certain bacteria convert light energy into chemical energy stored in glucose. The process occurs primarily in chloroplasts, where chlorophyll pigments absorb photons and drive electron transport chains that produce ATP and NADPH. These energy carriers then power the Calvin cycle, which fixes atmospheric carbon dioxide into three-carbon sugars. Without photosynthesis, the vast majority of life on Earth would cease to exist, as it forms the foundation of nearly every food chain.",
    "The development of the printing press by Johannes Gutenberg around 1440 transformed European society in profound ways. Before movable type, books were copied by hand in monastic scriptoria, making them rare and expensive. Gutenberg's innovation reduced the cost of book production by orders of magnitude, enabling wider literacy, the rapid spread of scientific ideas, and ultimately the Protestant Reformation. Historians often rank the printing press alongside the internet as one of the most consequential communication technologies ever created.",
    "Volcanic eruptions shape landscapes and influence global climate patterns in ways that are both destructive and creative. When magma reaches the surface, it can produce explosive eruptions that send ash columns into the stratosphere, or effusive flows of lava that build new land over time. The 1815 eruption of Mount Tambora in Indonesia ejected so much sulfur dioxide that the following year became known as the Year Without a Summer, causing crop failures across the Northern Hemisphere. On geological timescales, volcanic activity has built island chains, fertilized soils, and driven mass extinctions.",
    "Jazz music originated in the African-American communities of New Orleans in the late nineteenth and early twentieth centuries, blending elements of blues, ragtime, and brass band marches. Early pioneers like Louis Armstrong and Duke Ellington developed improvisational techniques that became hallmarks of the genre. The bebop revolution of the 1940s, led by Charlie Parker and Dizzy Gillespie, introduced complex harmonies and rapid tempos that pushed jazz toward art music. Today jazz continues to evolve, absorbing influences from rock, electronic music, and world traditions.",
    "The human immune system is a remarkably complex network of cells, tissues, and organs that work together to defend the body against pathogens. Innate immunity provides a rapid first line of defense through physical barriers like skin and mucous membranes, as well as specialized cells like macrophages and natural killer cells. Adaptive immunity develops more slowly but offers highly specific responses through T cells and B cells that can remember previous encounters with a pathogen. This immunological memory is the principle behind vaccination, which has eradicated smallpox and nearly eliminated polio.",
    "Architecture in ancient Rome combined engineering innovation with aesthetic ambition on a scale rarely matched in the ancient world. Roman builders perfected the use of concrete, arches, and vaults to create structures like the Colosseum, the Pantheon, and extensive aqueduct systems that supplied fresh water to cities across the empire. The Pantheon's unreinforced concrete dome, completed around 125 AD, remains the largest of its kind and has inspired architects for nearly two thousand years. Roman roads, built with multiple layers of gravel and stone, connected an empire stretching from Britain to Mesopotamia.",
    "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning approaches. Supervised learning trains models on labeled data to make predictions, as in image classification or spam detection. Unsupervised learning discovers hidden patterns in unlabeled data through clustering or dimensionality reduction. Reinforcement learning trains agents to make sequential decisions by maximizing cumulative rewards, and has achieved superhuman performance in games like Go and complex robotic control tasks.",
    "The water cycle describes the continuous movement of water through the Earth's atmosphere, surface, and subsurface. Evaporation from oceans and lakes lifts water vapor into the atmosphere, where it condenses into clouds and eventually falls as precipitation. Some of this water flows across the surface as runoff into rivers and lakes, while the rest infiltrates the soil to recharge groundwater aquifers. Plants return water to the atmosphere through transpiration, completing one of many interconnected loops in this planetary-scale circulation system.",
    "The Renaissance, meaning rebirth, was a cultural movement that began in fourteenth-century Italy and gradually spread across Europe over the following two centuries. It was characterized by renewed interest in classical Greek and Roman art, philosophy, and science, and produced towering figures like Leonardo da Vinci, Michelangelo, and Galileo Galilei. The movement was fueled by growing wealth from trade, the patronage of powerful families like the Medici, and the rediscovery of ancient texts preserved in Islamic libraries. The Renaissance laid the intellectual groundwork for the Scientific Revolution and the Enlightenment.",
]


@dataclass(frozen=True)
class NeedlePrompt:
    """Token-accurate prompt for long-context recall evaluation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    context_length: int
    filler_budget: int
    needle_start: int
    question_start: int
    actual_depth: float


def _encode_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("Tokenizer must return a dict containing input_ids.")
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], int):
        return list(ids)
    if len(ids) != 1:
        raise ValueError("Expected a single sequence when encoding text.")
    return list(ids[0])


def _repeat_to_length(token_ids: list[int], target_length: int) -> list[int]:
    if target_length <= 0:
        return []
    if not token_ids:
        raise ValueError("Cannot build filler from an empty token sequence.")
    repeats = (target_length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:target_length]


def build_needle_prompt(
    tokenizer,
    context_length: int,
    depth: float,
    needle: str,
    question: str,
    filler_text: str | None = None,
) -> NeedlePrompt:
    """
    Build a chat-templated needle-in-haystack prompt.

    Layout inside chat template:
      <bos><|turn>user
      I'm going to give you a document. Read it carefully.

      [prefix filler paragraphs]
      [needle sentence]
      [suffix filler paragraphs]

      [question]<turn|>
      <|turn>model

    Total token count targets `context_length`. Filler is natural English
    paragraphs cycled to fill the budget.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between 0.0 and 1.0.")

    # Build filler from natural paragraphs or custom text
    if filler_text is None:
        filler_text = "\n\n".join(FILLER_PARAGRAPHS)

    # Construct the chat message parts
    preamble = "I'm going to give you a document. Read it carefully.\n\n"
    needle_block = f"\n\n{needle.strip()}\n\n"
    question_block = f"\n\n{question.strip()}"

    # Build the full message template to measure overhead
    # We need: preamble + prefix_filler + needle + suffix_filler + question
    # All wrapped in chat template
    test_msg = preamble + question_block
    test_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    overhead_ids = _encode_text(tokenizer, test_chat)
    needle_ids = _encode_text(tokenizer, needle_block)

    filler_budget = context_length - len(overhead_ids) - len(needle_ids)
    if filler_budget < 0:
        raise ValueError(
            f"context_length ({context_length}) too small for overhead "
            f"({len(overhead_ids)}) + needle ({len(needle_ids)}). "
            f"Need at least {len(overhead_ids) + len(needle_ids)}."
        )

    filler_ids = _encode_text(tokenizer, filler_text)

    prefix_len = int(round(filler_budget * depth))
    prefix_len = min(max(prefix_len, 0), filler_budget)
    suffix_len = filler_budget - prefix_len

    prefix_ids = _repeat_to_length(filler_ids, prefix_len)
    suffix_ids = _repeat_to_length(filler_ids, suffix_len)

    # Decode filler back to text so we can wrap in chat template
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)

    full_content = preamble + prefix_text + needle_block + suffix_text + question_block
    full_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_encoded = tokenizer(full_chat, return_tensors="pt")
    actual_len = full_encoded["input_ids"].shape[1]

    # Locate needle position in tokens (approximate)
    pre_needle_text = preamble + prefix_text
    pre_needle_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": pre_needle_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    needle_start = len(_encode_text(tokenizer, pre_needle_chat))

    return NeedlePrompt(
        input_ids=full_encoded["input_ids"],
        attention_mask=full_encoded["attention_mask"],
        context_length=actual_len,
        filler_budget=filler_budget,
        needle_start=needle_start,
        question_start=actual_len - len(_encode_text(tokenizer, question_block)),
        actual_depth=(prefix_len / filler_budget) if filler_budget else 0.0,
    )
