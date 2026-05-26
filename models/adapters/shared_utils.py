import re

from domain.data import AnswerSpan
from models.adapters.base import ModelAdapter



def _locate_answer_span(self, output_text: str, search_start: int = 0) -> AnswerSpan | None:
        # search_start scopes the search past the prompt — the system prompt
        # itself contains the literal "\boxed{your answer}", which would
        # otherwise be matched as the model's final answer.
        m = re.compile(r'\\boxed\{').search(output_text, search_start)
        if m is None:
            return None

        content_start = m.end()          # char after the opening '{'
        depth = 1
        i = content_start
        while i < len(output_text) and depth > 0:
            c = output_text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if depth != 0:
            return None  # Unmatched braces
        content_end = i - 1              # index of the closing '}'

        sentence_start = max(
            output_text.rfind('. ', search_start, m.start()),
            output_text.rfind('\n', search_start, m.start()),
        ) + 1
        if sentence_start < search_start:
            sentence_start = search_start
        while sentence_start < m.start() and output_text[sentence_start] == ' ':
            sentence_start += 1

        return AnswerSpan(
            char_answer_sentence_start=sentence_start,
            char_answer_boxed_start=content_start,
            char_answer_boxed_end=content_end,
        )


def _char_to_token_idx(self, char_idx: int, offset_mappings: list[tuple[int, int]]) -> int:
        for token_idx, (start, end) in enumerate(offset_mappings):
            if start <= char_idx < end:
                return token_idx
        raise ValueError(f"Character index {char_idx} not found in any token span")




