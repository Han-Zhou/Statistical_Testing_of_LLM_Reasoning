import re

from domain.data import AnswerSpan
from models.adapters.base import ModelAdapter



def _locate_answer_span(self, output_text: str, search_start: int = 0) -> AnswerSpan | None:
        # search_start scopes the search past the prompt — the system prompt
        # itself contains the literal "\boxed{your answer}", which would
        # otherwise be matched as the model's final answer.
        
        # m = re.compile(r'\\boxed\{').search(output_text, search_start)
        # if m is None:
        #     return None

        m = output_text.rfind('\\boxed{', search_start)
        if m == -1:
            return None
        
        content_start = m + len('\\boxed{')  # char after the opening '\boxed{'
        # content_start = m.end()          # char after the opening '{'
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
            output_text.rfind('. ', search_start, m),
            output_text.rfind('\n', search_start, m),
            # output_text.rfind('. ', search_start, m.start()),
            # output_text.rfind('\n', search_start, m.start()),
        ) + 1
        if sentence_start < search_start:
            sentence_start = search_start
        while sentence_start < m and output_text[sentence_start] == ' ':
        # while sentence_start < m.start() and output_text[sentence_start] == ' ':
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

        # Exclusive end-of-text bound: char_idx points one char past the last
        # real character (e.g. an answer that runs to the very end of the
        # generation). Return one-past-the-last-content-token so the result
        # still works as a Python slice end. We can't use len(offset_mappings)
        text_end = max((end for _, end in offset_mappings), default=0)
        if char_idx == text_end:
            last_content = max(i for i, (_, e) in enumerate(offset_mappings) if e == text_end)
            return last_content + 1

        raise ValueError(f"Character index {char_idx} not found in any token span") 




