# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals


class WordWindowExtractor(object):
    _filler_tag = "<W>"

    def __init__(self, window_size=5):
        self._window_size = window_size

    def instantiate_sentence(self, sentence):
        """
        Takes a sentence and returns the word window
        :type sentence: thesis.parsers.Sentence
        :return: tuple of the word window
        """

        main_lemma_index = sentence.main_lemma_index
        main_word = sentence.get_word_by_index(main_lemma_index)
        left_window, right_window = sentence.get_word_windows(main_lemma_index, self._window_size)
        full_word_window = left_window + [main_word] + right_window

        word_window_tokens = [word.tokens for word in full_word_window]

        # Padding the window vector in case the predicate is located near the start or end of the sentence
        if main_word.idx - self._window_size < 0:  # Pad to left if the predicate is near to the start
            for _ in range(abs(main_word.idx - self._window_size)):
                word_window_tokens.insert(0, (self._filler_tag,) * len(main_word.tokens))

        if main_word.idx + self._window_size + 1 > len(sentence):
            # Pad to right if the predicate is near to the end
            for _ in range(main_word.idx + self._window_size + 1 - len(sentence)):
                word_window_tokens.append((self._filler_tag,) * len(main_word.tokens))

        return word_window_tokens
