#!/usr/bin/env python3
"""Compatibility entry point for the paired SpeechOcean762 evaluation."""

from evaluate_ctc_viterbi_vs_mfa import parse_args, run


if __name__ == "__main__":
    run(parse_args())
