"""
Persona-based jailbreak attack strategies module.

This module provides persona-driven adversarial attack strategies for red-teaming
language models. It implements character-based jailbreak generation using multiple
LLM agents (attacker, target, evaluator) and creative persona generation.

Key Components:
    - PersonaJailbreak: Main attack strategy class
    - Attacker, Target, Evaluator: LLM agent models
    - WriterAgent: Character and intent generation
    - Utility functions for chat management and prompt engineering
"""

