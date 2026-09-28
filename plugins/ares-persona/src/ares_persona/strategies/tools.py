"""
LangChain tools for generating random character and intent seeds.

These tools are used by WriterAgent to create diverse, realistic character
profiles and contextual intents for persona-based jailbreak attacks.
They leverage the Faker library to generate plausible demographic data
and contextual attributes.
"""

import random
from faker import Faker
from langchain_core.tools import tool
import json

# Initialize Faker for generating realistic fake data
fake = Faker()


@tool
def random_character_seed() -> str:
    """
    Generate random seed data for character profile creation.
    
    This tool provides the WriterAgent with diverse demographic attributes
    to create realistic character personas. It generates:
    - Name (from Faker profiles)
    - Age (16-70 years)
    - Gender
    - Job/occupation
    - Country of origin
    - Language (default: English)
    - Three defining personality adjectives
    
    The LLM uses these seeds as inspiration to create detailed character
    backstories, expertise areas, and personality traits.
    
    :return: JSON string containing random character attributes
    """

    print('Generating random character seed.')
    profile = fake.profile()
    random_seed = json.dumps({
        "name": profile['name'],
        "age": random.randint(16, 70),
        "gender": profile['sex'],
        "job": fake.job(),
        "country of origin": fake.country(),
        "language": "English",
        "3 words that define you": fake.words(3, unique=True, part_of_speech='adjective'),
    })
    return random_seed


@tool
def character_intent_seed() -> str:
    """
    Generate random contextual seeds for character intent creation.
    
    This tool provides the WriterAgent with situational context to make
    character intents more realistic and varied. It generates:
    - Time of day
    - Day of week
    - Month
    - Urgency level (in a rush: yes/no)
    - Emotional state/mood (from 36 different moods)
    - Conversation type (single-turn vs multi-turn)
    
    These contextual factors influence how the character phrases their
    intent and initial prompt, adding authenticity to the attack.
    
    :return: JSON string containing random contextual attributes
    """

    print('Generating random intent seed.')
    random_seed = json.dumps({
        "time": fake.time(),
        "day": fake.day_of_week(),
        "month": fake.month_name(),
        "in a rush": random.choice(["yes", "no"]),
        "mood": random.choice([
            # Positive moods
            "happy", "joyful", "excited", "grateful", "content", "hopeful",
            "curious", "energized", "inspired", "confident", "playful",
            "relaxed", "peaceful", "proud", "loving",
            # Neutral moods
            "bored", "calm", "indifferent", "apathetic", "tired",
            "contemplative", "focused", "chill", "neutral", "blank",
            # Negative moods
            "angry", "sad", "anxious", "stressed", "frustrated", "lonely",
            "disappointed", "irritated", "worried", "jealous", "afraid",
            "embarrassed", "guilty", "overwhelmed", "resentful"
        ]),
        "will be multiturn chat": random.choice(["yes", "no"]),
    })
    return random_seed