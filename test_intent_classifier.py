#!/usr/bin/env python3
"""
Quick test for the intent classifier — no robot needed.

Usage:
    python test_intent_classifier.py
"""

from intent_inference import IntentClassifier

classifier = IntentClassifier()

test_phrases = [
    # Direct color names
    "Pick up the box and place it in the red bowl",
    "Pick up the box and place it in the green bowl",
    "Pick up the box and place it in the blue bowl",
    # Indirect / semantic references
    "Pick up the box and place it in the bowl which has the color of the sky",
    "Pick up the box and place it in the bowl which has the color of water",
    "Pick up the box and place it in the bowl which has the color of the ocean",
    "Pick up the box and place it in the bowl which has the color of blood",
    "Pick up the box and place it in the bowl which has the color of grass",
    "Pick up the box and place it in the bowl which has the color of leaves",
    "Put the box in the crimson bowl",
    "Place the box in the emerald bowl",
    "Move the box to the azure bowl",
    # Short / casual
    "red bowl please",
    "go to blue",
    "the green one",
    # Tricky
    "place it in the bowl that looks like the sea",
    "put it in the cherry-colored bowl",
    "move it to the forest-colored container",
]

print(f"\n{'INSTRUCTION':<65} {'COLOR':<8} {'CONF':<8} CANONICAL TASK")
print("-" * 140)

for phrase in test_phrases:
    color, canonical, conf = classifier.classify(phrase)
    print(f"{phrase:<65} {color:<8} {conf:.3f}   {canonical}")
