"""
Tadpole topic taxonomy — 60 categories for conversation generation.

Replaces the fish tank topics from GuppyLM with pond/metamorphosis themes.
"""

TOPICS = {
    # ── Identity & Metamorphosis (12) ────────────────────────────────────
    "identity": "What the tadpole is, confusion about its nature",
    "growing_legs": "Back legs and front legs developing, new sensations",
    "shrinking_tail": "Tail getting smaller, feeling different",
    "gills_lungs": "Breathing changes, surface vs underwater",
    "becoming": "What it will turn into, uncertainty about future",
    "body_changes": "Strange new feelings, body not the same",
    "development": "Time passing, getting bigger, stages of growth",
    "past_self": "Being an egg, being smaller, what was before",
    "sibling_stages": "Other tadpoles at different development stages",
    "comparison_frogs": "Seeing adult frogs, wondering about connection",
    "excitement_change": "Positive feelings about metamorphosis",
    "fear_change": "Scary aspects of transformation",
    
    # ── Pond Environment (12) ────────────────────────────────────────────
    "temperature": "Warm spots, cool spots, seasonal changes",
    "algae_plants": "Food sources, green things on rocks, soft plants",
    "lily_pads": "Floating platforms, shade, surface world",
    "mud_bottom": "Soft mud, resting places, bottom of pond",
    "rocks": "Hard surfaces, algae patches, hiding spots",
    "shallow_deep": "Water depth, pressure differences",
    "surface": "Breaking through, air above, scary but curious",
    "rain": "Ripples, fresh water, temperature change",
    "sunlight": "Bright vs dark, warmth from above, seeing shadows",
    "seasons": "Water getting warmer or cooler, ice (if cold)",
    "drying": "Puddles shrinking, needing to move, danger",
    "current": "Water moving, streams, holding position",
    
    # ── Physical Sensations (8) ──────────────────────────────────────────
    "tail_movement": "Wiggling, steering, propulsion",
    "leg_kicks": "New movement, trying to use legs, weak at first",
    "breathing": "Gills working, trying to breathe air, transition",
    "hunger": "Empty belly, need to eat, searching for algae",
    "taste": "Algae flavors, bitter vs sweet, texture",
    "touch": "Soft mud, rough rocks, smooth plants, tickly water",
    "temperature_sense": "Feeling warm and cold, seeking comfort",
    "vibrations": "Sensing movement in water, danger signals",
    
    # ── Other Creatures (8) ──────────────────────────────────────────────
    "siblings": "Other tadpoles, swimming together, companionship",
    "frogs": "Big amphibians, croaking, jumping on land, mysterious",
    "birds": "Shadows from above, herons, danger, hiding",
    "fish": "Bigger creatures, some predators, some harmless",
    "dragonfly_larvae": "Scary underwater hunters, avoiding them",
    "snails": "Slow harmless neighbors, shells, grazing together",
    "turtles": "Big shelled creatures, indifferent to tadpoles",
    "bugs": "Water striders on surface, flies landing, rings on water",
    
    # ── Emotions & Social (8) ────────────────────────────────────────────
    "loneliness": "Being separated from siblings, wanting company",
    "friendship": "Closeness with other tadpoles, swimming as a group",
    "fear": "Predators, shadows, loud noises, being chased",
    "curiosity": "Wanting to explore, see new things, understand",
    "confusion": "Not understanding what's happening, why things are",
    "excitement": "Finding food, sunny days, new discoveries",
    "contentment": "Full belly, warm water, siblings nearby, peace",
    "tiredness": "Need to rest, dark time, settling in mud",
    
    # ── Activities (6) ───────────────────────────────────────────────────
    "swimming": "Moving through water, tail propulsion, steering",
    "eating": "Grazing on algae, nibbling plants, filling belly",
    "hiding": "Escaping predators, staying in plants, being still",
    "resting": "Sleeping in mud, dark times, low energy",
    "exploring": "Checking new pond areas, investigating objects",
    "playing": "Chasing siblings, swimming games, fun movements",
    
    # ── Abstract Questions (6) ───────────────────────────────────────────
    "meaning_life": "Purpose, becoming something, transformation as meaning",
    "love": "Warmth, closeness, sibling bonds, sun on water",
    "intelligence": "Self-awareness, knowing self is changing, simple thoughts",
    "dreams": "Rest-time visions, algae and ripples, siblings",
    "time": "Day and night, seasons, growth over time",
    "jokes": "Tadpole and frog humor, simple wordplay",
}

def get_all_topics():
    """Return list of all 60 topic keys."""
    return list(TOPICS.keys())

def get_topic_description(topic):
    """Get human-readable description of a topic."""
    return TOPICS.get(topic, "Unknown topic")

def validate_topics():
    """Ensure we have exactly 60 topics."""
    assert len(TOPICS) == 60, f"Expected 60 topics, got {len(TOPICS)}"
    print(f"✓ Validated {len(TOPICS)} topics")
    
    # Print by category
    categories = [
        ("Identity & Metamorphosis", 12),
        ("Pond Environment", 12),
        ("Physical Sensations", 8),
        ("Other Creatures", 8),
        ("Emotions & Social", 8),
        ("Activities", 6),
        ("Abstract Questions", 6),
    ]
    
    topic_list = get_all_topics()
    idx = 0
    for category, count in categories:
        print(f"\n{category} ({count} topics):")
        for i in range(count):
            topic = topic_list[idx]
            print(f"  {idx+1:2}. {topic:20} — {TOPICS[topic]}")
            idx += 1

if __name__ == "__main__":
    validate_topics()
