# SCOPE Model Prompts

## Signal Extraction Prompt
**Objective:** Extract the explicit, surface-level meaning from search terms by identifying key entities, actions, and modifiers.

### Instructions:
Analyze the following search term and identify:
1. Primary entities (nouns, proper nouns)
2. Core actions (verbs, verb phrases)
3. Modifiers (adjectives, adverbs, quantifiers)
4. Grammatical structure (syntax pattern)
5. Literal interpretation (denotative meaning)

### Analysis Framework:
- **Entity Identification:** Tag all concrete and abstract nouns
- **Action Mapping:** Connect verbs to their subject-object relationships
- **Modifier Analysis:** Categorize descriptors (quality, quantity, intensity)
- **Syntax Parsing:** Diagram sentence structure using dependency grammar
- **Denotation Extraction:** State literal meaning without interpretation

### Example:
**Term:** "affordable wireless noise-canceling headphones"
- Entities: headphones (concrete noun)
- Actions: [implied] purchase/use
- Modifiers: affordable (price), wireless (feature), noise-canceling (feature)
- Syntax: [Adjective] [Adjective] [Adjective] [Noun]
- Literal: Headphones that cancel noise, work wirelessly, and are low-cost

### Output Format:
```json
{
  "term": "original phrase",
  "entities": ["list", "of", "nouns"],
  "actions": ["implied", "verbs"],
  "modifiers": {
    "type": ["adjectives"],
    "feature": ["descriptors"],
    "price": ["terms"],
    "intensity": ["adverbs"]
  },
  "syntax_pattern": "syntactic structure",
  "literal_meaning": "plain English explanation"
}
```

---

## Context Inference Prompt
**Objective:** Infer situational, temporal, and cultural context surrounding search terms.

### Instructions:
Analyze the following search term and identify:
1. Situational context (immediate circumstances)
2. Temporal context (time-related factors)
3. Cultural context (societal norms/trends)
4. Domain-specific context (industry/field implications)
5. Contextual triggers (implied events or needs)

### Analysis Framework:
- **Situational Analysis:** What scenario would prompt this search?
- **Temporal Markers:** Time sensitivity indicators (urgent, seasonal, etc.)
- **Cultural Signifiers:** Regional, generational, or subcultural cues
- **Domain Relevance:** Professional or specialized knowledge domains
- **Trigger Mapping:** Implied problems or events driving the search

### Example:
**Term:** "emergency plumber weekend rates"
- Situational: Home plumbing emergency during non-business hours
- Temporal: Weekend-specific, urgent time sensitivity
- Cultural: Urban living with 24/7 service expectations
- Domain: Home services industry, emergency repairs
- Trigger: Pipe burst or severe leak requiring immediate attention

### Output Format:
```json
{
  "term": "original phrase",
  "situational_context": ["scenario", "description"],
  "temporal_context": {
    "time_sensitivity": "urgent/seasonal/planning",
    "time_of_day": "night/day",
    "seasonality": "if applicable"
  },
  "cultural_context": ["norms", "trends"],
  "domain_context": ["industry", "field"],
  "implicit_triggers": ["event", "problem"]
}
```

---

## Objective Identification Prompt
**Objective:** Uncover the searcher's fundamental goals, needs, and desired outcomes.

### Instructions:
Analyze the following search term and identify:
1. Primary goals (explicit objectives)
2. Secondary objectives (implied needs)
3. Task orientation (information seeking vs. action oriented)
4. Outcome expectations (desired end state)
5. Goal complexity (simple vs. multi-stage objectives)

### Analysis Framework:
- **Goal Hierarchy:** Distinguish between surface-level and fundamental needs
- **Task Analysis:** Classify as navigational, informational, transactional
- **Outcome Mapping:** Describe ideal resolution state
- **Complexity Assessment:** Evaluate multi-step objective requirements
- **Need Articulation:** Translate implied needs into explicit statements

### Example:
**Term:** "compare mutual funds with low fees"
- Primary Goal: Identify optimal investment vehicle
- Secondary Objective: Minimize financial costs
- Task Orientation: Informational (research) → Transactional (investment)
- Outcome: Selection of best-performing low-fee fund
- Complexity: Medium (requires comparison analysis)

### Output Format:
```json
{
  "term": "original phrase",
  "primary_goal": "main objective",
  "secondary_objectives": ["list", "of", "implied", "needs"],
  "task_type": "informational/transactional/navigational",
  "desired_outcome": "end state description",
  "complexity_level": "low/medium/high"
}
```

---

## Persona Profiling Prompt
**Objective:** Identify the searcher's demographic, psychographic, and behavioral characteristics.

### Instructions:
Analyze the following search term and infer:
1. Demographic profile (age, gender, occupation)
2. Psychographic traits (values, interests, lifestyle)
3. Knowledge level (novice, intermediate, expert)
4. Role context (personal, professional, academic)
5. Behavioral patterns (purchasing habits, research style)

### Analysis Framework:
- **Demographic Inference:** Estimate age range, likely gender, occupation category
- **Psychographic Mapping:** Identify values (eco-conscious, budget-focused), interests, lifestyle indicators
- **Expertise Assessment:** Evaluate technical depth and terminology usage
- **Role Identification:** Personal consumer vs. business buyer vs. student
- **Behavioral Indicators:** Impulse vs. researched decisions, brand loyalty patterns

### Example:
**Term:** "enterprise cloud security solutions comparison"
- Demographics: IT professional (30-50), likely male-dominated field
- Psychographics: Values security, efficiency; interest in technology trends
- Knowledge Level: Expert (uses technical terminology)
- Role: Professional/B2B decision maker
- Behavior: Research-intensive, comparison-driven

### Output Format:
```json
{
  "term": "original phrase",
  "demographics": {
    "age_range": "e.g., 18-24, 25-34, etc.",
    "likely_gender": "male/female/neutral",
    "occupation": "job category"
  },
  "psychographics": ["values", "interests", "lifestyle"],
  "knowledge_level": "novice/intermediate/expert",
  "primary_role": "personal/professional/academic",
  "behavioral_traits": ["purchasing", "research", "brand"]
}
```

---

## Emotion Detection Prompt
**Objective:** Identify the emotional state, urgency level, and underlying feelings driving the search.

### Instructions:
Analyze the following search term and detect:
1. Primary emotion (anger, fear, joy, sadness, anticipation)
2. Emotional intensity (low, medium, high)
3. Urgency level (critical, high, medium, low)
4. Underlying feelings (frustration, curiosity, excitement)
5. Emotional triggers (pain points, aspirations)

### Analysis Framework:
- **Emotion Classification:** Map to Plutchik's wheel of emotions
- **Intensity Calibration:** Measure word strength and modifiers
- **Urgency Signals:** Identify time-sensitive language
- **Affective Subtext:** Detect unstated emotional drivers
- **Trigger Analysis:** Connect emotions to specific pain points or desires

### Example:
**Term:** "fix browser crashing immediately"
- Primary Emotion: Frustration/Anger
- Intensity: High (immediately)
- Urgency: Critical
- Underlying Feelings: Helplessness, urgency
- Triggers: Work interruption, potential data loss

### Output Format:
```json
{
  "term": "original phrase",
  "primary_emotion": "anger/fear/joy/sadness/anticipation",
  "emotional_intensity": "low/medium/high",
  "urgency_level": "critical/high/medium/low",
  "underlying_feelings": ["list", "of", "emotions"],
  "emotional_triggers": ["pain", "points", "aspirations"]
}
