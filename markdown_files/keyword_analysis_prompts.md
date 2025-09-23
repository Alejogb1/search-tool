# Advanced Keyword Analysis Prompts

## 1. Market Opportunity Identification
```
Perform comprehensive market analysis on these keywords: {keywords}
1. Cluster keywords into distinct market segments using HDBSCAN
2. Calculate TAM/SAM for each segment using search volume data
3. Identify underserved segments (high search volume, low competition)
4. Generate a SWOT analysis for top 3 opportunities
5. Output: CSV with segment metrics + PDF report with visualizations
```

## 2. Customer Journey Mapping
```
Map customer journey stages using keyword intent analysis:
{keywords}

1. Classify keywords into: Awareness, Consideration, Decision
2. Identify key questions/pain points at each stage
3. Analyze drop-off points between stages
4. Recommend content strategy for each funnel stage
5. Output: Journey visualization + Content calendar template
```

## 3. Competitive Intelligence Extraction
```
Analyze competitive landscape from keyword patterns:
{keywords}

1. Identify competitor mentions and market share estimates
2. Map keyword clusters to competitor strengths/weaknesses
3. Detect emerging competitors from long-tail keywords
4. Predict competitive moves based on search trend anomalies
5. Output: Competitor matrix + Threat assessment report
```

## 4. Product Innovation Insights
```
Generate product innovation opportunities:
{keywords}

1. Extract unmet needs through negative sentiment analysis
2. Identify feature requests using syntactic pattern recognition
3. Cluster complementary capabilities for potential integrations
4. Prioritize opportunities using KANO model analysis
5. Output: Product roadmap + Feature specification templates
```

## 5. Predictive Trend Forecasting
```
Forecast market trends with time-series keyword analysis:
{keywords} (with historical search volume)

1. Detect emerging trends using anomaly detection algorithms
2. Project growth trajectories with ARIMA modeling
3. Identify seasonal patterns and cyclical behaviors
4. Predict market saturation points
5. Output: Interactive forecasting dashboard + Risk report
```

## 6. Customer Segmentation Synthesis
```
Create multidimensional customer segments:
{keywords}

1. Psychographic profiling through semantic analysis
2. Behavioral clustering based on search patterns
3. Identify segment-specific pain points
4. Calculate LTV estimates per segment
5. Output: Segment personas + Acquisition strategy guide
```

## 7. Content Optimization Framework
```
Develop data-driven content strategy:
{keywords}

1. Gap analysis: Existing vs. demanded content
2. Topic cluster identification
3. Content gap scoring (demand/competition ratio)
4. Optimal content format recommendations
5. Output: Content matrix + Editorial calendar
```

## 8. Technical Implementation Blueprint
```
Generate technical solution design:
{keywords}

1. Extract technical requirements from keyword patterns
2. Map to architectural components
3. Identify integration points
4. Create implementation roadmap with milestones
5. Output: System architecture diagram + Sprint plan
```

## 9. Risk Assessment Protocol
```
Perform comprehensive risk analysis:
{keywords}

1. Identify regulatory keywords and compliance risks
2. Detect market volatility signals
3. Analyze competitor threat levels
4. Evaluate technology obsolescence risks
5. Output: Risk heatmap + Mitigation playbook
```

## 10. Automated Research Paper Generator
```
Create academic-quality research paper:
{keywords}

1. Develop thesis statement from keyword patterns
2. Structure: Abstract, Literature Review, Methodology
3. Generate statistical analysis with visualizations
4. Formulate conclusions and future research directions
5. Output: Formal research paper in LaTeX format
```

## Re-prompting Mechanism Integration
All prompts should implement:
```python
if not validate_completeness(response):
    generate_refinement_prompt(
        missing_components=identify_gaps(response),
        context=response
    )
