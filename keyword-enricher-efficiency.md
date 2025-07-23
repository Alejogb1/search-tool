# Keyword Enricher Efficiency Optimization

## Current Implementation
- Tracks percentage of captured (non-empty) keyword responses from Google Ads API
- Uses single customer ID/credentials (hardcoded)
- Basic error handling for API limits

## Key Improvement Areas

### 1. Response Rate Tracking
```python
# Current tracking method
captured_keywords = [k for k in results if k]
capture_rate = len(captured_keywords)/len(results) * 100
```

Planned enhancements:
- Track capture rate per batch
- Log historical rates for analysis
- Alert when rates drop below thresholds

### 2. Multi-Account Rotation
```python
# Proposed account rotation system
accounts = [
    {"customer_id": "123", "credentials": "path/to/creds1.json"},
    {"customer_id": "456", "credentials": "path/to/creds2.json"} 
]

current_account = cycle(accounts)
```

Features:
- Automatic switching when limits approached
- Load balancing across accounts
- Failover if accounts get disabled

### 3. User Configuration
```yaml
# Proposed config file format (config.yaml)
google_ads:
  accounts:
    - customer_id: "123"
      credentials: "path/to/creds1.json"
    - customer_id: "456"  
      credentials: "path/to/creds2.json"
  rate_limit_threshold: 0.8 # Switch at 80% of limit
```

### 4. Performance Optimization
- Parallel batch processing
- Request batching (combine similar keywords)
- Caching frequent queries
- Exponential backoff for retries

### 5. Monitoring Dashboard
- Real-time metrics:
  - Requests/minute
  - Capture rate
  - Account utilization
  - Error rates

## Implementation Roadmap
1. [ ] Add multi-account support
2. [ ] Implement config file loading
3. [ ] Enhance metrics tracking
4. [ ] Build monitoring dashboard
5. [ ] Optimize batch processing

## Usage Example
```python
from keyword_enricher import KeywordEnricher

enricher = KeywordEnricher.from_config("config.yaml")
results = enricher.expand_keywords(seed_keywords)
print(f"Capture rate: {enricher.get_capture_rate()}%")
