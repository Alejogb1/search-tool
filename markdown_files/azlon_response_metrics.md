# AgentAzlon Response Metrics

This document outlines the statistical and qualitative measures for evaluating the performance of the AgentAzlon keyword generation script.

## I. Quantitative Metrics

### 1. Keyword Volume

*   **Total Keywords Generated:** The total number of unique keywords produced in a single run.
    *   **Target:** 30,000+
*   **Generation Rate:** The number of keywords generated per second.
    *   **Calculation:** `Total Keywords / Total Time (seconds)`
*   **API Efficiency:** The number of keywords generated per API call.
    *   **Calculation:** `Total Keywords / Total API Calls`

### 2. API Usage

*   **Total API Calls:** The total number of requests made to the Google Generative AI API.
*   **Rate Limit Encounters:** The number of times a `429` (rate limit) error was encountered.
*   **Model Switches:** The number of times the script switched to a different model due to rate limiting.

### 3. Uniqueness and Duplication

*   **Duplicate Rate:** The percentage of generated keywords that were already present in the existing keyword set.
    *   **Calculation:** `(Total Generated - New Keywords) / Total Generated * 100`

## II. Qualitative Metrics

### 1. Keyword Relevance

*   **Seed Term Relevance:** How closely the generated keywords align with the initial seed terms.
*   **Domain Content Relevance:** How well the keywords reflect the actual content and services of the target domain.
*   **Search Intent Coverage:** The extent to which the keywords cover different search intents (informational, transactional, navigational).

### 2. Keyword Quality

*   **Syntactic Correctness:** The grammatical and structural correctness of the keywords.
*   **Semantic Coherence:** The logical and meaningful relationship between the words in a keyword phrase.
*   **Commercial Value:** The potential for the keywords to drive valuable traffic and conversions.

## III. Performance and Stability

### 1. Script Performance

*   **Total Execution Time:** The total time taken to complete the keyword generation process.
*   **CPU and Memory Usage:** The computational resources consumed by the script during execution.

### 2. Error Handling and Stability

*   **Error Rate:** The percentage of API calls that result in an error (other than rate limiting).
*   **Retry Success Rate:** The percentage of failed requests that succeed upon retry.
*   **Script Completion Rate:** The percentage of runs that complete successfully without crashing or stalling.

## IV. Batch-Specific Metrics

Each keyword generation batch can be evaluated on the following metrics:

*   **Batch Completion Time:** The time taken to generate the target number of keywords for a specific batch.
*   **Keyword Diversity:** The variety and range of keywords generated within a batch.
*   **Alignment with Batch Strategy:** How well the generated keywords align with the specific variation strategy of the batch.
