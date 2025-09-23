# Metabase Guide: Building an Interactive Semantic Keyword Dashboard

## Introduction

This guide provides a step-by-step walkthrough for building a powerful, interactive keyword analysis dashboard directly within Metabase. It assumes that the data processing (embedding, clustering, etc.) has already been completed, and you have a table in your PostgreSQL database named `keyword_semantic_analytics` with the following key columns:

*   `text`: The keyword itself.
*   `avg_monthly_searches`: The average monthly search volume.
*   `competition`: The competition level ('LOW', 'MEDIUM', 'HIGH').
*   `SemanticCluster`: The numerical ID of the semantic cluster the keyword belongs to (e.g., from HDBSCAN).
*   `umap_x`, `umap_y`: The 2D coordinates for visualizing the keyword in semantic space.

Our goal is to use Metabase's powerful features to transform this raw data into a strategic tool for exploring topics, identifying opportunities, and making data-driven decisions.

---

## Part 1: Creating Your Core Analytics "Questions"

In Metabase, every chart or table is a "Question." We will build several key questions that will later become the components of our dashboard.

### Question 1: The Semantic Keyword Map

This is the centerpiece of our dashboard—a visual map of your entire keyword landscape.

1.  **Start a New Question**: Click **+ New > Question**.
2.  **Select Your Data**: Choose **Raw Data** > Your Database > `keyword_semantic_analytics`.
3.  **Choose Visualization**: At the bottom, click **Visualize** and select the **Scatter plot** icon.
4.  **Configure the Axes**:
    *   Set the **X-axis** to `umap_x`.
    *   Set the **Y-axis** to `umap_y`.
5.  **Add Semantic Grouping**:
    *   In the **Group by** section, select `SemanticCluster`. This will automatically color the points on your scatter plot according to their cluster ID, creating a beautiful map of your topics.
6.  **Filter Out Noise**: The HDBSCAN algorithm labels unclustered "noise" points as `-1`. Let's hide these by default.
    *   Click the **Filter** button.
    *   Select `SemanticCluster` > **Is not** > and enter `-1`.
7.  **Save the Question**: Click **Save**, name it "Semantic Keyword Map", and save it to a collection.

### Question 2: The Cluster Performance Profile (using SQL)

We need a summary table that profiles each cluster's performance. While this can be done with the query builder, using the SQL editor gives us more power to create custom metrics like an "Opportunity Score".

1.  **Start a New SQL Question**: Click **+ New > SQL query**.
2.  **Select Your Database**.
3.  **Enter the SQL Query**: Copy and paste the following query. This query calculates key metrics for each cluster and creates a composite `OpportunityScore`.

    ```sql
    -- First, we map the text-based competition to a number
    WITH keyword_with_intent AS (
      SELECT
        *,
        CASE
          WHEN competition = 'HIGH' THEN 3
          WHEN competition = 'MEDIUM' THEN 2
          ELSE 1
        END AS "CommercialIntent"
      FROM
        keyword_semantic_analytics
      WHERE
        "SemanticCluster" != -1
    )
    -- Now, we group by cluster to create the profiles
    SELECT
      "SemanticCluster",
      COUNT(*) AS "KeywordCount",
      SUM(avg_monthly_searches) AS "TotalSearchVolume",
      AVG("CommercialIntent") AS "AvgCommercialIntent",
      -- This formula scores clusters based on volume and intent, penalized by size.
      (SUM(avg_monthly_searches) * AVG("CommercialIntent")) / SQRT(COUNT(*)) AS "OpportunityScore"
    FROM
      keyword_with_intent
    GROUP BY
      "SemanticCluster"
    ORDER BY
      "OpportunityScore" DESC;
    ```

4.  **Run and Visualize**: Click the blue "play" button to run the query. It will display as a table.
5.  **Save the Question**: Click **Save**, name it "Cluster Performance Profiles", and save it.

### Question 3: Top Opportunity Clusters Chart

Let's visualize the output of our SQL query to easily spot the best clusters.

1.  **Start a New Question**: Click **+ New > Question**.
2.  **Select Your Data**: This time, choose **Saved Questions** and select the "Cluster Performance Profiles" question you just created.
3.  **Choose Visualization**: Click **Visualize** and select the **Bar chart** icon.
4.  **Configure the Chart**:
    *   Set the **X-axis** to `SemanticCluster`.
    *   Set the **Y-axis** to `OpportunityScore`.
    *   Metabase will likely do this automatically. Ensure it's sorted by `OpportunityScore` descending.
5.  **Save the Question**: Click **Save** and name it "Top Opportunity Clusters".

---

## Part 2: Assembling the Interactive Dashboard

Now we'll combine our saved Questions into a single, interactive dashboard.

1.  **Create a New Dashboard**: Click **+ New > Dashboard**. Give it a name like "Keyword Intelligence Dashboard".

2.  **Add Your Questions**:
    *   Click the **+** icon in the center of the dashboard.
    *   Find and add your three saved questions: "Semantic Keyword Map", "Cluster Performance Profiles", and "Top Opportunity Clusters".
    *   Resize and arrange them. A good layout is the "Semantic Keyword Map" taking up the top half, with the "Top Opportunity Clusters" bar chart and the "Cluster Performance Profiles" table below it.

3.  **Add Dashboard Filters (The Interactive Part)**:
    *   Click the **pencil icon** to edit the dashboard, then click the **Add a Filter** button.
    *   **Add a Cluster Filter**:
        *   Choose **ID** as the filter type.
        *   In the filter's settings, under "Which column(s) should this filter?", select the `SemanticCluster` column for **all three** of your questions. This links the filter to every chart.
        *   Change the "Filter widget type" to **Dropdown list**.
    *   **Add a Commercial Intent Filter**:
        *   Click **Add a Filter** again.
        *   Choose **Number** > **Equal to**.
        *   Link this filter to the `CommercialIntent` column in your "Semantic Keyword Map" question. (You may need to add this column to the SQL query if it's not already there, or create a new Question from the base table).

4.  **Enable Cross-Filtering (Click Actions)**:
    *   This is the most powerful feature. We want to be able to click on the bar chart or table and have the map update.
    *   Click on the "Top Opportunity Clusters" bar chart on your dashboard to select it.
    *   In the side panel that appears, click **Add a custom destination**.
    *   Choose **To a dashboard** > **This dashboard**.
    *   A mapping will appear. Ensure that the `SemanticCluster` column from the bar chart is mapped to the `SemanticCluster` **dashboard filter** you created.
    *   Repeat this process for the `SemanticCluster` column in your "Cluster Performance Profiles" table.

5.  **Save and Use**: Click **Save**. Your dashboard is now ready. You can select a cluster from the dropdown filter, or click directly on a bar in the "Top Opportunity Clusters" chart, and the entire dashboard—especially the Semantic Map—will instantly update to show you data for only that topic.
