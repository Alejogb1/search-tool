
More in-depth written article and vision behind -> https://alejogb1.vercel.app/blog/search-tool-g2-alternative
Search-Tool (created by me):
 a market-intelligence tool that connects the Google Ads API + LLMs ...  meant to be a Semrush.com or Ahrefs alternative.

● it’s something new, and it replaces heavy, complex infrastructure orchestrations like the
kind you see at ahrefs’ big-data systems: https://ahrefs.com/big-data (read more about the computers of Ahrefs)

● designed based on the experience i gained working for a year at top.legal, analyzing
Google Search data from our potential market, with the goal of “getting closer to the
market” (go-to-market)

UI: https://search-tool2.vercel.app/
Backend API: https://search-tool-vwmv.onrender.com
Repo: https://github.com/Alejogb1/search-tool
RUN Api: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload   

## Project Structure

The project is organized into the following main directories:

-   `api/`: Contains the FastAPI application that exposes the analysis endpoints.
-   `core/`: Houses the core business logic of the application, including services for analysis orchestration, keyword clustering, and enrichment.
-   `data_access/`: Manages all data-related operations, including database models and repository classes for interacting with the database.
-   `integrations/`: Includes modules for connecting to external services like Google Ads, BigQuery, and Large Language Models (LLMs).
-   `alembic/`: Contains database migration scripts managed by Alembic.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Alejogb1/search-tool.git
    cd search-tool
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add the necessary configuration for database connections and external APIs. Refer to the existing `integrations` and `data_access` modules to see what is required.

5.  **Run database migrations:**
    ```bash
    alembic upgrade head
    ```

## Usage

To run the application, use the following command:

```bash
uvicorn api.main:app --reload
```

The API documentation will be available at `http://127.0.0.1:8000/docs`.

### API Endpoints

The main API endpoint is located at `/api/v1/analysis/`. You can send a POST request with the required parameters to trigger a new market analysis.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.
