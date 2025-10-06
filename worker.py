#!/usr/bin/env python3
"""
RQ Worker for processing background jobs
Run this with: python worker.py
"""

import os
import redis
import rq
from core.services.job_queue import redis_conn

def start_worker():
    """Start the RQ worker"""
    print("Starting RQ worker...")

    # Create worker for the keyword_expansion queue
    worker = rq.Worker(['keyword_expansion'], connection=redis_conn)

    print("Worker started. Waiting for jobs...")
    print(f"Redis URL: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")

    # Start processing jobs
    worker.work()

if __name__ == "__main__":
    start_worker()
