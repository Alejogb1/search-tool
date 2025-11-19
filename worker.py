#!/usr/bin/env python3
"""
RQ Worker for processing background jobs with recovery and health monitoring
Run this with: python worker.py
"""

import os
import redis
import rq
import time
import threading
import logging
import sys
from core.services.job_queue import redis_conn, recover_failed_jobs, update_worker_health

# Configure comprehensive logging to stdout for Render compatibility
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

def health_monitor_worker(worker_id: str):
    """Background thread to monitor worker health"""
    while True:
        try:
            update_worker_health(worker_id, "alive")
            time.sleep(30)  # Update health every 30 seconds
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            time.sleep(10)

def start_worker():
    """Start the RQ worker with recovery and monitoring"""
    logger.info("🚀 Starting RQ worker with recovery and monitoring...")

    # Log environment variables for debugging split-brain issues
    redis_url = os.getenv('REDIS_URL', 'NOT_SET')
    database_url = os.getenv('DATABASE_URL', 'NOT_SET')
    logger.info(f"🔧 Worker Process Environment - REDIS_URL: {redis_url[:50]}...")
    logger.info(f"🔧 Worker Process Environment - DATABASE_URL: {database_url[:50]}...")

    # Generate unique worker ID
    worker_id = f"worker_{os.getpid()}_{int(time.time())}"
    logger.info(f"👷 Worker ID: {worker_id}")

    # Start health monitoring in background thread
    health_thread = threading.Thread(target=health_monitor_worker, args=(worker_id,), daemon=True)
    health_thread.start()
    logger.info("💓 Health monitoring started")

    # Attempt to recover any failed jobs before starting
    try:
        logger.info("🔄 Checking for jobs to recover...")
        recover_failed_jobs()
    except Exception as recovery_error:
        logger.warning(f"⚠️ Job recovery failed: {recovery_error}")

    # Create worker for the keyword_expansion queue
    worker = rq.Worker(['keyword_expansion'], connection=redis_conn)

    logger.info("✅ Worker started. Waiting for jobs...")
    logger.info(f"🔗 Redis URL: {redis_url}")
    logger.info(f"📋 Listening on queue: keyword_expansion")

    try:
        # Start processing jobs
        worker.work()
    except KeyboardInterrupt:
        logger.info("🛑 Worker interrupted, shutting down gracefully...")
        update_worker_health(worker_id, "stopped")
    except Exception as e:
        logger.error(f"💥 Worker crashed: {e}")
        update_worker_health(worker_id, "crashed")
        raise
    finally:
        logger.info("👋 Worker shutdown complete")

if __name__ == "__main__":
    start_worker()
