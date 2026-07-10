import logging
import json
import sys
from datetime import datetime
from typing import Any

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger

log = get_logger("bifrost")

def log_routing_decision(prompt: str, category: str, confidence: float, score: float, target: str, reasoning: str) -> None:
    log.info(
        "Routing decision made", 
        extra={
            "extra_data": {
                "event": "routing_decision",
                "prompt_length": len(prompt),
                "category": category,
                "confidence": confidence,
                "complexity_score": score,
                "routed_to": target,
                "reasoning": reasoning
            }
        }
    )

def log_task_completion(task_id: str, latency_ms: float, target: str, tokens: int, success: bool, error: str = None) -> None:
    log.info(
        "Task completed",
        extra={
            "extra_data": {
                "event": "task_completion",
                "task_id": task_id,
                "latency_ms": latency_ms,
                "tier": target,
                "total_tokens": tokens,
                "success": success,
                "error": error
            }
        }
    )
