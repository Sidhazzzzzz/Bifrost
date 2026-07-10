"""
Bifrost Verification Script
Starts the FastAPI server in the background, performs an asynchronous validation check,
asserts correct routing metadata, and prints success.
"""

from __future__ import annotations

import json
import subprocess
import time
import sys
import os

try:
    import httpx
except ImportError:
    print("[Verify] Please run 'pip install httpx' before executing verify.py")
    sys.exit(1)


def main() -> None:
    print("[Verify] Starting Bifrost server in the background...")
    
    # Set BIFROST_MODE to serve so it runs the FastAPI application
    env = os.environ.copy()
    env["BIFROST_MODE"] = "serve"
    env["PORT"] = "8000"
    
    # Start the server process
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main", "--serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Give the server 3 seconds to spin up and bind to port 8000
    time.sleep(3)
    
    # Check if the process crashed immediately
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        print("[Verify] ERROR: Server failed to start:")
        print(stderr.decode())
        sys.exit(1)
        
    print("[Verify] Server running. Performing async curl assertion...")
    
    client = httpx.Client()
    try:
        # 1. Health check assertion
        health_resp = client.get("http://localhost:8000/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
        print("[Verify] Health check assertion: PASSED OK")
        
        # 2. Chat routing assertion
        payload = {"message": "Is the sentiment of this text positive: 'I love AMD ROCm!'"}
        headers = {"Content-Type": "application/json"}
        
        chat_resp = client.post(
            "http://localhost:8000/v1/chat",
            json=payload,
            headers=headers,
            timeout=60.0
        )
        
        assert chat_resp.status_code == 200, f"Chat endpoint failed: {chat_resp.status_code}"
        data = chat_resp.json()
        
        # Validate schema contract
        assert "response" in data, "Response missing 'response' field"
        assert "routed_to" in data, "Response missing 'routed_to' field"
        assert data["routed_to"] in ["LOCAL", "REMOTE"], f"Invalid routed_to tier: {data['routed_to']}"
        
        print("[Verify] Chat routing contract assertion: PASSED OK")
        print(f"[Verify] Sample response: '{data['response']}' (routed to: {data['routed_to']})")
        print("\n[Verify] ALL VERIFICATION ASSERTIONS PASSED SUCCESSFULLY!")
        success = True
        
    except AssertionError as exc:
        print(f"[Verify] ASSERTION FAILED: {exc}")
        success = False
    except Exception as exc:
        print(f"[Verify] ERROR calling server: {exc}")
        success = False
    finally:
        # Shut down client and terminate the background server process
        client.close()
        print("[Verify] Stopping background server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
