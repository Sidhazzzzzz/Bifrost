import asyncio
from app.config import load_settings
from app.client import LLMClient
from app.classifier import Category, RouteTarget
from app.prompts import build_messages, MAX_TOKENS_HINT

async def main():
    settings = load_settings()
    client = LLMClient(
        local_base_url=settings.local_base_url,
        remote_base_url=settings.remote_base_url,
        remote_api_key=settings.remote_api_key,
        remote_fallback_model=settings.remote_model
    )
    
    prompts = [
        "Debug this Python code that causes a KeyError. def get_val(d, k1, k2): return d[k1][k2] where it might not exist",
        "Fix this React component that fails to re-render when the array state is mutated using push. function List({items}) { const [list, setList] = useState(items); const add = () => { list.push('new'); setList(list); }; return <div onClick={add}>{list.length}</div> }",
        "My SQL query is returning duplicate rows. Fix it using a window function or group by. SELECT u.id, u.name, o.order_date FROM users u LEFT JOIN orders o ON u.id = o.user_id",
        "Debug this C++ code giving a segmentation fault. void process(int* arr, int size) { for(int i=0; i<=size; i++) { arr[i] = i * 2; } }",
        "Fix this async Javascript code where the promises are executing sequentially instead of in parallel. async function fetchAll() { const a = await fetch('/a'); const b = await fetch('/b'); return [a, b]; }"
    ]
    
    truncations = 0
    for i, p in enumerate(prompts):
        messages = build_messages(p, Category.CODE_DEBUG)
        max_tokens = MAX_TOKENS_HINT[Category.CODE_DEBUG] # 50
        
        # force remote to see if it truncates
        res = await client.chat(messages, RouteTarget.REMOTE, settings.remote_model, max_tokens=max_tokens)
        print(f"Prompt {i+1}:")
        print(f"Completion Tokens: {res.completion_tokens}")
        print(f"Text length: {len(res.text)}")
        print(f"Truncated: {'YES' if res.completion_tokens >= max_tokens else 'NO'}")
        print(f"Content: {res.text}")
        print("-" * 40)
        
        if res.completion_tokens >= max_tokens:
            truncations += 1
            
    if truncations > 0:
        print(f"Found {truncations} truncations with cap {MAX_TOKENS_HINT[Category.CODE_DEBUG]}")
    else:
        print("No truncations found.")
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
