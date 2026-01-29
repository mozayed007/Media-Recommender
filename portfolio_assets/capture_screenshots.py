"""
Portfolio Screenshot & Diagram Generator
Captures high-res screenshots and renders Mermaid diagrams to PNG
"""

import asyncio
from playwright.async_api import async_playwright
import os

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# High DPI settings for crisp images
VIEWPORT = {"width": 1920, "height": 1080}
DEVICE_SCALE_FACTOR = 2  # 2x for Retina/HiDPI quality

async def capture_frontend_screenshots():
    """Capture screenshots of the frontend application"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=DEVICE_SCALE_FACTOR
        )
        page = await context.new_page()
        
        print("[CAMERA] Capturing frontend screenshots...")
        
        # 1. Hero Section
        print("  -> Hero section...")
        await page.goto("http://localhost:3000", wait_until="networkidle")
        await page.wait_for_timeout(2000)  # Wait for animations
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "01_hero_section.png"),
            full_page=False
        )
        
        # 2. Full page with Trending
        print("  -> Full page with trending...")
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "02_full_page_trending.png"),
            full_page=True
        )
        
        # 3. Search Results
        print("  -> Search results...")
        search_input = page.locator("input[placeholder*='Ask for something']")
        await search_input.fill("emotional anime that will make me cry")
        await page.click("button:has-text('Search')")
        await page.wait_for_timeout(3000)  # Wait for results
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "03_search_results.png"),
            full_page=False
        )
        
        # 4. Anime Card Hover (if possible)
        print("  -> Hovering on anime card...")
        cards = page.locator("[class*='group overflow-hidden cursor-pointer']")
        if await cards.count() > 0:
            await cards.first.hover()
            await page.wait_for_timeout(500)
            await page.screenshot(
                path=os.path.join(SCREENSHOTS_DIR, "04_card_hover.png"),
                full_page=False
            )
        
        # 5. Click on anime to open modal
        print("  -> Anime details modal...")
        if await cards.count() > 0:
            await cards.first.click()
            await page.wait_for_timeout(1000)
            await page.screenshot(
                path=os.path.join(SCREENSHOTS_DIR, "05_anime_modal.png"),
                full_page=False
            )
            
            # 6. Click Find Similar for recommendations
            print("  -> Getting recommendations...")
            find_similar_btn = page.locator("button:has-text('Find Similar')")
            if await find_similar_btn.count() > 0:
                await find_similar_btn.click()
                await page.wait_for_timeout(3000)
                await page.screenshot(
                    path=os.path.join(SCREENSHOTS_DIR, "06_recommendations.png"),
                    full_page=True
                )
        
        # 7. Discover mode
        print("  -> Discover mode...")
        await page.goto("http://localhost:3000", wait_until="networkidle")
        await page.wait_for_timeout(1000)
        discover_btn = page.locator("button:has-text('Discover')")
        if await discover_btn.count() > 0:
            await discover_btn.click()
            await page.wait_for_timeout(3000)
            await page.screenshot(
                path=os.path.join(SCREENSHOTS_DIR, "07_discover_mode.png"),
                full_page=True
            )
        
        await browser.close()
        print("[OK] Frontend screenshots captured!")

async def capture_api_docs():
    """Capture API documentation screenshot"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=DEVICE_SCALE_FACTOR
        )
        page = await context.new_page()
        
        print("[CAMERA] Capturing API docs...")
        await page.goto("http://localhost:8000/docs", wait_until="networkidle")
        await page.wait_for_timeout(2000)
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "08_api_docs.png"),
            full_page=True
        )
        await browser.close()
        print("[OK] API docs screenshot captured!")

async def render_mermaid_diagrams():
    """Render Mermaid diagrams to high-res PNG using Mermaid.live"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1600, "height": 1200},
            device_scale_factor=3  # 3x for extra crisp diagrams
        )
        page = await context.new_page()
        
        print("[CHART] Rendering Mermaid diagrams...")
        
        # Architecture diagram
        architecture_mermaid = '''
graph TD
    subgraph Frontend["Frontend - Next.js 16"]
        UI[React 19 UI]
        Search[Semantic Search]
        Cards[Anime Cards]
        Modal[Details Modal]
        Watchlist[Watchlist]
    end
    
    subgraph Backend["Backend - FastAPI"]
        API[REST API]
        Recommender[Recommender Service]
        Embedding[Embedding Service]
        ContentFilter[Content Filter]
    end
    
    subgraph ML["ML Layer"]
        Model[EmbeddingGemma-300M]
        Vectors[768-dim Vectors]
    end
    
    subgraph Storage["Data Layer"]
        Qdrant[(Qdrant Vector DB)]
        Parquet[(Parquet Files)]
    end
    
    UI --> API
    Search --> API
    API --> Recommender
    Recommender --> Embedding
    Recommender --> ContentFilter
    Embedding --> Model
    Model --> Vectors
    Vectors --> Qdrant
    Recommender --> Qdrant
    ContentFilter --> Parquet
    
    style Frontend fill:#1a1a2e,stroke:#7c3aed,color:#fff
    style Backend fill:#16213e,stroke:#0ea5e9,color:#fff
    style ML fill:#1e3a5f,stroke:#f59e0b,color:#fff
    style Storage fill:#134e4a,stroke:#10b981,color:#fff
'''
        
        # Data flow diagram
        dataflow_mermaid = '''
flowchart LR
    subgraph Input
        Q[/"User Query"/]
    end
    
    subgraph Processing
        E[Embed Query]
        S[Vector Search]
        R[Re-rank Results]
        F[Apply Filters]
    end
    
    subgraph Output
        RES[/"Recommendations"/]
    end
    
    Q --> E
    E --> S
    S --> R
    R --> F
    F --> RES
    
    style Input fill:#7c3aed,stroke:#7c3aed,color:#fff
    style Processing fill:#0ea5e9,stroke:#0ea5e9,color:#fff
    style Output fill:#10b981,stroke:#10b981,color:#fff
'''

        # Tech stack diagram
        techstack_mermaid = '''
graph TB
    subgraph Client["Client Layer"]
        Next[Next.js 16]
        React[React 19]
        TS[TypeScript]
        TW[Tailwind CSS]
        FM[Framer Motion]
    end
    
    subgraph Server["Server Layer"]
        FastAPI[FastAPI]
        Python[Python 3.10+]
        Pydantic[Pydantic]
        Uvicorn[Uvicorn]
    end
    
    subgraph AI["AI/ML Layer"]
        ST[Sentence Transformers]
        Gemma[EmbeddingGemma]
        Torch[PyTorch]
    end
    
    subgraph Data["Data Layer"]
        Qdrant[(Qdrant)]
        Pandas[Pandas]
        Parquet[(Parquet)]
    end
    
    Client --> Server
    Server --> AI
    Server --> Data
    AI --> Data
    
    style Client fill:#7c3aed,stroke:#a78bfa,color:#fff
    style Server fill:#0ea5e9,stroke:#38bdf8,color:#fff
    style AI fill:#f59e0b,stroke:#fbbf24,color:#fff
    style Data fill:#10b981,stroke:#34d399,color:#fff
'''

        diagrams = [
            ("architecture_diagram", architecture_mermaid),
            ("dataflow_diagram", dataflow_mermaid),
            ("techstack_diagram", techstack_mermaid),
        ]
        
        for name, mermaid_code in diagrams:
            print(f"  -> Rendering {name}...")
            
            # Use mermaid.live to render
            import base64
            import json
            
            # Encode the mermaid code for the URL
            state = {
                "code": mermaid_code,
                "mermaid": {"theme": "dark"},
                "autoSync": True,
                "updateDiagram": True
            }
            state_json = json.dumps(state)
            state_base64 = base64.urlsafe_b64encode(state_json.encode()).decode()
            
            url = f"https://mermaid.live/edit#pako:{state_base64}"
            
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(3000)
            
            # Find the preview pane and screenshot it
            preview = page.locator("#view")
            if await preview.count() > 0:
                await preview.screenshot(
                    path=os.path.join(SCREENSHOTS_DIR, f"{name}.png")
                )
            else:
                # Fallback: screenshot the whole page
                await page.screenshot(
                    path=os.path.join(SCREENSHOTS_DIR, f"{name}.png"),
                    full_page=False
                )
        
        await browser.close()
        print("[OK] Mermaid diagrams rendered!")

async def main():
    print("Starting Portfolio Asset Generation...\n")
    
    try:
        await capture_frontend_screenshots()
    except Exception as e:
        print(f"Frontend screenshots error: {e}")
    
    try:
        await capture_api_docs()
    except Exception as e:
        print(f"API docs screenshot error: {e}")
    
    try:
        await render_mermaid_diagrams()
    except Exception as e:
        print(f"Mermaid diagrams error: {e}")
    
    print(f"\nDone! Screenshots saved to: {SCREENSHOTS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())

