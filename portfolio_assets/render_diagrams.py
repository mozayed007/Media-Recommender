"""
Render Mermaid diagrams to high-res PNG locally using Playwright
"""

import asyncio
from playwright.async_api import async_playwright
import os

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Architecture diagram
ARCHITECTURE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: #0f0f1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 40px;
        }
        .mermaid {
            background: #0f0f1a;
        }
    </style>
</head>
<body>
    <pre class="mermaid">
graph TD
    subgraph Frontend["Frontend - Next.js 16 + React 19"]
        UI[UI Components]
        Search[Semantic Search]
        Cards[Anime Cards]
        Modal[Details Modal]
        WL[Watchlist]
    end
    
    subgraph Backend["Backend - FastAPI + Python"]
        API[REST API]
        Rec[Recommender Service]
        Emb[Embedding Service]
        CF[Content Filter]
    end
    
    subgraph ML["ML/AI Layer"]
        Model[EmbeddingGemma-300M]
        Vec[768-dim Vectors]
    end
    
    subgraph Storage["Data Layer"]
        Qdrant[(Qdrant Vector DB)]
        Parquet[(Parquet Files)]
    end
    
    UI --> API
    Search --> API
    API --> Rec
    Rec --> Emb
    Rec --> CF
    Emb --> Model
    Model --> Vec
    Vec --> Qdrant
    Rec --> Qdrant
    CF --> Parquet
    
    style Frontend fill:#1e1b4b,stroke:#7c3aed,color:#e0e7ff
    style Backend fill:#172554,stroke:#3b82f6,color:#dbeafe
    style ML fill:#422006,stroke:#f59e0b,color:#fef3c7
    style Storage fill:#14532d,stroke:#22c55e,color:#dcfce7
    </pre>
    <script>
        mermaid.initialize({ 
            startOnLoad: true, 
            theme: 'dark',
            themeVariables: {
                primaryColor: '#7c3aed',
                primaryTextColor: '#fff',
                primaryBorderColor: '#7c3aed',
                lineColor: '#6366f1',
                secondaryColor: '#3b82f6',
                tertiaryColor: '#1e1b4b'
            }
        });
    </script>
</body>
</html>
'''

# Data flow diagram
DATAFLOW_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: #0f0f1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 40px;
        }
    </style>
</head>
<body>
    <pre class="mermaid">
flowchart LR
    subgraph Input["Input"]
        Q[/"User Query"/]
    end
    
    subgraph Processing["Processing Pipeline"]
        E[Embed Query<br/>768-dim vector]
        S[Vector Search<br/>Qdrant HNSW]
        R[Re-rank Results<br/>Score + Popularity]
        F[Apply Filters<br/>Genre, Type]
    end
    
    subgraph Output["Output"]
        RES[/"Top-N Recommendations"/]
    end
    
    Q --> E
    E --> S
    S --> R
    R --> F
    F --> RES
    
    style Input fill:#581c87,stroke:#a855f7,color:#f3e8ff
    style Processing fill:#1e3a8a,stroke:#3b82f6,color:#dbeafe
    style Output fill:#166534,stroke:#22c55e,color:#dcfce7
    </pre>
    <script>
        mermaid.initialize({ 
            startOnLoad: true, 
            theme: 'dark'
        });
    </script>
</body>
</html>
'''

# Tech stack diagram
TECHSTACK_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: #0f0f1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 40px;
        }
    </style>
</head>
<body>
    <pre class="mermaid">
graph TB
    subgraph Client["Client Layer"]
        Next["Next.js 16"]
        React["React 19"]
        TS["TypeScript"]
        TW["Tailwind CSS"]
        FM["Framer Motion"]
        Shadcn["Shadcn UI"]
    end
    
    subgraph Server["Server Layer"]
        FastAPI["FastAPI"]
        Python["Python 3.10+"]
        Pydantic["Pydantic"]
        Uvicorn["Uvicorn ASGI"]
    end
    
    subgraph AI["AI/ML Layer"]
        ST["Sentence Transformers"]
        Gemma["Google EmbeddingGemma"]
        Torch["PyTorch"]
        SKlearn["Scikit-learn"]
    end
    
    subgraph Data["Data Layer"]
        Qdrant[("Qdrant<br/>24K+ vectors")]
        Pandas["Pandas"]
        Parquet[("Parquet<br/>29K records")]
    end
    
    Client --> Server
    Server --> AI
    Server --> Data
    AI --> Data
    
    style Client fill:#581c87,stroke:#a855f7,color:#f3e8ff
    style Server fill:#1e3a8a,stroke:#3b82f6,color:#dbeafe
    style AI fill:#78350f,stroke:#f59e0b,color:#fef3c7
    style Data fill:#166534,stroke:#22c55e,color:#dcfce7
    </pre>
    <script>
        mermaid.initialize({ 
            startOnLoad: true, 
            theme: 'dark'
        });
    </script>
</body>
</html>
'''

# Class diagram
CLASS_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: #0f0f1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 40px;
        }
    </style>
</head>
<body>
    <pre class="mermaid">
classDiagram
    direction LR
    
    class AbstractEmbeddingModel {
        &lt;&lt;Interface&gt;&gt;
        +get_text_embedding()
        +get_text_embeddings()
    }
    
    class SentenceTransformerModel {
        -model
        +get_text_embedding()
        +get_text_embeddings()
    }
    
    class AbstractVectorDatabase {
        &lt;&lt;Interface&gt;&gt;
        +add_items()
        +find_similar()
        +count()
    }
    
    class QdrantVectorDatabase {
        -client
        -collection_name
        +add_items()
        +find_similar()
        +count()
    }
    
    class RecommenderService {
        -embedding_model
        -vector_db
        -data
        +initialize()
        +search()
        +recommend()
        +get_trending()
    }
    
    AbstractEmbeddingModel <|-- SentenceTransformerModel
    AbstractVectorDatabase <|-- QdrantVectorDatabase
    RecommenderService o-- AbstractEmbeddingModel
    RecommenderService o-- AbstractVectorDatabase
    </pre>
    <script>
        mermaid.initialize({ 
            startOnLoad: true, 
            theme: 'dark'
        });
    </script>
</body>
</html>
'''

async def render_diagrams():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1400, "height": 1000},
            device_scale_factor=3  # 3x for high DPI
        )
        page = await context.new_page()
        
        diagrams = [
            ("architecture_diagram", ARCHITECTURE_HTML),
            ("dataflow_diagram", DATAFLOW_HTML),
            ("techstack_diagram", TECHSTACK_HTML),
            ("class_diagram", CLASS_HTML),
        ]
        
        print("Rendering Mermaid diagrams...")
        
        for name, html_content in diagrams:
            print(f"  -> Rendering {name}...")
            
            # Load the HTML content directly
            await page.set_content(html_content)
            await page.wait_for_timeout(3000)  # Wait for Mermaid to render
            
            # Find the SVG element and screenshot it
            svg = page.locator("svg.mermaid")
            if await svg.count() > 0:
                await svg.screenshot(
                    path=os.path.join(SCREENSHOTS_DIR, f"{name}.png")
                )
            else:
                # Fallback: screenshot the pre element
                pre = page.locator("pre.mermaid")
                if await pre.count() > 0:
                    await pre.screenshot(
                        path=os.path.join(SCREENSHOTS_DIR, f"{name}.png")
                    )
                else:
                    # Last fallback: full page
                    await page.screenshot(
                        path=os.path.join(SCREENSHOTS_DIR, f"{name}.png"),
                        full_page=False
                    )
        
        await browser.close()
        print("[OK] All diagrams rendered!")

if __name__ == "__main__":
    asyncio.run(render_diagrams())

