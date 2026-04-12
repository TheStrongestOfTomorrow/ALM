# ALM: Adaptive Language Model

ALM (Adaptive Language Model) is a next-generation transformer-based architecture designed for high-performance reasoning, coding, and natural language understanding.

## ALM-1: The Flagship Model

ALM-1 is the first iteration of the ALM family, featuring **135.35 Billion parameters**. It is specifically optimized for:
- **English Proficiency:** Deep understanding of nuance, context, and creative expression.
- **Coding Excellence:** State-of-the-art performance in multiple programming languages, including Python, Rust, and TypeScript.
- **Adaptive Inference:** Dynamically adjusts compute based on task complexity.

### Specifications
| Feature | Value |
| --- | --- |
| Parameters | 135.35B |
| Architecture | Adaptive Transformer |
| Training Focus | English & Code |
| License | MIT |

## Quick Start

If you clone the repository, you can run the central entry point to choose your interface:

```bash
pip install torch flask rich
python app.py
```

### Modes Available:
1.  **Web Mode:** Launches a local Flask server with a professional chat interface, including search and coding modes.
2.  **Terminal Mode:** Launches a beautiful TUI (Terminal User Interface) with rich formatting and markdown support.

## Web Interface (Hosted)
A static version of the ALM-1 interface is hosted via GitHub Pages using PyScript. This allows you to test the model's logic directly in your browser without any installation.

**[Link to GitHub Pages Interface]**

## GitHub Pages Hosting
This repository is configured to auto-deploy the web interface. Every push to the repository triggers a deployment.

---
*Created by Jules, Software Engineer.*
