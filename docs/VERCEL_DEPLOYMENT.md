# Deploying OpsPilot AI to Vercel

Target team: `krishagrawal07s-projects`. Source repository: `KrishAgarwal04/claimlens`.

## Entrypoint

`asgi.py` exports a FastAPI application with the existing Streamlit dashboard
mounted at `/`. The parent application manages Streamlit's ASGI lifespan.
`/api/health` reports server readiness; it does not certify model inference.
The original local command `streamlit run app/main.py` continues to work.

`pyproject.toml` explicitly selects `asgi:app`, avoiding accidental detection of
the UI script `app/main.py` as an ASGI entrypoint. `vercel.json` selects the
FastAPI preset. `.python-version` requests Python 3.12, matching local model use.

## Platform requirements

- Enable Fluid compute: Vercel's WebSocket support requires it.
- Preserve WebSocket upgrades for Streamlit's `/_stcore/stream` endpoint.
- Include the three source CSVs, the `app/` and `src/` modules, active model
  artifacts, and `.streamlit/config.toml` in the deployment.
- Use `requirements.txt`; the optional experiment dependencies are not needed.
- Check the actual build for Python dependency and function-bundle size limits.
- Vercel function duration limits can end an interactive connection. Reconnects
  or instance changes may lose session-only ticket changes; this setup does not
  add durable workflow storage or cross-instance session affinity.

## Account and deployment

The Vercel integration must be connected to an account with access to the target
team, or the Vercel CLI must be authenticated. A dashboard link alone grants no
deployment access. Never put tokens in source files or chat.

When using the CLI from the repository root:

```powershell
vercel link --scope krishagrawal07s-projects
vercel deploy --scope krishagrawal07s-projects
```

First verify the preview, then deploy production:

```powershell
vercel deploy --prod --scope krishagrawal07s-projects
```

If using Git import, choose the target team, import the repository, use the
repository root and FastAPI preset, and keep Fluid compute enabled. The new
deployment files must be committed and pushed before importing that revision.

## Verification

1. `/api/health` responds successfully.
2. `/` serves Streamlit's HTML and static assets load.
3. The browser establishes `/_stcore/stream` and renders the actual dashboard.
4. Models load and risk predictions appear.
5. Supervisor All/High/Medium/Low filters and an agent claim action work.
6. Inspect deployment logs for missing files, package errors, and function limits.

A successful HTTP shell response alone is insufficient to call the deployment
working. Hosting compatibility is provisional until the actual Vercel build and
interactive browser flow pass.

## Sources

- [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python)
- [Vercel FastAPI deployment](https://vercel.com/docs/frameworks/backend/fastapi)
- [Vercel WebSockets and Fluid compute](https://vercel.com/docs/functions/websockets)
- [Streamlit ASGI App](https://docs.streamlit.io/develop/api-reference/server/st.app)
