# CSV Translator Frontend (React + Vite)

## Configure backend API URL (no code changes)

### Option A: Runtime config (recommended for offline package)

After you build/copy the `dist/` folder, you can edit:

- `dist/config.json`

Example:

```json
{ "apiUrl": "http://192.168.1.10:8000/api" }
```

No rebuild needed.

### Option B: Build-time env (Vite)

The frontend also supports a **build-time** env var:

- `VITE_API_URL` (example: `http://192.168.1.10:8000/api`)

If neither runtime config nor env is set, it falls back to:

- `${window.location.origin}/api` (useful when you serve frontend and backend on the same host)

### How to set it

Create a file `env.local` in `frontend/`:

```bash
VITE_API_URL=http://localhost:8000/api
```

Then rebuild:

```bash
npm run build
```

## Build

```bash
npm install
npm run build
```

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
