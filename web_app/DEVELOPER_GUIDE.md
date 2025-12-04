# EndoScribe Web App - Developer Guide

## Quick Start

### Running the App
```bash
# Option 1: Quick script (builds + runs)
./web_app/run_react.sh

# Option 2: Manual
cd web_app/frontend && npm run build && cd ../..
uvicorn web_app.server:app --host localhost --port 8001
```

Access at: http://localhost:8001

### Development Mode (with hot reload)
```bash
# Terminal 1: Start backend
uvicorn web_app.server:app --host localhost --port 8001 --reload

# Terminal 2: Start Vite dev server
cd web_app/frontend
npm run dev
```

Access dev server at: http://localhost:5173 (auto-proxies to backend)

## Project Structure

### Component Architecture
```
App (main state container)
├── Header (health status)
├── ProcedureSelector (dropdown)
├── AudioRecorder (recording controls)
├── TranscriptionDisplay (transcript + buttons)
├── ResultsDisplay (formatted data tables)
└── ErrorDisplay (error alerts)
```

### State Flow
1. **Audio Recording**: `useAudioRecorder` hook → chunks sent via WebSocket
2. **Transcription**: `useWebSocketTranscription` hook → real-time updates
3. **Processing**: API call → display results in `ResultsDisplay`
4. **PEP Risk**: Separate API call (ERCP only) → display in separate card

## Adding New Features

### Adding a New Component
```jsx
// 1. Create component in src/components/MyComponent.jsx
export function MyComponent({ prop1, prop2 }) {
  return <div>{prop1} {prop2}</div>;
}

// 2. Import in App.jsx
import { MyComponent } from './components/MyComponent';

// 3. Use in render

### Adding a New API Endpoint
```jsx
// 1. Add function in src/utils/api.js
export async function myNewEndpoint(param) {
  const response = await fetch('/api/my-endpoint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ param })
  });
  return response.json();
}

// 2. Use in component
import { myNewEndpoint } from './utils/api';
const result = await myNewEndpoint('value');
```

## Common Tasks

### Rebuilding After Changes
```bash
cd web_app/frontend
npm run build
# Restart server to see changes
```

### Checking Build Output
```bash
ls -lh web_app/static/dist/
# Should see index.html and assets/ folder
```

### Debugging
1. Open browser DevTools (F12)
2. Check Console for errors
3. Check Network tab for failed requests
4. React DevTools extension for component inspection

### Reverting to Legacy Version
```bash
rm -rf web_app/static/dist/
# Restart server - will serve Alpine.js template
```

## Best Practices
### State Management
- Keep state close to where it's used
- Lift state up only when multiple components need it
- Use `useCallback` for functions passed as props
- Use `useEffect` for side effects only

### Performance
- Vite build is already optimized
- React.StrictMode catches common issues
- Lazy load heavy components if needed
- Use React DevTools Profiler for optimization

## Troubleshooting

### Build Fails
```bash
# Clear node_modules and reinstall
cd web_app/frontend
rm -rf node_modules package-lock.json
npm install
```

### Changes Not Reflecting
```bash
# Hard refresh browser (Ctrl+Shift+R)
# Or clear dist and rebuild
rm -rf web_app/static/dist
npm run build
```

### WebSocket Not Connecting
- Check backend is running on port 8001
- Check browser console for connection errors
- Verify proxy configuration in vite.config.js

## File Organization

```
frontend/src/
├── App.jsx                 # Main container - owns all state
├── main.jsx               # React entry point
├── constants.js           # Config values, procedure types
├── components/            # UI components (presentational)
│   ├── Header.jsx
│   ├── ProcedureSelector.jsx
│   ├── AudioRecorder.jsx
│   ├── TranscriptionDisplay.jsx
│   ├── ResultsDisplay.jsx
│   └── ErrorDisplay.jsx
├── hooks/                 # Custom hooks (business logic)
│   ├── useAudioRecorder.js
│   └── useWebSocketTranscription.js
└── utils/                 # Utilities
    └── api.js            # API client functions
```

## Testing (Future Enhancement)

### Component Tests (Vitest + React Testing Library)
```jsx
import { render, screen } from '@testing-library/react';
import { Header } from './Header';

test('renders health status', () => {
  render(<Header healthStatus="healthy" />);
  expect(screen.getByText('System Ready')).toBeInTheDocument();
});
```

### Hook Tests (React Hooks Testing Library)
```jsx
import { renderHook, act } from '@testing-library/react';
import { useAudioRecorder } from './useAudioRecorder';

test('starts recording', async () => {
  const { result } = renderHook(() => useAudioRecorder());
  await act(async () => {
    await result.current.startRecording();
  });
  expect(result.current.recording).toBe(true);
});
```
