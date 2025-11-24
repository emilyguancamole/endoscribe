# EndoScribe Frontend (React)

This directory contains the refactored React-based frontend for the EndoScribe web application.

## Architecture

### Component Structure
- **App.jsx** - Main application container with state management
- **components/** - Reusable UI components
  - `Header.jsx` - App header with health status
  - `ProcedureSelector.jsx` - Procedure type dropdown
  - `AudioRecorder.jsx` - Recording controls and status
  - `TranscriptionDisplay.jsx` - Real-time transcription with action buttons
  - `ResultsDisplay.jsx` - Display extracted procedure data
  - `ErrorDisplay.jsx` - Error message alerts

### Custom Hooks
- **useAudioRecorder.js** - Manages audio recording with RecordRTC
- **useWebSocketTranscription.js** - Handles WebSocket connection for real-time transcription

### Utilities
- **api.js** - API client functions for backend communication
- **constants.js** - Configuration constants and procedure types

## Development

### Setup
```bash
cd web_app/frontend
npm install
```

### Run Development Server (with hot reload)
```bash
npm run dev
```
This starts Vite dev server on port 5173 with proxy to backend on port 8001.

### Build for Production
```bash
npm run build
```
Outputs optimized bundle to `web_app/static/dist/`

## Backend Integration

The FastAPI server automatically serves the React build from `static/dist/` when available. If the build doesn't exist, it falls back to the legacy Alpine.js template.

To run the full stack:
1. Build frontend: `cd web_app/frontend && npm run build`
2. Start backend: `uvicorn web_app.server:app --host localhost --port 8001`
3. Access at http://localhost:8001

## Features

All original functionality is preserved:
- ✅ Real-time audio recording and transcription
- ✅ WebSocket-based transcription updates
- ✅ Multiple procedure types (Colonoscopy, EUS, ERCP, EGD)
- ✅ PEP Risk calculation for ERCP procedures
- ✅ Structured data extraction and display
- ✅ Error handling and user feedback

## Migration Notes

The React version is a drop-in replacement for the Alpine.js version. All state management, API calls, and WebSocket handling have been preserved with improved modularity and maintainability.
