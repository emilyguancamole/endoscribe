// API client for EndoScribe backend

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'error';
  transcription_service?: string;
  llm_available?: boolean;
  whisper_available?: boolean;
  device?: string;
}

export interface ProcessRequest {
  procedure_type: string;
  transcript: string;
  session_id?: string;
}

export interface ProcessResponse {
  status: string;
  procedure_type: string;
  colonoscopy_data?: any;
  polyps_data?: any[];
  procedure_data?: any;
  pep_risk_data?: any;
  pep_risk_score?: number;
  pep_risk_category?: string;
  raw_output?: string;
  formatted_note?: string;
}

export interface SessionData {
  session_id: string;
  procedure_type: string;
  transcript: string;
  created_at: string;
  processed: boolean;
  results?: ProcessResponse;
}

export interface SaveSessionRequest {
  note_content: string;
  procedure_type?: string;
  transcript?: string;
  results?: ProcessResponse;
}

// Health check
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch('/health');
  if (!response.ok) {
    throw new Error('Health check failed');
  }
  return response.json();
}

// Process transcript
export async function processTranscript(data: ProcessRequest): Promise<ProcessResponse> {
  const response = await fetch('/api/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Failed to process transcript');
  }
  return response.json();
}

// Get session data //todo need to add this endpoint to the backend
export async function getSession(sessionId: string): Promise<SessionData | null> {
  try {
    const response = await fetch(`/api/sessions/${sessionId}`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

// List all sessions //todo need to add this endpoint to the backend
export async function listSessions(): Promise<SessionData[]> {
  try {
    const response = await fetch('/api/sessions');
    if (!response.ok) return [];
    return response.json();
  } catch (error) {
    return [];
  }
}

export async function saveSessionNote(sessionId: string, payload: SaveSessionRequest): Promise<boolean> {
  const response = await fetch(`/api/sessions/${sessionId}/save`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return response.ok;
}

// WebSocket message types. needs to match python server.py
export interface WebSocketMessage {
  type: string;
  data?: {
    text?: string;
    session_id?: string;
    chunk_count?: number;
    timestamp?: number;
    [key: string]: any;
  } | null;
  text?: string | null;  // Some messages send text directly
  message?: string | null;
  session_id?: string | null;
  error?: string | null;
}

// WebSocket connection helper
export class TranscriptionWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;

  constructor(
    private onMessage: (message: WebSocketMessage) => void,
    private onConnect: () => void,
    private onDisconnect: () => void,
    private onError: (error: string) => void
  ) { }

  connect(): void {
    // In dev, Vite proxies /ws to the backend (localhost:8001)
    // In prod, the app is served from the same origin as the API
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

    // Use the current host, which will be proxied by Vite in dev mode
    const wsUrl = `${protocol}//${window.location.host}/ws/transcribe`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onConnect();
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.onError('WebSocket connection error');
    };

    this.ws.onclose = () => {
      this.onDisconnect();

      // Attempt to reconnect
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), 1000 * this.reconnectAttempts);
      }
    };
  }

  send(data: ArrayBuffer | string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  sendJSON(data: any): void {
    this.send(JSON.stringify(data));
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
