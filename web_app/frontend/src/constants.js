// Configuration Constants
export const PROCESSING_TRANSCRIPTION_TIMEOUT_MS = 6000;
export const WEBSOCKET_FINALIZING_TIMEOUT_MS = 6000;
export const AUDIO_CHUNK_INTERVAL_MS = 5000;
export const ERROR_DISPLAY_DURATION_MS = 5000;

export const PROCEDURE_TYPES = [
  { value: 'col', label: 'Colonoscopy' },
  { value: 'eus', label: 'EUS' },
  { value: 'ercp', label: 'ERCP' },
  { value: 'egd', label: 'EGD' }
];
