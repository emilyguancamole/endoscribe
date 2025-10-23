/**
 * EndoScribe Web App - Alpine.js Implementation
 * Handles audio recording, WebSocket communication, and UI updates
 */

document.addEventListener('alpine:init', () => {
    Alpine.data('endoscribe', () => ({
        // State
        recording: false,
        paused: false,
        mediaRecorder: null,
        websocket: null,
        sessionId: null,
        fullTranscript: '',
        procedureType: 'col',

        // UI State
        recordingStatus: '',
        recordingStatusClass: 'badge badge-neutral',
        errorMessage: '',
        showError: false,
        showProcessing: false,
        showResults: false,
        showColResults: false,
        showOtherResults: false,

        // Results Data
        colonoscopyData: {},
        polypsData: [],
        procedureData: {},

        // Computed Properties
        get canSubmit() {
            return !this.recording && this.fullTranscript.trim().length > 0;
        },

        get showSubmitButton() {
            return !this.recording && this.fullTranscript.trim().length > 0;
        },

        get showSessionInfo() {
            return this.sessionId !== null;
        },

        get showRecordingIndicator() {
            return this.recording && !this.paused;
        },

        get transcriptionDisplay() {
            return this.fullTranscript.trim() || 'Transcription will appear here as you speak...';
        },

        // Lifecycle - Initialize
        init() {
            console.log('EndoScribe app initialized with Alpine.js');
        },

        // Lifecycle - Cleanup
        destroy() {
            this.closeWebSocket();
            if (this.mediaRecorder) {
                this.mediaRecorder.stopRecording();
            }
        },

        // WebSocket Methods
        connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/transcribe`;

            console.log('Connecting to WebSocket:', wsUrl);
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.websocket.send(JSON.stringify({
                    type: 'start',
                    session_id: this.sessionId
                }));
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log('WebSocket message:', message);

                    switch (message.type) {
                        case 'status':
                            if (message.session_id) {
                                this.sessionId = message.session_id;
                            }
                            // Only show processing status if currently recording and not paused
                            if (message.message && message.message.includes('Processing') &&
                                this.recording && !this.paused) {
                                this.recordingStatus = message.message;
                                this.recordingStatusClass = 'badge badge-info';
                            }
                            break;

                        case 'transcript':
                            const transcriptText = message.data?.text || '';
                            if (transcriptText) {
                                this.updateTranscript(transcriptText, true);
                                if (this.recording && !this.paused) {
                                    this.recordingStatus = 'Recording';
                                    this.recordingStatusClass = 'badge badge-error';
                                }
                            }
                            break;

                        case 'error':
                            this.displayError(message.message || 'WebSocket error occurred');
                            break;
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.displayError('WebSocket connection error - server may be processing audio');
            };

            this.websocket.onclose = (event) => {
                console.log('WebSocket closed', event.code, event.reason);
                if (event.code !== 1000 && event.code !== 1001 && this.recording) {
                    this.displayError(`Connection closed unexpectedly (code: ${event.code}). Try again or reduce audio chunk length.`);
                }
            };
        },

        closeWebSocket() {
            if (this.websocket) {
                this.websocket.send(JSON.stringify({ type: 'end' }));
                this.websocket.close();
                this.websocket = null;
            }
        },

        // Transcript Management
        updateTranscript(text, append = true) {
            if (append) {
                this.fullTranscript += ' ' + text;
            } else {
                this.fullTranscript = text;
            }

            // Auto-scroll to bottom
            this.$nextTick(() => {
                const container = this.$refs.transcriptionContainer;
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },

        // Recording Methods
        async startRecording() {
            try {
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                // Connect WebSocket
                this.connectWebSocket();

                // Create RecordRTC instance
                this.mediaRecorder = new RecordRTC(stream, {
                    type: 'audio',
                    mimeType: 'audio/webm;codecs=opus',
                    recorderType: RecordRTC.StereoAudioRecorder,
                    timeSlice: 2000,
                    ondataavailable: async (blob) => {
                        if (blob.size > 0 && this.websocket?.readyState === WebSocket.OPEN) {
                            console.log('Sending audio chunk:', blob.size, 'bytes');
                            this.websocket.send(blob);
                        }
                    },
                    onstop: (blob) => {
                        console.log('RecordRTC stopped');
                        stream.getTracks().forEach(track => track.stop());
                    }
                });

                // Start recording
                this.mediaRecorder.startRecording();
                this.recording = true;
                this.paused = false;

                // Update status
                this.updateTranscript('', false); // Clear previous transcript
                this.recordingStatus = 'Recording';
                this.recordingStatusClass = 'badge badge-error';

            } catch (error) {
                console.error('Error starting recording:', error);
                this.displayError('Failed to start recording. Please check microphone permissions.');
            }
        },

        pauseRecording() {
            if (this.mediaRecorder && this.recording) {
                this.mediaRecorder.pauseRecording();
                this.paused = true;
                this.recordingStatus = 'Paused';
                this.recordingStatusClass = 'badge badge-warning';
            }
        },

        resumeRecording() {
            if (this.mediaRecorder && this.paused) {
                this.mediaRecorder.resumeRecording();
                this.paused = false;
                this.recordingStatus = 'Recording';
                this.recordingStatusClass = 'badge badge-error';
            }
        },

        stopRecording() {
            if (this.mediaRecorder) {
                this.mediaRecorder.stopRecording();
                this.recording = false;
                this.paused = false;

                // Close WebSocket
                this.closeWebSocket();

                // Update status
                this.recordingStatus = 'Stopped';
                this.recordingStatusClass = 'badge badge-neutral';
            }
        },

        // Process Transcript
        async processTranscript() {
            const transcript = this.fullTranscript.trim();

            if (!transcript) {
                this.displayError('No transcript available to process');
                return;
            }

            // Show processing status
            this.showProcessing = true;
            this.showResults = false;

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        transcript: transcript,
                        procedure_type: this.procedureType,
                        session_id: this.sessionId
                    })
                });

                const result = await response.json();

                // Hide processing status
                this.showProcessing = false;

                if (result.success) {
                    this.displayResults(result.data, result.procedure_type);
                } else {
                    this.displayError(result.error || 'Processing failed');
                }

            } catch (error) {
                console.error('Error processing transcript:', error);
                this.showProcessing = false;
                this.displayError('Failed to process transcript: ' + error.message);
            }
        },

        // Display Results
        displayResults(data, procedureType) {
            this.showResults = true;

            if (procedureType === 'col') {
                this.showColResults = true;
                this.showOtherResults = false;
                this.colonoscopyData = data.colonoscopy || {};
                this.polypsData = data.polyps || [];
            } else {
                this.showColResults = false;
                this.showOtherResults = true;
                this.procedureData = data;
            }
        },

        // Error Handling
        displayError(message) {
            this.errorMessage = message;
            this.showError = true;
            setTimeout(() => {
                this.showError = false;
            }, 5000);
        },

        // Helper: Create Data Table HTML
        createDataTable(data) {
            if (!data || Object.keys(data).length === 0) {
                return '<p class="text-base-content/70">No data available</p>';
            }

            let html = '<table class="table table-zebra w-full"><tbody>';
            for (const [key, value] of Object.entries(data)) {
                const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                let displayValue = value;

                if (Array.isArray(value)) {
                    displayValue = '<ul class="list-disc list-inside">' +
                        value.map(item => `<li>${item}</li>`).join('') +
                        '</ul>';
                } else if (typeof value === 'object' && value !== null) {
                    displayValue = JSON.stringify(value, null, 2);
                }

                html += `<tr><th class="w-1/3">${displayKey}</th><td>${displayValue}</td></tr>`;
            }
            html += '</tbody></table>';
            return html;
        },

        // Helper: Create Polyps Table HTML
        createPolypsTable(polyps) {
            if (!polyps || polyps.length === 0) {
                return '<p class="text-base-content/70">No polyps detected</p>';
            }

            let html = `
                <table class="table table-zebra w-full">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Size (mm)</th>
                            <th>Location</th>
                            <th>Resection</th>
                            <th>Method</th>
                            <th>NICE Class</th>
                            <th>Paris Class</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            polyps.forEach((polyp, index) => {
                html += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${polyp.size_min_mm || '-'} - ${polyp.size_max_mm || '-'}</td>
                        <td>${polyp.location || '-'}</td>
                        <td>${polyp.resection_performed ? 'Yes' : 'No'}</td>
                        <td>${polyp.resection_method || '-'}</td>
                        <td>${polyp.nice_class || '-'}</td>
                        <td>${polyp.paris_class || '-'}</td>
                    </tr>
                `;
            });

            html += '</tbody></table>';
            return html;
        }
    }));
});
