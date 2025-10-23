/**
 * EndoScribe Web App - Frontend JavaScript
 * Handles audio recording, WebSocket communication, and UI updates
 */

// Application state
const state = {
    recording: false,
    paused: false,
    mediaRecorder: null,
    websocket: null,
    sessionId: null,
    fullTranscript: '',
    audioChunks: []
};

// DOM elements
const elements = {
    startBtn: document.getElementById('start-recording'),
    pauseBtn: document.getElementById('pause-recording'),
    resumeBtn: document.getElementById('resume-recording'),
    stopBtn: document.getElementById('stop-recording'),
    submitBtn: document.getElementById('submit-transcript'),
    recordingIndicator: document.getElementById('recording-indicator'),
    recordingStatus: document.getElementById('recording-status'),
    sessionInfo: document.getElementById('session-info'),
    sessionIdEl: document.getElementById('session-id'),
    transcriptionText: document.getElementById('transcription-text'),
    transcriptionContainer: document.getElementById('transcription-container'),
    processingStatus: document.getElementById('processing-status'),
    resultsContainer: document.getElementById('results-container'),
    errorContainer: document.getElementById('error-container'),
    errorMessage: document.getElementById('error-message'),
    procedureType: document.getElementById('procedure-type')
};

// Utility functions
function showElement(el) {
    el?.classList.remove('hidden');
}

function hideElement(el) {
    el?.classList.add('hidden');
}

function showError(message) {
    elements.errorMessage.textContent = message;
    showElement(elements.errorContainer);
    setTimeout(() => hideElement(elements.errorContainer), 5000);
}

function updateTranscript(text, append = true) {
    if (append) {
        state.fullTranscript += ' ' + text;
    } else {
        state.fullTranscript = text;
    }
    elements.transcriptionText.textContent = state.fullTranscript.trim();
    // Auto-scroll to bottom
    elements.transcriptionContainer.scrollTop = elements.transcriptionContainer.scrollHeight;
}

// WebSocket functions
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/transcribe`;

    console.log('Connecting to WebSocket:', wsUrl);
    state.websocket = new WebSocket(wsUrl);

    state.websocket.onopen = () => {
        console.log('WebSocket connected');
        // Send start message
        state.websocket.send(JSON.stringify({
            type: 'start',
            session_id: state.sessionId
        }));
    };

    state.websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            console.log('WebSocket message:', message);

            switch (message.type) {
                case 'status':
                    if (message.session_id) {
                        state.sessionId = message.session_id;
                        elements.sessionIdEl.textContent = state.sessionId;
                        showElement(elements.sessionInfo);
                    }
                    // Update recording status with processing message
                    if (message.message && message.message.includes('Processing')) {
                        elements.recordingStatus.textContent = message.message;
                        elements.recordingStatus.className = 'badge badge-info';
                    }
                    break;

                case 'transcript':
                    const transcriptText = message.data?.text || '';
                    if (transcriptText) {
                        updateTranscript(transcriptText, true);
                        // Reset status back to recording after transcript received
                        if (state.recording && !state.paused) {
                            elements.recordingStatus.textContent = 'Recording';
                            elements.recordingStatus.className = 'badge badge-error';
                        }
                    }
                    break;

                case 'error':
                    showError(message.message || 'WebSocket error occurred');
                    break;
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    state.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('WebSocket connection error - server may be processing audio');
    };

    state.websocket.onclose = (event) => {
        console.log('WebSocket closed', event.code, event.reason);
        if (event.code !== 1000 && event.code !== 1001 && state.recording) {
            // Abnormal closure while recording
            showError(`Connection closed unexpectedly (code: ${event.code}). Try again or reduce audio chunk length.`);
        }
    };
}

function closeWebSocket() {
    if (state.websocket) {
        state.websocket.send(JSON.stringify({ type: 'end' }));
        state.websocket.close();
        state.websocket = null;
    }
}

// MediaRecorder functions
async function startRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Create MediaRecorder
        const options = { mimeType: 'audio/webm' };
        state.mediaRecorder = new MediaRecorder(stream, options);

        // Connect WebSocket
        connectWebSocket();

        // Handle dataavailable event (audio chunks)
        state.mediaRecorder.ondataavailable = async (event) => {
            if (event.data.size > 0 && state.websocket?.readyState === WebSocket.OPEN) {
                console.log('Sending audio chunk:', event.data.size, 'bytes');
                // Send binary audio data via WebSocket
                state.websocket.send(event.data);
            }
        };

        state.mediaRecorder.onstop = () => {
            console.log('MediaRecorder stopped');
            stream.getTracks().forEach(track => track.stop());
        };

        // Start recording with chunks every 3 seconds
        state.mediaRecorder.start(3000);
        state.recording = true;
        state.paused = false;

        // Update UI
        updateUI();
        updateTranscript('', false); // Clear previous transcript
        elements.recordingStatus.textContent = 'Recording';
        elements.recordingStatus.className = 'badge badge-error';
        showElement(elements.recordingStatus);

    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Failed to start recording. Please check microphone permissions.');
    }
}

function pauseRecording() {
    if (state.mediaRecorder && state.recording) {
        state.mediaRecorder.pause();
        state.paused = true;
        updateUI();
        elements.recordingStatus.textContent = 'Paused';
        elements.recordingStatus.className = 'badge badge-warning';
    }
}

function resumeRecording() {
    if (state.mediaRecorder && state.paused) {
        state.mediaRecorder.resume();
        state.paused = false;
        updateUI();
        elements.recordingStatus.textContent = 'Recording';
        elements.recordingStatus.className = 'badge badge-error';
    }
}

function stopRecording() {
    if (state.mediaRecorder) {
        state.mediaRecorder.stop();
        state.recording = false;
        state.paused = false;

        // Close WebSocket
        closeWebSocket();

        // Update UI
        updateUI();
        elements.recordingStatus.textContent = 'Stopped';
        elements.recordingStatus.className = 'badge badge-neutral';

        // Show submit button
        showElement(elements.submitBtn);
    }
}

// UI update function
function updateUI() {
    if (state.recording) {
        hideElement(elements.startBtn);
        showElement(elements.recordingIndicator);

        if (state.paused) {
            hideElement(elements.pauseBtn);
            showElement(elements.resumeBtn);
            showElement(elements.stopBtn);
        } else {
            showElement(elements.pauseBtn);
            hideElement(elements.resumeBtn);
            showElement(elements.stopBtn);
        }
    } else {
        showElement(elements.startBtn);
        hideElement(elements.pauseBtn);
        hideElement(elements.resumeBtn);
        hideElement(elements.stopBtn);
        hideElement(elements.recordingIndicator);
    }
}

// Process transcript function
async function processTranscript() {
    const transcript = state.fullTranscript.trim();
    const procedureType = elements.procedureType.value;

    if (!transcript) {
        showError('No transcript available to process');
        return;
    }

    // Show processing status
    showElement(elements.processingStatus);
    hideElement(elements.submitBtn);
    hideElement(elements.resultsContainer);

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transcript: transcript,
                procedure_type: procedureType,
                session_id: state.sessionId
            })
        });

        const result = await response.json();

        // Hide processing status
        hideElement(elements.processingStatus);

        if (result.success) {
            displayResults(result.data, result.procedure_type);
        } else {
            showError(result.error || 'Processing failed');
        }

    } catch (error) {
        console.error('Error processing transcript:', error);
        hideElement(elements.processingStatus);
        showError('Failed to process transcript: ' + error.message);
    }
}

// Display results function
function displayResults(data, procedureType) {
    showElement(elements.resultsContainer);

    if (procedureType === 'col') {
        // Display colonoscopy results
        displayColonoscopyResults(data);
    } else {
        // Display other procedure results
        displayOtherResults(data, procedureType);
    }
}

function displayColonoscopyResults(data) {
    showElement(document.getElementById('col-results'));
    hideElement(document.getElementById('other-results'));

    // Display colonoscopy data
    const colonoscopyData = data.colonoscopy || {};
    const colonoscopyHTML = createDataTable(colonoscopyData);
    document.getElementById('colonoscopy-data').innerHTML = colonoscopyHTML;

    // Display polyps data
    const polypsData = data.polyps || [];
    const polypsHTML = createPolypsTable(polypsData);
    document.getElementById('polyps-data').innerHTML = polypsHTML;
}

function displayOtherResults(data, procedureType) {
    hideElement(document.getElementById('col-results'));
    showElement(document.getElementById('other-results'));

    const dataHTML = createDataTable(data);
    document.getElementById('procedure-data').innerHTML = dataHTML;
}

function createDataTable(data) {
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
}

function createPolypsTable(polyps) {
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

// Event listeners
elements.startBtn.addEventListener('click', startRecording);
elements.pauseBtn.addEventListener('click', pauseRecording);
elements.resumeBtn.addEventListener('click', resumeRecording);
elements.stopBtn.addEventListener('click', stopRecording);
elements.submitBtn.addEventListener('click', processTranscript);

// Initialize UI
updateUI();
console.log('EndoScribe app initialized');
