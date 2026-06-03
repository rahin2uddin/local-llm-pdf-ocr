// --------------------------------------------------------------------------
// Unified State & API Store - DocuVerse AI Workstation
// --------------------------------------------------------------------------

// 1. Unified State Store
const state = {
    settings: {},
    selectedFile: null,
    currentJobId: null,
    currentJobToken: null,
    progressChannelId: null,
    progressSessionToken: null,
    processingStartTime: null,
    elapsedInterval: null,
    resultBlob: null,
    resultFilename: null,
    rawTextResult: null,
    translatedText: "",
    extractedJson: null
};

// Workspace rendering state
const workspaceState = {
    activeFile: null,         // Active PDF/Image File or Blob object
    activeFileName: '',       // Active filename string
    totalPages: 0,            // Total pages in active document
    currentPageIdx: 0,        // Active page viewed (0-indexed)
    zoomLevel: 1.0,           // Render scaling multiplier
    extractedText: {},        // Extracted OCR text per page: { pageIndex: lines[] }
    isProcessing: false,      // Active OCR processing flag
    pdfDoc: null,             // PDF.js document handle
    layoutBboxes: null,       // Bounding boxes per page
    viewMode: 'single'        // Only single mode supported now
};

// 2. DOM Elements References
const refs = {
    // Layout containers
    workspaceDropZone: document.getElementById('workspace-drop-zone'),
    workspaceViewport: document.getElementById('workspace-viewport'),
    workspaceCanvas: document.getElementById('workspace-canvas'),
    workspaceBboxSvg: document.getElementById('workspace-bbox-svg'),
    workspaceTextLayer: document.getElementById('workspace-text-layer'),
    workspaceViewportWelcome: document.getElementById('workspace-viewport-welcome'),
    workspaceCanvasContainer: document.getElementById('workspace-canvas-container'),
    
    // Top workspace header controls
    ribbonCurrentPage: document.getElementById('ribbon-current-page'),
    ribbonTotalPages: document.getElementById('ribbon-total-pages'),
    ribbonPrevPage: document.getElementById('ribbon-prev-page'),
    ribbonNextPage: document.getElementById('ribbon-next-page'),
    ribbonZoomLabel: document.getElementById('ribbon-zoom-label'),
    ribbonZoomIn: document.getElementById('ribbon-zoom-in'),
    ribbonZoomOut: document.getElementById('ribbon-zoom-out'),
    ribbonFitWidth: document.getElementById('ribbon-fit-width'),
    
    // Theme & Navigation
    themeBtn: document.getElementById('theme-btn'),
    moonIcon: document.getElementById('moon-icon'),
    sunIcon: document.getElementById('sun-icon'),
    connStatus: document.getElementById('connection-status'),
    
    // OCR Engine configuration panel (Left Sidebar)
    apiBase: document.getElementById('setting-api-base'),
    apiKey: document.getElementById('setting-api-key'),
    modelSelect: document.getElementById('setting-model'),
    refreshModelsBtn: document.getElementById('refresh-models-btn'),
    pipelineModes: document.getElementsByName('mode'),
    dpi: document.getElementById('setting-dpi'),
    dpiVal: document.getElementById('dpi-val'),
    concurrency: document.getElementById('setting-concurrency'),
    concurrencyVal: document.getElementById('concurrency-val'),
    denseMode: document.getElementById('setting-dense-mode'),
    maxImageDim: document.getElementById('setting-max-image-dim'),
    maxImageDimVal: document.getElementById('max-image-dim-val'),
    denseThreshold: document.getElementById('setting-dense-threshold'),
    denseThresholdVal: document.getElementById('dense-threshold-val'),
    refine: document.getElementById('setting-refine'),
    selfCorrection: document.getElementById('setting-self-correction'),
    binarize: document.getElementById('setting-binarize'),
    dualEngine: document.getElementById('setting-dual-engine'),
    spellcheck: document.getElementById('setting-spellcheck'),
    crossPage: document.getElementById('setting-cross-page'),
    pages: document.getElementById('setting-pages'),
    
    // Upload slots & active file card
    fileInput: document.getElementById('file-input'),
    filePreview: document.getElementById('file-preview'),
    fileName: document.getElementById('file-name-display'),
    fileSize: document.getElementById('file-size-display'),
    clearFileBtn: document.getElementById('clear-file-btn'),
    startBtn: document.getElementById('start-btn'),
    
    // Progress Screen Overlay
    processView: document.getElementById('process-view'),
    progressBar: document.getElementById('progress-bar'),
    statusText: document.getElementById('status-text'),
    subStatus: document.getElementById('sub-status'),
    elapsedTime: document.getElementById('elapsed-time'),
    cancelBtn: document.getElementById('cancel-btn'),
    
    stageConvert: document.getElementById('stage-convert'),
    stageDetect: document.getElementById('stage-detect'),
    stageOcr: document.getElementById('stage-ocr'),
    stageRefine: document.getElementById('stage-refine'),
    stageEmbed: document.getElementById('stage-embed'),

    // AI Results sidebar (Right Sidebar Tabs & Panes)
    tabBtns: document.querySelectorAll('.ai-tab-btn'),
    tabPanels: document.querySelectorAll('.ai-tab-panel'),
    mdContent: document.getElementById('md-content'),
    textContent: document.getElementById('text-content'),
    copyTextBtn: document.getElementById('copy-text-btn'),
    dlTxtBtn: document.getElementById('dl-txt-btn'),
    copyMdBtn: document.getElementById('copy-md-btn'),
    dlMdBtn: document.getElementById('dl-md-btn'),
    
    // Translation features
    translateLangSelect: document.getElementById('translate-lang-select'),
    translateBtn: document.getElementById('translate-btn'),
    translatedMarkdownContent: document.getElementById('translated-markdown-content'),
    copyTransBtn: document.getElementById('copy-trans-btn'),
    dlTransBtn: document.getElementById('dl-trans-btn'),
    
    // Structured Data extraction features
    extractorTemplateSelect: document.getElementById('extractor-template-select'),
    extractorCustomPromptContainer: document.getElementById('extractor-custom-prompt-container'),
    extractorCustomPrompt: document.getElementById('extractor-custom-prompt'),
    extractBtn: document.getElementById('extract-btn'),
    extractedJsonVisualCards: document.getElementById('extracted-json-visual-cards'),
    extractedJsonRaw: document.getElementById('extracted-json-raw'),
    copyJsonBtn: document.getElementById('copy-json-btn'),
    dlJsonBtn: document.getElementById('dl-json-btn'),
    
    // Global Server Connection modal
    btnOpenSettingsModal: document.getElementById('btn-open-settings-modal'),
    settingsModal: document.getElementById('settings-modal'),
    settingsModalClose: document.getElementById('settings-modal-close'),
    settingsModalCancel: document.getElementById('settings-modal-cancel'),
    settingsModalSave: document.getElementById('settings-modal-save'),
    connectionStatusDot: document.getElementById('connection-status-dot'),
    
    toastContainer: document.getElementById('toast-container')
};

// 3. Settings Configuration Module
async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        if (!res.ok) throw new Error('Failed to load config');
        const config = await res.json();
        state.settings = config;
        
        if (refs.apiBase) refs.apiBase.value = config.api_base || '';
        if (refs.apiKey) refs.apiKey.value = config.api_key || '';
        if (refs.dpi) {
            refs.dpi.value = config.dpi || 200;
            refs.dpiVal.textContent = refs.dpi.value;
        }
        if (refs.concurrency) {
            refs.concurrency.value = config.concurrency || 3;
            refs.concurrencyVal.textContent = refs.concurrency.value;
        }
        if (refs.denseMode) refs.denseMode.value = config.dense_mode || 'auto';
        if (refs.maxImageDim) {
            refs.maxImageDim.value = config.max_image_dim || 1024;
            refs.maxImageDimVal.textContent = refs.maxImageDim.value;
        }
        if (refs.denseThreshold) {
            refs.denseThreshold.value = config.dense_threshold || 60;
            refs.denseThresholdVal.textContent = refs.denseThreshold.value;
        }
        if (refs.refine) refs.refine.checked = config.refine !== false;
        if (refs.selfCorrection) refs.selfCorrection.checked = config.self_correction === true;
        if (refs.binarize) refs.binarize.checked = config.binarize === true;
        if (refs.dualEngine) refs.dualEngine.checked = config.dual_engine === true;
        if (refs.spellcheck) refs.spellcheck.value = config.spellcheck || 'none';
        if (refs.crossPage) refs.crossPage.checked = config.cross_page === true;
        
        if (refs.pipelineModes) {
            for (const radio of refs.pipelineModes) {
                if (radio.value === config.pipeline_mode) {
                    radio.checked = true;
                }
            }
        }
        
        await fetchModels(config.model);
    } catch (e) {
        showToast('Error loading configuration', 'error');
    }
}

async function saveConfig() {
    const activePipelineMode = document.querySelector('input[name="mode"]:checked')?.value || 'hybrid';
    const data = {
        api_base: refs.apiBase ? refs.apiBase.value : 'http://localhost:1234/v1',
        api_key: refs.apiKey ? refs.apiKey.value : 'lm-studio',
        model: refs.modelSelect ? refs.modelSelect.value : '',
        pipeline_mode: activePipelineMode,
        dpi: refs.dpi ? parseInt(refs.dpi.value) : 200,
        concurrency: refs.concurrency ? parseInt(refs.concurrency.value) : 3,
        dense_mode: refs.denseMode ? refs.denseMode.value : 'auto',
        max_image_dim: refs.maxImageDim ? parseInt(refs.maxImageDim.value, 10) : 1024,
        dense_threshold: refs.denseThreshold ? parseInt(refs.denseThreshold.value, 10) : 60,
        refine: refs.refine ? refs.refine.checked : true,
        self_correction: refs.selfCorrection ? refs.selfCorrection.checked : false,
        binarize: refs.binarize ? refs.binarize.checked : false,
        dual_engine: refs.dualEngine ? refs.dualEngine.checked : false,
        spellcheck: refs.spellcheck ? refs.spellcheck.value : 'none',
        cross_page: refs.crossPage ? refs.crossPage.checked : false
    };
    
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
    } catch (e) {
        console.error('Failed to save config:', e);
    }
}

let saveTimeout;
function debounceSave() {
    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(saveConfig, 500);
}

// Auto-save listeners
[refs.apiBase, refs.apiKey, refs.modelSelect, refs.dpi, refs.concurrency, refs.denseMode, refs.maxImageDim, refs.denseThreshold, refs.refine, refs.selfCorrection, refs.binarize, refs.dualEngine, refs.spellcheck, refs.crossPage].forEach(el => {
    if (!el) return;
    el.addEventListener('change', debounceSave);
    el.addEventListener('input', debounceSave);
});
refs.pipelineModes?.forEach(r => r.addEventListener('change', debounceSave));

// Range labels
refs.dpi?.addEventListener('input', (e) => { if(refs.dpiVal) refs.dpiVal.textContent = e.target.value; });
refs.concurrency?.addEventListener('input', (e) => { if(refs.concurrencyVal) refs.concurrencyVal.textContent = e.target.value; });
refs.maxImageDim?.addEventListener('input', (e) => { if(refs.maxImageDimVal) refs.maxImageDimVal.textContent = e.target.value; });
refs.denseThreshold?.addEventListener('input', (e) => { if(refs.denseThresholdVal) refs.denseThresholdVal.textContent = e.target.value; });

async function fetchModels(selectedModel = null) {
    if(refs.refreshModelsBtn) refs.refreshModelsBtn.disabled = true;
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        
        if (data.error) {
            if(refs.connStatus) {
                refs.connStatus.className = 'status-dot offline';
                refs.connStatus.title = `Error: ${data.error}`;
            }
            if(refs.connectionStatusDot) refs.connectionStatusDot.className = 'status-dot offline';
            showToast(`Model fetch error: ${data.error}`, 'error');
        } else {
            if(refs.connStatus) {
                refs.connStatus.className = 'status-dot online';
                refs.connStatus.title = 'Connected';
            }
            if(refs.connectionStatusDot) refs.connectionStatusDot.className = 'status-dot online';
            
            if(refs.modelSelect) {
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = 'Select a model...';
                refs.modelSelect.replaceChildren(placeholder);
                data.models.forEach(modelId => {
                    const opt = document.createElement('option');
                    opt.value = modelId;
                    opt.textContent = modelId;
                    refs.modelSelect.appendChild(opt);
                });
                
                if (selectedModel) {
                    if (!data.models.includes(selectedModel)) {
                        const opt = document.createElement('option');
                        opt.value = selectedModel;
                        opt.textContent = selectedModel + ' (Custom)';
                        refs.modelSelect.appendChild(opt);
                    }
                    refs.modelSelect.value = selectedModel;
                } else if (state.settings.model) {
                    refs.modelSelect.value = state.settings.model;
                }
            }
        }
    } catch (e) {
        if(refs.connStatus) refs.connStatus.className = 'status-dot offline';
        if(refs.connectionStatusDot) refs.connectionStatusDot.className = 'status-dot offline';
        showToast('Failed to fetch models', 'error');
    } finally {
        if(refs.refreshModelsBtn) refs.refreshModelsBtn.disabled = false;
    }
}

if(refs.refreshModelsBtn && refs.modelSelect) {
    refs.refreshModelsBtn.addEventListener('click', () => fetchModels(refs.modelSelect.value));
}

function getFormSettings() {
    const activePipelineMode = document.querySelector('input[name="mode"]:checked')?.value || 'hybrid';
    return {
        api_base: refs.apiBase ? refs.apiBase.value : '',
        api_key: refs.apiKey ? refs.apiKey.value : '',
        model: refs.modelSelect ? refs.modelSelect.value : '',
        pipeline_mode: activePipelineMode,
        dpi: refs.dpi ? refs.dpi.value : '200',
        concurrency: refs.concurrency ? refs.concurrency.value : '3',
        dense_mode: refs.denseMode ? refs.denseMode.value : 'auto',
        max_image_dim: refs.maxImageDim ? refs.maxImageDim.value : '1024',
        dense_threshold: refs.denseThreshold ? refs.denseThreshold.value : '60',
        refine: refs.refine ? (refs.refine.checked ? 'true' : 'false') : 'true',
        self_correction: refs.selfCorrection ? (refs.selfCorrection.checked ? 'true' : 'false') : 'false',
        binarize: refs.binarize ? (refs.binarize.checked ? 'true' : 'false') : 'false',
        dual_engine: refs.dualEngine ? (refs.dualEngine.checked ? 'true' : 'false') : 'false',
        spellcheck: refs.spellcheck ? refs.spellcheck.value : 'none',
        cross_page: refs.crossPage ? (refs.crossPage.checked ? 'true' : 'false') : 'false',
        pages: refs.pages ? refs.pages.value.trim() : ''
    };
}

// 4. File Helper Module
const ALLOWED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif'];

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function handleFile(file) {
    if (!file) return;
    
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
        showToast(`Unsupported file type: ${ext}`, 'error');
        return;
    }
    
    state.selectedFile = file;
    if(refs.fileName) refs.fileName.textContent = file.name;
    if(refs.fileSize) refs.fileSize.textContent = formatSize(file.size);
    
    if(refs.workspaceDropZone) refs.workspaceDropZone.classList.add('hidden');
    if(refs.filePreview) refs.filePreview.classList.remove('hidden');
    if(refs.startBtn) refs.startBtn.disabled = false;
    
    // Trigger Visual workspace document load
    if (typeof loadWorkspaceDocument === 'function') {
        loadWorkspaceDocument(file);
    }
}

// 5. New AI Workstation API Methods
async function translateText(text, targetLang) {
    if (!text.trim()) return "";
    
    const settings = getFormSettings();
    const body = {
        text: text,
        target_language: targetLang,
        api_base: settings.api_base,
        api_key: settings.api_key,
        model: settings.model
    };
    
    const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Translation request failed');
    }
    
    const data = await res.json();
    return data.translated_text || "";
}

async function extractData(text, template, customPrompt = "") {
    if (!text.trim()) return {};
    
    const settings = getFormSettings();
    const body = {
        text: text,
        template: template,
        custom_prompt: customPrompt,
        api_base: settings.api_base,
        api_key: settings.api_key,
        model: settings.model
    };
    
    const res = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Data extraction request failed');
    }
    
    const data = await res.json();
    return data.extracted_data || {};
}
