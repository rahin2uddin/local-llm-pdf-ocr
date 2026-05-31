// --------------------------------------------------------------------------
// Unified Visual Document Viewport & UI Notifications
// --------------------------------------------------------------------------

// 1. Load active document into the canvas visualizer
async function loadWorkspaceDocument(file) {
    workspaceState.activeFile = file;
    workspaceState.activeFileName = file.name;
    
    // Hide welcome overlay, show canvas viewport, and show header toolbar
    if (refs.workspaceViewportWelcome) refs.workspaceViewportWelcome.classList.add('hidden');
    if (refs.workspaceViewport) refs.workspaceViewport.classList.remove('hidden');
    if (refs.headerWkToolbar) refs.headerWkToolbar.classList.remove('hidden');
    if (refs.filePreview) refs.filePreview.classList.remove('hidden');

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (ext === '.pdf') {
        try {
            await loadPdfJs();
            
            const fileReader = new FileReader();
            fileReader.onload = async function() {
                const typedarray = new Uint8Array(this.result);
                try {
                    const pdfDoc = await window.pdfjsLib.getDocument({ data: typedarray }).promise;
                    workspaceState.pdfDoc = pdfDoc;
                    workspaceState.totalPages = pdfDoc.numPages;
                    workspaceState.currentPageIdx = 0;
                    
                    if (refs.ribbonTotalPages) refs.ribbonTotalPages.textContent = pdfDoc.numPages;
                    if (refs.ribbonCurrentPage) refs.ribbonCurrentPage.textContent = 1;
                    
                    // Render page 1
                    await renderWorkspacePage(1);
                    showToast('PDF loaded successfully!', 'success');
                } catch (err) {
                    showToast('Failed to parse PDF document: ' + err.message, 'error');
                }
            };
            fileReader.readAsArrayBuffer(file);
        } catch (err) {
            showToast(err.message, 'error');
        }
    } else {
        // Image rendering
        workspaceState.pdfDoc = null;
        workspaceState.totalPages = 1;
        workspaceState.currentPageIdx = 0;
        
        if (refs.ribbonTotalPages) refs.ribbonTotalPages.textContent = '1';
        if (refs.ribbonCurrentPage) refs.ribbonCurrentPage.textContent = 1;
        
        await renderWorkspaceImage(file);
        showToast('Image loaded successfully!', 'success');
    }
}

// 2. Render standard PDF page on canvas
async function renderWorkspacePage(pageNum) {
    if (!workspaceState.pdfDoc) return;
    if (pageNum < 1 || pageNum > workspaceState.totalPages) return;

    workspaceState.currentPageIdx = pageNum - 1;
    if (refs.ribbonCurrentPage) refs.ribbonCurrentPage.textContent = pageNum;

    try {
        const page = await workspaceState.pdfDoc.getPage(pageNum);
        const canvas = refs.workspaceCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const viewport = page.getViewport({ scale: workspaceState.zoomLevel });
        canvas.width = viewport.width;
        canvas.height = viewport.height;

        const renderContext = {
            canvasContext: ctx,
            viewport: viewport
        };

        await page.render(renderContext).promise;

        // Bounding boxes rendering
        drawLayoutBboxes(workspaceState.currentPageIdx, viewport);

        // Render transparent selectable text layer for searchability
        if (refs.workspaceTextLayer) {
            refs.workspaceTextLayer.innerHTML = '';
            refs.workspaceTextLayer.style.width = `${viewport.width}px`;
            refs.workspaceTextLayer.style.height = `${viewport.height}px`;

            try {
                const textContent = await page.getTextContent();
                window.pdfjsLib.renderTextLayer({
                    textContent: textContent,
                    container: refs.workspaceTextLayer,
                    viewport: viewport,
                    textDivs: []
                });
            } catch (err) {
                console.error('Failed to render text layer:', err);
            }
        }
    } catch (err) {
        console.error('Error rendering page:', err);
    }
}

// 3. Render image file onto canvas
async function renderWorkspaceImage(file) {
    const canvas = refs.workspaceCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = () => {
        const width = img.width * workspaceState.zoomLevel;
        const height = img.height * workspaceState.zoomLevel;
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);
        
        if (refs.workspaceTextLayer) refs.workspaceTextLayer.innerHTML = '';
        if (refs.workspaceBboxSvg) refs.workspaceBboxSvg.innerHTML = '';
    };
    img.src = URL.createObjectURL(file);
}

// 4. Draw layout bounding boxes overlay
function drawLayoutBboxes(pageIdx, viewport) {
    if (!refs.workspaceBboxSvg) return;
    refs.workspaceBboxSvg.innerHTML = '';
    
    const bboxes = workspaceState.layoutBboxes ? workspaceState.layoutBboxes[pageIdx] : null;
    if (!bboxes || !Array.isArray(bboxes)) return;
    
    bboxes.forEach(box => {
        const [nx0, ny0, nx1, ny1] = box;
        const x = nx0 * viewport.width;
        const y = ny0 * viewport.height;
        const width = (nx1 - nx0) * viewport.width;
        const height = (ny1 - ny0) * viewport.height;
        
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('fill', 'rgba(139, 92, 246, 0.12)');
        rect.setAttribute('stroke', 'var(--primary)');
        rect.setAttribute('stroke-width', '1.5');
        
        refs.workspaceBboxSvg.appendChild(rect);
    });
}

// 5. Dynamic script loader for PDF.js
function loadPdfJs() {
    return new Promise((resolve, reject) => {
        if (window.pdfjsLib) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js';
        script.onload = () => {
            window.pdfjsLib = window.pdfjsLib || window['pdfjs-dist/build/pdf'];
            window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';
            resolve();
        };
        script.onerror = () => reject(new Error('Failed to load PDF.js library'));
        document.head.appendChild(script);
    });
}

// 6. Navigation ribbon events wiring
refs.ribbonPrevPage?.addEventListener('click', () => {
    if (workspaceState.currentPageIdx > 0) {
        renderWorkspacePage(workspaceState.currentPageIdx);
    }
});

refs.ribbonNextPage?.addEventListener('click', () => {
    if (workspaceState.currentPageIdx < workspaceState.totalPages - 1) {
        renderWorkspacePage(workspaceState.currentPageIdx + 2);
    }
});

refs.ribbonZoomIn?.addEventListener('click', () => {
    if (workspaceState.zoomLevel < 3.0) {
        workspaceState.zoomLevel = Math.round((workspaceState.zoomLevel + 0.1) * 10) / 10;
        if(refs.ribbonZoomLabel) refs.ribbonZoomLabel.textContent = `${Math.round(workspaceState.zoomLevel * 100)}%`;
        if (workspaceState.pdfDoc) {
            renderWorkspacePage(workspaceState.currentPageIdx + 1);
        } else if (workspaceState.activeFile) {
            renderWorkspaceImage(workspaceState.activeFile);
        }
    }
});

refs.ribbonZoomOut?.addEventListener('click', () => {
    if (workspaceState.zoomLevel > 0.5) {
        workspaceState.zoomLevel = Math.round((workspaceState.zoomLevel - 0.1) * 10) / 10;
        if(refs.ribbonZoomLabel) refs.ribbonZoomLabel.textContent = `${Math.round(workspaceState.zoomLevel * 100)}%`;
        if (workspaceState.pdfDoc) {
            renderWorkspacePage(workspaceState.currentPageIdx + 1);
        } else if (workspaceState.activeFile) {
            renderWorkspaceImage(workspaceState.activeFile);
        }
    }
});

refs.ribbonFitWidth?.addEventListener('click', () => {
    // Basic automatic fit-to-width toggle
    workspaceState.zoomLevel = 1.0;
    if(refs.ribbonZoomLabel) refs.ribbonZoomLabel.textContent = '100%';
    if (workspaceState.pdfDoc) {
        renderWorkspacePage(workspaceState.currentPageIdx + 1);
    } else if (workspaceState.activeFile) {
        renderWorkspaceImage(workspaceState.activeFile);
    }
});

// Clear active document completely
refs.clearFileBtn?.addEventListener('click', () => {
    state.selectedFile = null;
    workspaceState.activeFile = null;
    workspaceState.activeFileName = '';
    workspaceState.totalPages = 0;
    workspaceState.currentPageIdx = 0;
    workspaceState.pdfDoc = null;
    workspaceState.layoutBboxes = null;

    if (refs.fileInput) refs.fileInput.value = '';
    if (refs.filePreview) refs.filePreview.classList.add('hidden');
    if (refs.workspaceDropZone) refs.workspaceDropZone.classList.remove('hidden');
    if (refs.workspaceViewport) refs.workspaceViewport.classList.add('hidden');
    if (refs.workspaceViewportWelcome) refs.workspaceViewportWelcome.classList.remove('hidden');
    if (refs.headerWkToolbar) refs.headerWkToolbar.classList.add('hidden');
    if (refs.startBtn) refs.startBtn.disabled = true;

    // Reset AI panels
    if (refs.mdContent) refs.mdContent.value = '';
    if (refs.textContent) refs.textContent.value = '';
    if (refs.translatedMarkdownContent) refs.translatedMarkdownContent.value = '';
    if (refs.extractedJsonRaw) refs.extractedJsonRaw.value = '';
    if (refs.extractedJsonVisualCards) {
        refs.extractedJsonVisualCards.innerHTML = '<div style="font-size:0.75rem; color:var(--text-muted); text-align:center; padding:1rem;">No data extracted yet.</div>';
    }
});

// 7. Visual UI Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Style toast with curating premium HSL rules
    let accentColor = 'var(--primary)';
    if (type === 'success') accentColor = 'var(--success)';
    if (type === 'error') accentColor = 'var(--error)';
    
    toast.style.cssText = `
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid var(--border);
        border-left: 4px solid ${accentColor};
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        box-shadow: var(--shadow-lg);
        color: var(--text-main);
        font-size: 0.8rem;
        min-width: 250px;
        max-width: 380px;
        backdrop-filter: blur(8px);
        transform: translateY(20px);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    `;
    
    toast.textContent = message;
    
    const container = refs.toastContainer || document.getElementById('toast-container');
    if (container) {
        container.appendChild(toast);
        // Trigger show animation
        setTimeout(() => {
            toast.style.transform = 'translateY(0)';
            toast.style.opacity = '1';
        }, 10);
        
        // Auto remove
        setTimeout(() => {
            toast.style.transform = 'translateY(-20px)';
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    } else {
        console.log(`[Toast ${type}]: ${message}`);
    }
}
