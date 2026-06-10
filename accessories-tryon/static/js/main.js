/**
 * main.js — Frontend logic for Accessories Virtual Try-On
 *
 * Handles: catalogue loading, tab switching, accessory selection,
 * file upload with validation, snapshot capture, face shape analysis,
 * and toast notifications.
 */

// ── State ────────────────────────────────────────────────────────────
let currentCategory = 'glasses';
let selectedCard = null;
let catalogue = {};
let uploadedFile = null;

// ── DOM References ───────────────────────────────────────────────────
const grid = document.getElementById('accessory-grid');
const tabs = document.querySelectorAll('.tab');
const btnSnapshot = document.getElementById('btn-snapshot');
const btnClear = document.getElementById('btn-clear');
const btnFaceShape = document.getElementById('btn-face-shape');
const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('upload-dropzone');
const uploadPreview = document.getElementById('upload-preview');
const uploadPreviewImg = document.getElementById('upload-preview-img');
const btnApplyUpload = document.getElementById('btn-apply-upload');

// Modals
const snapshotModal = document.getElementById('snapshot-modal');
const snapshotImg = document.getElementById('snapshot-img');
const snapshotDownload = document.getElementById('snapshot-download');
const faceShapeModal = document.getElementById('face-shape-modal');
const faceShapeResult = document.getElementById('face-shape-result');

// ── Toast Notification System ────────────────────────────────────────
let toastEl = null;
let toastTimeout = null;

function showToast(message, type = 'success') {
    if (!toastEl) {
        toastEl = document.createElement('div');
        toastEl.className = 'toast';
        document.body.appendChild(toastEl);
    }

    clearTimeout(toastTimeout);
    toastEl.textContent = message;
    toastEl.className = `toast ${type}`;

    // Trigger reflow for animation restart
    void toastEl.offsetWidth;
    toastEl.classList.add('show');

    toastTimeout = setTimeout(() => {
        toastEl.classList.remove('show');
    }, 3000);
}

// ── Load Accessories Catalogue ───────────────────────────────────────
async function loadCatalogue() {
    try {
        const res = await fetch('/accessories');
        catalogue = await res.json();
        renderGrid(currentCategory);
    } catch (err) {
        console.error('Failed to load catalogue:', err);
        grid.innerHTML = '<div class="empty-state"><p>Failed to load accessories</p></div>';
    }
}

// ── Render Accessory Grid ────────────────────────────────────────────
function renderGrid(category) {
    const items = catalogue[category] || [];

    if (items.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                </svg>
                <p>No ${category} available yet</p>
                <p>Upload your own below!</p>
            </div>`;
        return;
    }

    grid.innerHTML = items.map(filename => {
        const name = filename.replace('.png', '').replace(/_/g, ' ');
        const thumbUrl = `/static/accessories/${category}/${filename}`;
        return `
            <div class="accessory-card" 
                 data-category="${category}" 
                 data-filename="${filename}"
                 id="card-${filename.replace('.png', '')}"
                 title="${name}">
                <img src="${thumbUrl}" alt="${name}" loading="lazy">
                <div class="accessory-card-name">${name}</div>
            </div>`;
    }).join('');

    // Attach click handlers
    grid.querySelectorAll('.accessory-card').forEach(card => {
        card.addEventListener('click', () => selectAccessory(card));
    });
}

// ── Select Accessory ─────────────────────────────────────────────────
async function selectAccessory(card) {
    const category = card.dataset.category;
    const filename = card.dataset.filename;

    // Visual selection
    if (selectedCard) selectedCard.classList.remove('selected');
    card.classList.add('selected');
    selectedCard = card;

    try {
        const res = await fetch('/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category, filename })
        });

        const data = await res.json();
        if (res.ok) {
            showToast(`Applied ${filename.replace('.png', '').replace(/_/g, ' ')}`, 'success');
        } else {
            showToast(data.error || 'Failed to apply', 'error');
        }
    } catch (err) {
        showToast('Network error', 'error');
    }
}

// ── Tab Switching ────────────────────────────────────────────────────
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentCategory = tab.dataset.category;

        // Clear selection when switching tabs
        if (selectedCard) {
            selectedCard.classList.remove('selected');
            selectedCard = null;
        }

        renderGrid(currentCategory);
    });
});

// ── Clear Accessory ──────────────────────────────────────────────────
btnClear.addEventListener('click', async () => {
    try {
        await fetch('/clear', { method: 'POST' });

        if (selectedCard) {
            selectedCard.classList.remove('selected');
            selectedCard = null;
        }

        showToast('Accessory removed', 'success');
    } catch (err) {
        showToast('Failed to clear', 'error');
    }
});

// ── Snapshot ─────────────────────────────────────────────────────────
btnSnapshot.addEventListener('click', async () => {
    try {
        const res = await fetch('/snapshot');
        if (!res.ok) throw new Error('Snapshot failed');

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        snapshotImg.src = url;
        snapshotDownload.href = url;
        snapshotModal.style.display = 'flex';
    } catch (err) {
        showToast('Failed to capture snapshot', 'error');
    }
});

// Close snapshot modal
function closeSnapshotModal() {
    snapshotModal.style.display = 'none';
}

document.getElementById('modal-close').addEventListener('click', closeSnapshotModal);
document.getElementById('modal-close-btn').addEventListener('click', closeSnapshotModal);
snapshotModal.addEventListener('click', (e) => {
    if (e.target === snapshotModal) closeSnapshotModal();
});

// ── Face Shape Analysis ──────────────────────────────────────────────
btnFaceShape.addEventListener('click', async () => {
    faceShapeModal.style.display = 'flex';
    faceShapeResult.innerHTML = '<div class="spinner"></div><p>Analyzing your face shape…</p>';

    try {
        const res = await fetch('/face_shape', { method: 'POST' });
        const data = await res.json();

        if (!res.ok) {
            faceShapeResult.innerHTML = `<p style="color:var(--text-muted);">${data.error || 'Analysis failed'}</p>`;
            return;
        }

        const m = data.measurements || {};
        faceShapeResult.innerHTML = `
            <div class="face-shape-card">
                <div class="face-shape-badge">🎭 ${data.shape}</div>
                <div class="face-shape-suggestion">${data.suggestion}</div>
                <div class="face-shape-measurements">
                    <div class="measurement-item">
                        <div class="measurement-label">Face Width</div>
                        <div class="measurement-value">${m.face_width || '—'}</div>
                    </div>
                    <div class="measurement-item">
                        <div class="measurement-label">Face Height</div>
                        <div class="measurement-value">${m.face_height || '—'}</div>
                    </div>
                    <div class="measurement-item">
                        <div class="measurement-label">W/H Ratio</div>
                        <div class="measurement-value">${m.ratio || '—'}</div>
                    </div>
                    <div class="measurement-item">
                        <div class="measurement-label">Jawline</div>
                        <div class="measurement-value">${m.jawline_width || '—'}</div>
                    </div>
                </div>
            </div>`;
    } catch (err) {
        faceShapeResult.innerHTML = '<p style="color:var(--text-muted);">Network error</p>';
    }
});

// Close face shape modal
function closeFaceShapeModal() {
    faceShapeModal.style.display = 'none';
}

document.getElementById('face-shape-modal-close').addEventListener('click', closeFaceShapeModal);
faceShapeModal.addEventListener('click', (e) => {
    if (e.target === faceShapeModal) closeFaceShapeModal();
});

// ── File Upload ──────────────────────────────────────────────────────

// Drag and drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
});

function handleFileSelect(file) {
    // Validate PNG
    if (!file.name.toLowerCase().endsWith('.png')) {
        showToast('Only PNG files are accepted', 'error');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showToast('File too large (max 16 MB)', 'error');
        return;
    }

    uploadedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreviewImg.src = e.target.result;
        uploadPreview.style.display = 'flex';
    };
    reader.readAsDataURL(file);
}

// Apply uploaded accessory
btnApplyUpload.addEventListener('click', async () => {
    if (!uploadedFile) {
        showToast('No file selected', 'error');
        return;
    }

    const category = document.querySelector('input[name="upload-category"]:checked').value;

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('category', category);

    btnApplyUpload.textContent = 'Applying…';
    btnApplyUpload.disabled = true;

    try {
        const res = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (res.ok) {
            showToast('Custom accessory applied!', 'success');

            // Clear selection from catalogue
            if (selectedCard) {
                selectedCard.classList.remove('selected');
                selectedCard = null;
            }
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (err) {
        showToast('Upload failed', 'error');
    } finally {
        btnApplyUpload.textContent = 'Apply';
        btnApplyUpload.disabled = false;
    }
});

// ── Keyboard shortcuts ───────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSnapshotModal();
        closeFaceShapeModal();
    }
});

// ── Initialize ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadCatalogue();
});
