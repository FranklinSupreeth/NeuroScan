/* ==========================================================
   NeuroScan — front-end logic
   ========================================================== */

const $ = (id) => document.getElementById(id);

const imageInput   = $("imageInput");
const dropArea     = $("dropArea");
const dropEmpty    = $("dropEmpty");
const previewWrap  = $("previewWrap");
const previewImage = $("previewImage");
const fileMeta     = $("fileMeta");
const predictBtn   = $("predictBtn");
const resetBtn     = $("resetBtn");

const resultEmpty = $("resultEmpty");
const resultBody  = $("resultBody");
const resultError = $("resultError");
const errorText   = $("errorText");
const resultTime  = $("resultTime");

const CLASS_COLORS = {
    glioma: "var(--c-glioma)",
    meningioma: "var(--c-meningioma)",
    notumor: "var(--c-notumor)",
    pituitary: "var(--c-pituitary)",
};

const GAUGE_LEN = 2 * Math.PI * 52; // circumference of the gauge circle
const ALLOWED = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"];
const MAX_BYTES = 10 * 1024 * 1024;

let selectedFile = null;
let selectedSample = null;

/* ---------------- File selection ---------------- */

dropArea.addEventListener("click", (e) => {
    if (dropArea.classList.contains("has-image")) return;
    if (e.target.tagName !== "LABEL") imageInput.click();
});

dropArea.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !dropArea.classList.contains("has-image")) {
        e.preventDefault();
        imageInput.click();
    }
});

imageInput.addEventListener("change", () => {
    if (imageInput.files.length) handleFile(imageInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
    dropArea.addEventListener(evt, (e) => {
        e.preventDefault();
        dropArea.classList.add("dragging");
    })
);

["dragleave", "drop"].forEach((evt) =>
    dropArea.addEventListener(evt, (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragging");
    })
);

dropArea.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED.includes(ext)) {
        showError("Unsupported file format. Use PNG, JPG, BMP or TIFF.");
        return;
    }
    if (file.size > MAX_BYTES) {
        showError("File too large. Maximum size is 10 MB.");
        return;
    }

    selectedFile = file;
    selectedSample = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        showPreview(e.target.result, `${file.name} · ${(file.size / 1024).toFixed(0)} KB`);
    };
    reader.readAsDataURL(file);
}

/* ---------------- Samples ---------------- */

document.querySelectorAll(".sample").forEach((btn) => {
    btn.addEventListener("click", () => {
        selectedSample = btn.dataset.sample;
        selectedFile = null;
        imageInput.value = "";
        showPreview(btn.querySelector("img").src, `sample · ${btn.textContent.trim()}`);
        analyze();
    });
});

/* ---------------- Preview / reset ---------------- */

function showPreview(src, meta) {
    previewImage.src = src;
    fileMeta.textContent = meta;
    dropEmpty.classList.add("hidden");
    previewWrap.classList.remove("hidden");
    dropArea.classList.add("has-image");
    resetBtn.classList.remove("hidden");
    predictBtn.disabled = false;
    setResultState("empty");
}

resetBtn.addEventListener("click", () => {
    selectedFile = null;
    selectedSample = null;
    imageInput.value = "";
    previewImage.removeAttribute("src");
    previewWrap.classList.add("hidden");
    dropEmpty.classList.remove("hidden");
    dropArea.classList.remove("has-image", "scanning");
    resetBtn.classList.add("hidden");
    predictBtn.disabled = true;
    setResultState("empty");
});

/* ---------------- Predict ---------------- */

predictBtn.addEventListener("click", analyze);

async function analyze() {
    if (!selectedFile && !selectedSample) return;

    const formData = new FormData();
    if (selectedSample) formData.append("sample", selectedSample);
    else formData.append("image", selectedFile);

    setLoading(true);
    const started = performance.now();

    try {
        const res = await fetch("/predict", { method: "POST", body: formData });
        let data;
        try {
            data = await res.json();
        } catch {
            throw new Error(`Server returned ${res.status}. Please try again.`);
        }

        if (!data.success) throw new Error(data.message || "Prediction failed.");

        // keep the scan animation visible for a moment so it feels deliberate
        const elapsed = performance.now() - started;
        if (elapsed < 700) await new Promise((r) => setTimeout(r, 700 - elapsed));

        showResult(data, performance.now() - started);
    } catch (err) {
        showError(err.message || "Prediction failed.");
        console.error(err);
    } finally {
        setLoading(false);
    }
}

function setLoading(on) {
    predictBtn.classList.toggle("loading", on);
    predictBtn.disabled = on;
    predictBtn.querySelector(".btn-label").textContent = on ? "Analyzing…" : "Analyze scan";
    dropArea.classList.toggle("scanning", on);
}

/* ---------------- Results ---------------- */

function setResultState(state) {
    resultEmpty.classList.toggle("hidden", state !== "empty");
    resultBody.classList.toggle("hidden", state !== "result");
    resultError.classList.toggle("hidden", state !== "error");
    if (state !== "result") resultTime.textContent = "";
}

function showError(msg) {
    errorText.textContent = msg;
    setResultState("error");
}

function showResult(data, ms) {
    const color = CLASS_COLORS[data.prediction] || "var(--accent)";
    const verdict = resultBody.querySelector(".verdict");
    verdict.style.setProperty("--c", color);

    $("prediction").textContent = data.label;
    $("summary").textContent = data.summary;
    $("verdictTag").textContent = data.prediction === "notumor" ? "Negative" : "Tumor detected";
    $("confidence").textContent = data.confidence.toFixed(1) + "%";
    resultTime.textContent = `${(ms / 1000).toFixed(2)} s`;

    // reset bars + gauge so the animation replays
    const gauge = $("gaugeFill");
    gauge.style.strokeDasharray = GAUGE_LEN;
    gauge.style.strokeDashoffset = GAUGE_LEN;
    document.querySelectorAll(".prob-fill").forEach((b) => (b.style.width = "0"));
    document.querySelectorAll(".prob-row").forEach((r) =>
        r.classList.toggle("top", r.dataset.class === data.prediction)
    );

    setResultState("result");

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            gauge.style.strokeDashoffset = GAUGE_LEN * (1 - data.confidence / 100);
            Object.entries(data.probabilities).forEach(([name, value]) => {
                const bar = $(name + "Bar");
                const val = $(name + "Value");
                if (bar) bar.style.width = value + "%";
                if (val) val.textContent = Number(value).toFixed(2) + "%";
            });
        });
    });

    if (window.innerWidth < 980) {
        resultBody.closest(".panel").scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
