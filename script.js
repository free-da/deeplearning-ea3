let currentText = "";
let autoGenerate = false;
let autoInterval = null;

let trainData = [];
let devData = [];
let testData = [];

async function loadCSV(path) {
    const response = await fetch(path);
    const csvText = await response.text();
    const lines = csvText.trim().split("\n");
    const data = [];

    for (let line of lines) {
        const [token, label] = line.split(",");
        if (token && label) {
            data.push({ token, label });
        }
    }

    return data;
}

async function loadData() {
    trainData = await loadCSV("data/train.csv");
    devData   = await loadCSV("data/dev.csv");
    testData  = await loadCSV("data/test.csv");

    console.log("Train data sample:", trainData.slice(0, 10));
}

// Dummy-Vorhersagefunktion (zum Testen ohne Modell)
function predictNextWords(prompt) {
    return [
        { word: "Haus", prob: 0.3 },
        { word: "Auto", prob: 0.25 },
        { word: "Baum", prob: 0.15 },
        { word: "geht", prob: 0.1 },
        { word: "steht", prob: 0.08 }
    ];
}

function renderPredictions(predictions) {
    const container = document.getElementById("predictions");
    container.innerHTML = "";
    predictions.forEach(p => {
        const btn = document.createElement("button");
        btn.className = "button tiny";
        btn.innerText = `${p.word} (${(p.prob * 100).toFixed(1)}%)`;
        btn.onclick = () => {
            appendWord(p.word);
            triggerPrediction();
        };
        container.appendChild(btn);
    });
}

function updateCurrentText() {
    document.getElementById("current-text").innerText = currentText || "[Noch kein Text]";
}

function appendWord(word) {
    currentText = currentText.trim() + " " + word;
    updateCurrentText();
}

function triggerPrediction() {
    const predictions = predictNextWords(currentText.trim());
    renderPredictions(predictions);
}

function createModel(vocabSize, seqLength) {
    const model = tf.sequential();

    model.add(tf.layers.lstm({
        units: 100,
        returnSequences: true,
        inputShape: [seqLength, vocabSize]
    }));
    model.add(tf.layers.lstm({ units: 100 }));
    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    model.summary();
    return model;
}

// Initialisiere Buttons
function initEventHandlers() {
    document.getElementById("predict-btn").addEventListener("click", () => {
        const userInput = document.getElementById("user-input").value.trim();
        if (userInput) {
            currentText = userInput;
            updateCurrentText();
            triggerPrediction();
        }
    });

    document.getElementById("weiter-btn").addEventListener("click", () => {
        const predictions = predictNextWords(currentText.trim());
        if (predictions.length > 0) {
            appendWord(predictions[0].word);
            triggerPrediction();
        }
    });

    document.getElementById("auto-btn").addEventListener("click", () => {
        let count = 0;
        autoGenerate = true;
        autoInterval = setInterval(() => {
            if (!autoGenerate || count >= 10) {
                clearInterval(autoInterval);
                return;
            }
            const predictions = predictNextWords(currentText.trim());
            if (predictions.length > 0) {
                appendWord(predictions[0].word);
                triggerPrediction();
            }
            count++;
        }, 1000);
    });

    document.getElementById("stop-btn").addEventListener("click", () => {
        autoGenerate = false;
        clearInterval(autoInterval);
    });

    document.getElementById("reset-btn").addEventListener("click", () => {
        currentText = "";
        document.getElementById("user-input").value = "";
        document.getElementById("predictions").innerHTML = "";
        updateCurrentText();
        autoGenerate = false;
        clearInterval(autoInterval);
    });
}

// Hauptfunktion, wird beim Laden der Seite aufgerufen
async function main() {
    await loadData();
    initEventHandlers();
    updateCurrentText();
}

window.onload = main;
