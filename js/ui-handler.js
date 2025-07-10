export class UIHandler {
    constructor(predictor) {
        this.currentText = "";
        this.autoInterval = null;
        this.autoGenerate = false;
        this.predictor = predictor;
    }

    init() {
        document.getElementById("predict-btn").addEventListener("click", () => {
            const input = document.getElementById("user-input").value.trim();
            if (input) {
                this.currentText = input;
                this.updateCurrentText();
                this.triggerPrediction();
            }
        });

        document.getElementById("weiter-btn").addEventListener("click", () => {
            const predictions = this.predictor.predict(this.currentText.trim());
            if (predictions.length > 0) {
                this.appendWord(predictions[0].word);
                this.triggerPrediction();
            }
        });

        document.getElementById("auto-btn").addEventListener("click", () => {
            let count = 0;
            this.autoGenerate = true;
            this.autoInterval = setInterval(() => {
                if (!this.autoGenerate || count >= 10) {
                    clearInterval(this.autoInterval);
                    return;
                }
                const predictions = this.predictor.predict(this.currentText.trim());
                if (predictions.length > 0) {
                    this.appendWord(predictions[0].word);
                    this.triggerPrediction();
                }
                count++;
            }, 1000);
        });

        document.getElementById("stop-btn").addEventListener("click", () => {
            this.autoGenerate = false;
            clearInterval(this.autoInterval);
        });

        document.getElementById("reset-btn").addEventListener("click", () => {
            this.currentText = "";
            document.getElementById("user-input").value = "";
            document.getElementById("predictions").innerHTML = "";
            this.updateCurrentText();
            this.autoGenerate = false;
            clearInterval(this.autoInterval);
        });

        this.updateCurrentText();
    }

    updateCurrentText() {
        document.getElementById("current-text").innerText = this.currentText || "[Noch kein Text]";
    }

    appendWord(word) {
        this.currentText = this.currentText.trim() + " " + word;
        this.updateCurrentText();
    }

    triggerPrediction() {
        const predictions = this.predictor.predict(this.currentText.trim());
        this.renderPredictions(predictions);
    }

    renderPredictions(predictions) {
        const container = document.getElementById("predictions");
        container.innerHTML = "";
        predictions.forEach(p => {
            const btn = document.createElement("button");
            btn.className = "button tiny";
            btn.innerText = `${p.word} (${(p.prob * 100).toFixed(1)}%)`;
            btn.onclick = () => {
                this.appendWord(p.word);
                this.triggerPrediction();
            };
            container.appendChild(btn);
        });
    }
}
