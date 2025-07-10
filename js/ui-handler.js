export class UIHandler {
    constructor(predictor) {
        this.currentText = "";
        this.autoInterval = null;
        this.autoGenerate = false;
        this.predictor = predictor;
    }

    init() {
        document.getElementById("predict-btn").addEventListener("click", async () => {
            const input = document.getElementById("user-input").value.trim();
            if (input) {
                this.currentText = input;
                this.updateCurrentText();
                await this.triggerPrediction();
            }
        });

        document.getElementById("weiter-btn").addEventListener("click", async () => {
            const predictions = await this.predictor.predictNextWord(this.currentText.trim());
            if (predictions.length > 0) {
                this.appendWord(predictions[0]);
                await this.triggerPrediction();
            }
        });

        document.getElementById("auto-btn").addEventListener("click", () => {
            let count = 0;
            this.autoGenerate = true;
            this.autoInterval = setInterval(async () => {
                if (!this.autoGenerate || count >= 10) {
                    clearInterval(this.autoInterval);
                    return;
                }
                const predictions = await this.predictor.predictNextWord(this.currentText.trim());
                if (predictions.length > 0) {
                    this.appendWord(predictions[0]);
                    await this.triggerPrediction();
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

    async triggerPrediction() {
        const topWords = this.predictor.predict(this.currentText.trim(), 5); // Top-5
        const predictions = topWords.map(p => p.word); // Nur Wörter extrahieren

        console.log(this.predictor.predict("The cell produces", 5));
        console.log(this.predictor.predict("The metabolism is", 5));

        this.renderPredictions(predictions);
    }

    renderPredictions(predictedWords) {
        const container = document.getElementById("predictions");
        container.innerHTML = "";

        predictedWords.forEach(word => {
            const btn = document.createElement("button");
            btn.className = "button tiny";
            btn.innerText = word;
            btn.onclick = async () => {
                this.appendWord(word);
                await this.triggerPrediction();
            };
            container.appendChild(btn);
        });
    }
}
