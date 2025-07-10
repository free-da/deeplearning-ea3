export class DummyPredictor {
    predict(prompt) {
        return [
            { word: "Haus", prob: 0.3 },
            { word: "Auto", prob: 0.25 },
            { word: "Baum", prob: 0.15 },
            { word: "geht", prob: 0.1 },
            { word: "steht", prob: 0.08 }
        ];
    }
}
