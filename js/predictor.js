
import { padSequences, tokenizeText } from './tokenizer.js';

/**
 * Wrapper-Klasse für Vorhersagen
 */
export class Predictor {
    constructor(model, vocab, maxLen) {
        this.model = model;
        this.vocab = vocab;
        this.id2word = Object.entries(vocab).reduce((obj, [word, id]) => {
            obj[id] = word;
            return obj;
        }, {});
        this.maxLen = maxLen;
    }

    setModel(model, vocab, maxLen) {
        this.model = model;
        this.vocab = vocab;
        this.id2word = Object.entries(vocab).reduce((obj, [word, id]) => {
            obj[id] = word;
            return obj;
        }, {});
        this.maxLen = maxLen;
    }

    async predict(inputText, topK = 5) {
        const tokens = tokenizeText(inputText);
        const tokenIds = tokens.map(t => this.vocab[t] ?? this.vocab['<UNK>']); // Fallback
        const trimmed = tokenIds.slice(-this.maxLen);
        const padded = padSequences([trimmed], this.maxLen); // [1, maxLen]

        const input = tf.tensor2d(padded);
        console.log("Padded input shape:", padded[0].length);
        console.log("Padded tokens:", padded[0].map(id => this.id2word[id]));
        const prediction = this.model.predict(input);
        const probs = await prediction.data(); // statt .dataSync()
        await tf.nextFrame(); // Lässt den Browser Rendering/Events verarbeiten

        // Top-K Wahrscheinlichkeiten
        const sorted = [...probs.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, topK)
            .map(([id, prob]) => ({
                word: this.id2word[id] || '[UNK]',
                prob,
            }));
        console.log("🔮 Top-5 Vorhersagen:");
        console.table(sorted);  // zeigt schön formatierte Tabelle


        tf.dispose([input, prediction]);
        return sorted;
    }
    // async predictNextWord(inputText) {
    //     const predictions = await this.predict(inputText, 1);
    //     console.log(predictions);
    //     return predictions.length > 0 ? predictions[0].word : null;
    // }
    async predictNextWord(inputText) {
        const temperature = 1.0;
        const topK = 5;
        const predictions = await this.predict(inputText, topK);

        // Zeige Tabelle der Top-K-Wörter mit Wahrscheinlichkeiten
        console.table(predictions.map(p => ({
            word: p.word,
            prob: (p.prob * 100).toFixed(2) + '%'
        })));

        const sampled = this.sampleWithTemperature(predictions, temperature);
        console.log(`🎲 Gewählt: ${sampled.word} (p = ${(sampled.prob * 100).toFixed(2)}%)`);
        return sampled.word;
    }

    sampleWithTemperature(predictions, temperature = 1.0) {
        const logits = predictions.map(p => Math.log(p.prob + 1e-8)); // numerische Stabilität
        const scaled = logits.map(logit => logit / temperature);

        // Softmax
        const maxLogit = Math.max(...scaled);
        const exps = scaled.map(s => Math.exp(s - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sumExps);

        // Zufällige Auswahl
        const rand = Math.random();
        let cumSum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return predictions[i];
            }
        }
        return predictions[0]; // Fallback
    }

}
